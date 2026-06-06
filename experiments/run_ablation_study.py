"""
Ablation Study: Dual-Layer Continual Learning Framework
========================================================

This script runs ablation experiments to evaluate the contribution of each
of the 4 innovations:

  Variant 0: Full Model          (all 4 innovations enabled)
  Variant 1: w/o Elastic CB      (old_cb_lr_ratio = 1.0, i.e., no differential LR)
  Variant 2: w/o Adaptive Exp    (fixed expansion = 64, no collision-aware sizing)
  Variant 3: w/o Distillation    (distill_weight = 0, no KL divergence loss)
  Variant 4: w/o Replay          (replay_ratio = 0, no experience replay)
  Variant 5: Baseline            (all 4 innovations disabled)

Key features:
  - --full_ckpt_dir: reuse an existing Full Model checkpoint (skip re-training)
  - Evaluates both RQ-VAE metrics and recommendation metrics (NDCG@5/10, Recall@5/10)
  - Time-saving: ablation variants use fewer LLM epochs (default 30 vs 50 for full)
"""

import argparse
import copy
import json
import math
import os
import sys
import time
import logging

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator, PercentFormatter
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup — reuse modules from run_dual_layer_cl.py
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RQVAE_DIR = os.path.join(SCRIPT_DIR, "..", "Reformer-TIGER", "RQ-VAE")
LCREC_DIR = os.path.join(SCRIPT_DIR, "..", "Reformer-LC-Rec")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, LCREC_DIR)
sys.path.insert(0, RQVAE_DIR)  # Must be first so local datasets.py takes priority over HuggingFace datasets

# Import core functions from the main experiment script
from run_dual_layer_cl import (
    set_seed, ensure_dir,
    load_rqvae_from_ckpt,
    ElasticCodebookTrainer,
    compute_adaptive_expansion,
    train_rqvae_phase,
    generate_index_file,
    ReplayAwareSeqRecDataset,
    compute_collision_risk_scores,
    DistillationTrainer,
    train_llm_phase,
    collect_rqvae_metrics,
    evaluate_recommendation,
    TestSeqRecDataset,
    EmbDataset, new_EmbDataset, warm_TokenDataset, cold_TokenDataset,
)
from torch.utils.data import DataLoader


# =====================================================================
# Ablation Variant Definitions
# =====================================================================

ABLATION_VARIANTS = {
    "full": {
        "name": "Full Model",
        "short": "Full",
        "description": "All 4 innovations enabled",
        "old_cb_lr_ratio": 0.05,       # ① Elastic Codebook
        "use_adaptive_expansion": True, # ② Adaptive Expansion
        "distill_weight": 0.5,          # ③ Semantic Distillation
        "align_weight": 0.1,            # ③ Alignment loss
        "replay_ratio": 0.3,            # ④ Experience Replay
        "color": "#E74C3C",
        "marker": "s",
        "linestyle": "-",
    },
    "wo_elastic": {
        "name": "w/o ① Elastic CB",
        "short": "w/o Elastic",
        "description": "Old codebook uses same LR as new (no differential LR)",
        "old_cb_lr_ratio": 1.0,         # ① DISABLED: same LR for old and new
        "use_adaptive_expansion": True,  # ② enabled
        "distill_weight": 0.5,           # ③ enabled
        "align_weight": 0.1,
        "replay_ratio": 0.3,             # ④ enabled
        "color": "#3498DB",
        "marker": "o",
        "linestyle": "--",
    },
    "wo_adaptive": {
        "name": "w/o ② Adaptive Exp",
        "short": "w/o Adaptive",
        "description": "Fixed expansion size = 64 (no collision-aware sizing)",
        "old_cb_lr_ratio": 0.05,         # ① enabled
        "use_adaptive_expansion": False,  # ② DISABLED: fixed expansion
        "distill_weight": 0.5,            # ③ enabled
        "align_weight": 0.1,
        "replay_ratio": 0.3,              # ④ enabled
        "color": "#2ECC71",
        "marker": "^",
        "linestyle": "-.",
    },
    "wo_distill": {
        "name": "w/o ③ Distillation",
        "short": "w/o Distill",
        "description": "No cross-phase semantic distillation (KL loss = 0)",
        "old_cb_lr_ratio": 0.05,         # ① enabled
        "use_adaptive_expansion": True,   # ② enabled
        "distill_weight": 0.0,            # ③ DISABLED: no distillation
        "align_weight": 0.0,              # ③ DISABLED: no alignment
        "replay_ratio": 0.3,              # ④ enabled
        "color": "#9B59B6",
        "marker": "D",
        "linestyle": ":",
    },
    "wo_replay": {
        "name": "w/o ④ Replay",
        "short": "w/o Replay",
        "description": "No collision-aware experience replay",
        "old_cb_lr_ratio": 0.05,         # ① enabled
        "use_adaptive_expansion": True,   # ② enabled
        "distill_weight": 0.5,            # ③ enabled
        "align_weight": 0.1,
        "replay_ratio": 0.0,              # ④ DISABLED: no replay
        "color": "#F39C12",
        "marker": "v",
        "linestyle": "--",
    },
    "baseline": {
        "name": "Baseline (No Innov.)",
        "short": "Baseline",
        "description": "All 4 innovations disabled (vanilla continual learning)",
        "old_cb_lr_ratio": 1.0,          # ① DISABLED
        "use_adaptive_expansion": False,  # ② DISABLED
        "distill_weight": 0.0,            # ③ DISABLED
        "align_weight": 0.0,
        "replay_ratio": 0.0,              # ④ DISABLED
        "color": "#95A5A6",
        "marker": "x",
        "linestyle": "-",
    },
}


# =====================================================================
# Ablation Training Pipeline
# =====================================================================

def train_ablation_variant(args, variant_key, variant_cfg, device):
    """
    Train one ablation variant through all phases and evaluate recommendation metrics.

    For the 'full' variant, if --full_ckpt_dir is provided, skip training and
    directly load metrics from the existing checkpoint.
    """
    variant_name = variant_cfg["name"]
    variant_dir = os.path.join(args.output_dir, f"ablation_{variant_key}")

    # ── Special case: reuse existing Full Model checkpoint ──────────────────
    if variant_key == "full" and args.full_ckpt_dir:
        print(f"\n{'#' * 70}")
        print(f"  ABLATION VARIANT: {variant_name}  [REUSING EXISTING CHECKPOINT]")
        print(f"  Source: {args.full_ckpt_dir}")
        print(f"{'#' * 70}")
        # Symlink or copy the directory structure so downstream code can find it
        ensure_dir(variant_dir)
        full_rqvae_dir = os.path.join(args.full_ckpt_dir, "rqvae_ckpts")
        full_llm_dir = os.path.join(args.full_ckpt_dir, "llm_ckpts")
        full_index_dir = os.path.join(args.full_ckpt_dir, "indices")

        # Build training_log pointing to existing checkpoints
        training_log = []
        # Try to load cached eval results first
        eval_cache_path = os.path.join(args.full_ckpt_dir, "eval_results.json")
        cached_eval = {}
        if os.path.exists(eval_cache_path):
            with open(eval_cache_path, "r") as _f:
                _cached = json.load(_f)
            for _k, _v in _cached.items():
                if _k.startswith("phase"):
                    try:
                        cached_eval[int(_k.replace("phase", ""))] = _v
                    except ValueError:
                        pass
            if cached_eval:
                print(f"  [Skip] Loaded cached eval results from {eval_cache_path}")
        for phase in range(args.num_phases):
            llm_ckpt = os.path.join(full_llm_dir, f"phase{phase}")
            index_file = os.path.join(full_index_dir, f"phase{phase}",
                                      f"{args.dataset}.index.json")
            phase_log = {"phase": phase, "llm_ckpt": llm_ckpt,
                         "rqvae_ckpt_dir": full_rqvae_dir}
            if os.path.exists(llm_ckpt) and os.path.exists(index_file) and not args.skip_llm:
                if phase in cached_eval:
                    print(f"  [Skip] Phase {phase} eval already cached, skipping")
                    phase_log["eval"] = cached_eval[phase]
                else:
                    v_args = copy.deepcopy(args)
                    v_args.data_path = args.data_path
                    eval_results = evaluate_recommendation(
                        v_args, phase, llm_ckpt, index_file, device
                    )
                    phase_log["eval"] = eval_results
            training_log.append(phase_log)
        return training_log, full_rqvae_dir

    # ── Normal variant training ──────────────────────────────────────────────
    print(f"\n{'#' * 70}")
    print(f"  ABLATION VARIANT: {variant_name}")
    print(f"  Description: {variant_cfg['description']}")
    print(f"  Output: {variant_dir}")
    print(f"{'#' * 70}")
    print(f"  old_cb_lr_ratio:        {variant_cfg['old_cb_lr_ratio']}")
    print(f"  use_adaptive_expansion: {variant_cfg['use_adaptive_expansion']}")
    print(f"  distill_weight:         {variant_cfg['distill_weight']}")
    print(f"  align_weight:           {variant_cfg['align_weight']}")
    print(f"  replay_ratio:           {variant_cfg['replay_ratio']}")
    print()

    # Create variant-specific args
    v_args = copy.deepcopy(args)
    v_args.output_dir = variant_dir
    v_args.rqvae_ckpt_dir = os.path.join(variant_dir, "rqvae_ckpts")
    v_args.llm_ckpt_dir = os.path.join(variant_dir, "llm_ckpts")
    v_args.index_dir = os.path.join(variant_dir, "indices")
    v_args.old_cb_lr_ratio = variant_cfg["old_cb_lr_ratio"]
    v_args.distill_weight = variant_cfg["distill_weight"]
    v_args.align_weight = variant_cfg["align_weight"]
    v_args.replay_ratio = variant_cfg["replay_ratio"]
    # Ablation variants use fewer LLM epochs to save time
    if not hasattr(args, "ablation_llm_epochs"):
        v_args.llm_epochs = max(20, args.llm_epochs // 2)
    else:
        v_args.llm_epochs = args.ablation_llm_epochs

    ensure_dir(variant_dir)

    training_log = []
    prev_rqvae_ckpt = None
    prev_llm_ckpt = None

    for phase in range(args.num_phases):
        print(f"\n  --- {variant_name}: Phase {phase} ---")
        phase_log = {"phase": phase}

        # Step 1: RQ-VAE Training
        if not args.skip_rqvae:
            print(f"\n  === Step 1: RQ-VAE Training (Phase {phase}) ===")
            if variant_cfg["use_adaptive_expansion"]:
                prev_rqvae_ckpt = train_rqvae_phase(
                    v_args, phase, prev_rqvae_ckpt, device
                )
            else:
                prev_rqvae_ckpt = _train_rqvae_fixed_expansion(
                    v_args, phase, prev_rqvae_ckpt, device
                )
        else:
            prev_rqvae_ckpt = os.path.join(
                v_args.rqvae_ckpt_dir, f"phase{phase}",
                "freq_best_collision_model.pth"
            )

        # Step 2: Generate Index
        print(f"\n  === Step 2: Generate Index (Phase {phase}) ===")
        index_file = generate_index_file(
            prev_rqvae_ckpt, args.data_path, args.dataset,
            phase, device, v_args.index_dir
        )

        # Step 3: LLM Training
        if not args.skip_llm:
            llm_phase_ckpt = os.path.join(v_args.llm_ckpt_dir, f"phase{phase}")
            # Check for cached eval result
            variant_eval_cache = os.path.join(v_args.output_dir, "eval_results.json")
            _cached_variant_eval = {}
            if os.path.exists(variant_eval_cache):
                with open(variant_eval_cache, "r") as _f:
                    _tmp = json.load(_f)
                for _k, _v in _tmp.items():
                    if _k.startswith("phase"):
                        try:
                            _cached_variant_eval[int(_k.replace("phase", ""))] = _v
                        except ValueError:
                            pass

            if (os.path.exists(llm_phase_ckpt) and os.path.exists(
                os.path.join(llm_phase_ckpt, "adapter_model.safetensors")
            )) or (os.path.exists(llm_phase_ckpt) and os.path.exists(
                os.path.join(llm_phase_ckpt, "adapter_model.bin")
            )):
                print(f"  [Skip] Phase {phase} LLM ckpt exists at {llm_phase_ckpt}, skipping training")
                prev_llm_ckpt = llm_phase_ckpt
            else:
                print(f"\n  === Step 3: LLM Training (Phase {phase}) ===")
                prev_llm_ckpt = train_llm_phase(
                    v_args, phase, prev_llm_ckpt, index_file, device
                )
            phase_log["llm_ckpt"] = prev_llm_ckpt

            # Step 4: Evaluate recommendation metrics
            if phase in _cached_variant_eval:
                print(f"  [Skip] Phase {phase} eval already cached, skipping")
                phase_log["eval"] = _cached_variant_eval[phase]
            else:
                print(f"\n  === Step 4: Recommendation Evaluation (Phase {phase}) ===")
                eval_results = evaluate_recommendation(
                    v_args, phase, prev_llm_ckpt, index_file, device
                )
                phase_log["eval"] = eval_results

        training_log.append(phase_log)

    return training_log, v_args.rqvae_ckpt_dir


def _train_rqvae_fixed_expansion(args, phase, prev_ckpt, device):
    """
    Train RQ-VAE with FIXED expansion size = 64 (no adaptive expansion).
    This is used for the 'wo_adaptive' and 'baseline' variants.
    """
    from models.rqvae import RQVAE

    ckpt_dir = os.path.join(args.rqvae_ckpt_dir, f"phase{phase}")
    ckpt_path = os.path.join(ckpt_dir, "freq_best_collision_model.pth")

    if os.path.exists(ckpt_path):
        print(f"  [RQ-VAE] Phase {phase}: checkpoint exists, skipping")
        return ckpt_path

    ensure_dir(ckpt_dir)

    # Load data
    warm_data = []
    for p in range(phase):
        if p == 0:
            warm_data.append(EmbDataset(args.data_path, p, args.dataset))
        else:
            warm_data.append(new_EmbDataset(args.data_path, p - 1, args.dataset))

    data = EmbDataset(args.data_path, 0, args.dataset) if phase == 0 \
        else new_EmbDataset(args.data_path, phase - 1, args.dataset)

    # Fixed expansion = 64 (no adaptive computation)
    expansion = 64
    num_emb_list = [expansion, expansion, expansion]
    print(f"  [RQ-VAE] Phase {phase}: FIXED expansion={expansion}")

    train_args = argparse.Namespace(
        lr=1e-3, epochs=args.rqvae_epochs, batch_size=1024, eval_step=50,
        weight_decay=1e-4, dropout_prob=0.0, bn=False, loss_type="mse",
        init="kmeans", kmeans_iters=100, sk_epsilons=[0.0, 0.0, 0.003],
        sk_iters=50, affine_lr=0.0, affine_groups=1, replace_freq=0,
        quant_loss_weight=1.0, e_dim=32, layers=[2048, 1024, 512, 256, 128, 64],
        a=[0.5, 0.0, 0.0], new_a=[0.5, 0.0, 0.0],
        b=[0.0, 0.0, 0.0], b_scale=[0.0, 0.0, 0.0],
        freq_policy="pow", phase=phase, dataset=args.dataset,
        device=str(device), iso=0, seed=args.seed,
        num_emb_list=num_emb_list, ckpt_dir=ckpt_dir,
        push_freq=0, push_start=1000, push_lr=1e-4, data_path=args.data_path,
    )

    model = RQVAE(
        in_dim=data.dim, num_emb_list=num_emb_list,
        e_dim=32, layers=[2048, 1024, 512, 256, 128, 64],
        dropout_prob=0.0, bn=False, loss_type="mse",
        quant_loss_weight=1.0, init="kmeans", kmeans_iters=100,
        sk_epsilons=[0.0, 0.0, 0.003], sk_iters=50,
        affine_lr=0.0, affine_groups=1, replace_freq=0,
        a=[0.5, 0.0, 0.0], new_a=[0.5, 0.0, 0.0],
        b=[0.0, 0.0, 0.0], b_scale=[0.0, 0.0, 0.0],
        freq_policy="pow", warm_codebook=prev_ckpt,
        device=str(device), iso=0, seed=args.seed,
    )

    l = len(data)
    sep_r = np.argmin([abs(math.ceil(l / i) - 1024) for i in range(1, 11)])
    bs = math.ceil(l / (sep_r + 1))

    warm_loader = [DataLoader(d, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True,
                              prefetch_factor=2, persistent_workers=True)
                   for d in warm_data] if warm_data else None
    data_loader = DataLoader(data, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True,
                             prefetch_factor=4, persistent_workers=True)

    trainer = ElasticCodebookTrainer(
        model, train_args, device,
        old_cb_lr_ratio=args.old_cb_lr_ratio
    )
    trainer.fit(warm_loader, data_loader, ckpt_dir)

    return ckpt_path


# =====================================================================
# Metrics Collection for All Variants
# =====================================================================

def collect_all_ablation_metrics(args, variant_keys, training_logs, device):
    """Collect RQ-VAE metrics and recommendation metrics for all ablation variants."""
    all_metrics = {}

    for vk in variant_keys:
        variant_dir = os.path.join(args.output_dir, f"ablation_{vk}")

        # Determine rqvae_ckpt_dir: for full variant reusing existing ckpt
        if vk == "full" and args.full_ckpt_dir:
            rqvae_ckpt_dir = os.path.join(args.full_ckpt_dir, "rqvae_ckpts")
        else:
            rqvae_ckpt_dir = os.path.join(variant_dir, "rqvae_ckpts")

        if not os.path.exists(rqvae_ckpt_dir):
            print(f"  [WARN] Skipping {vk}: no RQ-VAE checkpoints found at {rqvae_ckpt_dir}")
            continue

        print(f"\n  Collecting RQ-VAE metrics for: {ABLATION_VARIANTS[vk]['name']}")
        rqvae_metrics = collect_rqvae_metrics(
            rqvae_ckpt_dir, args.data_path, args.dataset,
            args.num_phases, device
        )

        # Merge recommendation metrics from training_log
        rec_metrics = {}  # phase -> eval dict
        if vk in training_logs:
            for phase_log in training_logs[vk]:
                if "eval" in phase_log and isinstance(phase_log["eval"], dict):
                    p = phase_log["phase"]
                    rec_metrics[p] = phase_log["eval"]

        # Also try to load from cached eval results
        eval_cache = os.path.join(
            args.full_ckpt_dir if (vk == "full" and args.full_ckpt_dir) else variant_dir,
            "eval_results.json"
        )
        if os.path.exists(eval_cache) and not rec_metrics:
            with open(eval_cache, "r") as f:
                cached = json.load(f)
            for k, v in cached.items():
                if k.startswith("phase"):
                    try:
                        p = int(k.replace("phase", ""))
                        rec_metrics[p] = v
                    except ValueError:
                        pass

        # Build per-phase recommendation metric lists aligned with rqvae phases
        ndcg5_list, ndcg10_list, recall5_list, recall10_list = [], [], [], []
        for p in rqvae_metrics["phase"]:
            ev = rec_metrics.get(p, {})
            ndcg5_list.append(ev.get("NDCG@5", None))
            ndcg10_list.append(ev.get("NDCG@10", None))
            recall5_list.append(ev.get("Recall@5", None))
            recall10_list.append(ev.get("Recall@10", None))

        all_metrics[vk] = {
            **rqvae_metrics,
            "ndcg5": ndcg5_list,
            "ndcg10": ndcg10_list,
            "recall5": recall5_list,
            "recall10": recall10_list,
        }

    return all_metrics


# =====================================================================
# Visualization: Ablation Comparison Plots
# =====================================================================

def plot_ablation_rqvae(all_metrics, output_dir):
    """
    Figure 1: RQ-VAE layer ablation comparison (2×3 grid).
    Shows how each innovation contributes to RQ-VAE performance.
    """
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle(
        "Ablation Study: RQ-VAE Layer Performance\n"
        "Contribution of Each Innovation to Tokenization Quality",
        fontsize=16, fontweight="bold", y=0.98
    )

    metric_panels = [
        ("warm_recon_mse", "Warm Recon MSE", "(a) Reconstruction MSE ↓", axes[0, 0]),
        ("encoder_drift_cosine", "Cosine Sim to Phase 0", "(b) Encoder Stability ↑", axes[0, 1]),
        ("collision_rate_all", "Collision Rate", "(c) Overall Collision Rate ↓", axes[0, 2]),
        ("collision_rate_warm", "Warm Collision Rate", "(d) Warm Item Collision ↓", axes[1, 0]),
        ("collision_rate_cold", "Cold Collision Rate", "(e) Cold Item Collision ↓", axes[1, 1]),
        (None, "Avg Utilization", "(f) Codebook Utilization ↑", axes[1, 2]),  # special
    ]

    for metric_key, ylabel, title, ax in metric_panels:
        for vk, metrics in all_metrics.items():
            cfg = ABLATION_VARIANTS[vk]
            phases = metrics["phase"]
            if not phases:
                continue

            if metric_key is None:
                values = [np.mean(u) for u in metrics["utilization"]]
            else:
                values = metrics.get(metric_key, [])

            if values:
                ax.plot(phases, values,
                        color=cfg["color"], marker=cfg["marker"],
                        linestyle=cfg["linestyle"], linewidth=2.0,
                        markersize=7, label=cfg["short"])

        ax.set_xlabel("Phase")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if "collision" in (metric_key or ""):
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        if metric_key is None:
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.set_ylim(0, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(output_dir, "ablation_rqvae_metrics.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [Plot] Saved: {path}")


def plot_ablation_rec_metrics(all_metrics, output_dir):
    """
    Figure NEW: Recommendation metrics (NDCG@5/10, Recall@5/10) comparison
    across all ablation variants, per phase.
    """
    rec_metric_defs = [
        ("ndcg5",   "NDCG@5",   "(a) NDCG@5 ↑"),
        ("ndcg10",  "NDCG@10",  "(b) NDCG@10 ↑"),
        ("recall5", "Recall@5", "(c) Recall@5 ↑"),
        ("recall10","Recall@10","(d) Recall@10 ↑"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Ablation Study: Recommendation Performance\n"
        "NDCG@5/10 and Recall@5/10 Across Phases",
        fontsize=16, fontweight="bold", y=0.98
    )
    axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    for (metric_key, ylabel, title), ax in zip(rec_metric_defs, axes_flat):
        has_data = False
        for vk, metrics in all_metrics.items():
            cfg = ABLATION_VARIANTS[vk]
            phases = metrics.get("phase", [])
            values = metrics.get(metric_key, [])
            # Filter out None values
            valid = [(p, v) for p, v in zip(phases, values) if v is not None]
            if not valid:
                continue
            ps, vs = zip(*valid)
            ax.plot(list(ps), list(vs),
                    color=cfg["color"], marker=cfg["marker"],
                    linestyle=cfg["linestyle"], linewidth=2.0,
                    markersize=7, label=cfg["short"])
            has_data = True

        ax.set_xlabel("Phase")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if not has_data:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=12, color="gray")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(output_dir, "ablation_rec_metrics.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [Plot] Saved: {path}")


def plot_ablation_codebook_growth(all_metrics, output_dir):
    """
    Figure 2: Codebook size growth comparison.
    Shows how adaptive expansion affects codebook size.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Ablation Study: Codebook Size Growth\n"
        "Impact of Innovation ② Adaptive Expansion",
        fontsize=14, fontweight="bold"
    )

    for li, ax in enumerate(axes):
        for vk, metrics in all_metrics.items():
            cfg = ABLATION_VARIANTS[vk]
            phases = metrics["phase"]
            if not phases or not metrics["codebook_sizes"]:
                continue

            sizes = [s[li] if isinstance(s, list) and len(s) > li else 0
                     for s in metrics["codebook_sizes"]]
            ax.plot(phases, sizes,
                    color=cfg["color"], marker=cfg["marker"],
                    linestyle=cfg["linestyle"], linewidth=2.0,
                    markersize=7, label=cfg["short"])

        ax.set_xlabel("Phase")
        ax.set_ylabel("Codebook Size")
        ax.set_title(f"Layer {li}")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    path = os.path.join(output_dir, "ablation_codebook_growth.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [Plot] Saved: {path}")


def plot_ablation_radar(all_metrics, output_dir):
    """
    Figure 3: Radar chart comparing all variants at the last phase.
    """
    labels_list = []
    variant_values = {}

    for vk, metrics in all_metrics.items():
        if not metrics["phase"]:
            continue

        vals = []
        names = []

        if metrics["warm_recon_mse"]:
            mse = metrics["warm_recon_mse"][-1]
            names.append("Recon\nQuality")
            vals.append(max(0, 1 - mse * 100))

        if metrics["encoder_drift_cosine"]:
            names.append("Encoder\nStability")
            vals.append(metrics["encoder_drift_cosine"][-1])

        if metrics["collision_rate_all"]:
            names.append("Low\nCollision")
            vals.append(1 - metrics["collision_rate_all"][-1])

        if metrics["utilization"]:
            names.append("Codebook\nUtilization")
            vals.append(np.mean(metrics["utilization"][-1]))

        if metrics["codebook_sizes"]:
            size = metrics["codebook_sizes"][-1][0] if isinstance(
                metrics["codebook_sizes"][-1], list) else metrics["codebook_sizes"][-1]
            names.append("Size\nEfficiency")
            vals.append(max(0, 1 - (size - 64) / 256))

        # Add NDCG@10 as a radar dimension (normalized to [0,1])
        ndcg10_vals = [v for v in metrics.get("ndcg10", []) if v is not None]
        if ndcg10_vals:
            names.append("NDCG@10")
            # Normalize: assume max reasonable value is 0.05
            vals.append(min(1.0, ndcg10_vals[-1] / 0.05))

        if not labels_list:
            labels_list = names
        variant_values[vk] = vals

    if len(labels_list) < 3 or not variant_values:
        print("  [Plot] Skipping radar chart (insufficient data)")
        return

    N = len(labels_list)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.suptitle(
        "Ablation Study: Overall Comparison (Last Phase)\n"
        "Outer = Better",
        fontsize=14, fontweight="bold", y=1.02
    )

    for vk, vals in variant_values.items():
        cfg = ABLATION_VARIANTS[vk]
        plot_vals = vals + vals[:1]
        ax.plot(angles, plot_vals, f"{cfg['marker']}{cfg['linestyle']}",
                color=cfg["color"], linewidth=2.0, markersize=7,
                label=cfg["short"])
        ax.fill(angles, plot_vals, color=cfg["color"], alpha=0.05)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_list, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "ablation_radar.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [Plot] Saved: {path}")


def plot_ablation_delta_bars(all_metrics, output_dir):
    """
    Figure 4: Bar chart showing the performance delta when removing each innovation.
    """
    if "full" not in all_metrics or not all_metrics["full"]["phase"]:
        print("  [Plot] Skipping delta bars (no full model data)")
        return

    full = all_metrics["full"]
    ablation_keys = ["wo_elastic", "wo_adaptive", "wo_distill", "wo_replay"]
    innovation_names = [
        "① Elastic CB",
        "② Adaptive Exp",
        "③ Distillation",
        "④ Replay",
    ]

    metric_defs = [
        ("Warm MSE", "warm_recon_mse", False),
        ("Collision (All)", "collision_rate_all", False),
        ("Encoder Drift", "encoder_drift_cosine", True),
        ("NDCG@10", "ndcg10", True),
        ("Recall@10", "recall10", True),
    ]

    fig, axes = plt.subplots(1, len(metric_defs), figsize=(5 * len(metric_defs), 6))
    fig.suptitle(
        "Ablation Study: Impact of Removing Each Innovation\n"
        "Bars show performance degradation (↑ = innovation is more important)",
        fontsize=14, fontweight="bold"
    )

    for mi, (metric_name, metric_key, higher_better) in enumerate(metric_defs):
        ax = axes[mi]
        deltas = []
        colors = []

        for ai, ak in enumerate(ablation_keys):
            if ak not in all_metrics or not all_metrics[ak]["phase"]:
                deltas.append(0)
                colors.append("#CCCCCC")
                continue

            ablated = all_metrics[ak]

            full_vals = [v for v in full.get(metric_key, []) if v is not None]
            abl_vals = [v for v in ablated.get(metric_key, []) if v is not None]
            full_val = full_vals[-1] if full_vals else 0
            abl_val = abl_vals[-1] if abl_vals else 0

            if higher_better:
                delta = full_val - abl_val
            else:
                delta = abl_val - full_val

            deltas.append(delta)
            colors.append(ABLATION_VARIANTS[ak]["color"])

        x = np.arange(len(ablation_keys))
        bars = ax.bar(x, deltas, color=colors, alpha=0.85, edgecolor="black")

        for bar, delta in zip(bars, deltas):
            if delta != 0:
                fmt = f"{delta:.4f}" if abs(delta) < 0.01 else f"{delta:.3f}"
                va = "bottom" if delta >= 0 else "top"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        fmt, ha="center", va=va, fontsize=8, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(innovation_names, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel(f"Δ {metric_name}")
        ax.set_title(f"{metric_name}\n(↑ = innovation helps)")
        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(output_dir, "ablation_delta_bars.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [Plot] Saved: {path}")


def plot_ablation_heatmap(all_metrics, output_dir):
    """
    Figure 5: Heatmap of last-phase metrics across all variants.
    """
    variant_order = ["full", "wo_elastic", "wo_adaptive", "wo_distill", "wo_replay", "baseline"]
    metric_defs = [
        ("Warm MSE", "warm_recon_mse", False),
        ("Drift (Cos)", "encoder_drift_cosine", True),
        ("Collision (All)", "collision_rate_all", False),
        ("Collision (Warm)", "collision_rate_warm", False),
        ("Avg Util", None, True),
        ("NDCG@5", "ndcg5", True),
        ("NDCG@10", "ndcg10", True),
        ("Recall@5", "recall5", True),
        ("Recall@10", "recall10", True),
    ]

    matrix = []
    y_labels = []
    x_labels = []

    for vk in variant_order:
        if vk not in all_metrics or not all_metrics[vk]["phase"]:
            continue
        x_labels.append(ABLATION_VARIANTS[vk]["short"])

    if not x_labels:
        print("  [Plot] Skipping heatmap (no data)")
        return

    for metric_name, metric_key, higher_better in metric_defs:
        row = []
        for vk in variant_order:
            if vk not in all_metrics or not all_metrics[vk]["phase"]:
                continue
            m = all_metrics[vk]
            if metric_key is None:
                val = np.mean(m["utilization"][-1]) if m["utilization"] else 0
            else:
                vals = [v for v in m.get(metric_key, []) if v is not None]
                val = vals[-1] if vals else 0
            row.append(val)
        if row:
            matrix.append(row)
            y_labels.append(metric_name)

    if not matrix:
        print("  [Plot] Skipping heatmap (no data)")
        return

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(3 + len(x_labels) * 2, 2 + len(y_labels) * 1.2))
    fig.suptitle(
        "Ablation Study: Last-Phase Metrics Comparison\n"
        "Raw Values Across All Variants",
        fontsize=14, fontweight="bold"
    )

    im = ax.imshow(matrix, cmap="YlOrRd_r", aspect="auto")

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=10)

    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            val = matrix[i, j]
            if val < 0.01:
                text = f"{val:.6f}"
            elif val < 1:
                text = f"{val:.4f}"
            else:
                text = f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(output_dir, "ablation_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [Plot] Saved: {path}")


def plot_ablation_summary_table(all_metrics, output_dir):
    """
    Figure 6: Publication-quality summary table as image.
    Includes both RQ-VAE metrics and recommendation metrics.
    """
    variant_order = ["full", "wo_elastic", "wo_adaptive", "wo_distill", "wo_replay", "baseline"]
    metric_defs = [
        ("Warm MSE ↓",         "warm_recon_mse",        ".6f",  False),
        ("Drift (Cos) ↑",      "encoder_drift_cosine",  ".4f",  True),
        ("Collision (All) ↓",  "collision_rate_all",    ".4%",  False),
        ("Collision (Warm) ↓", "collision_rate_warm",   ".4%",  False),
        ("Collision (Cold) ↓", "collision_rate_cold",   ".4%",  False),
        ("Avg Utilization ↑",  None,                    ".2%",  True),
        ("CB Size (L0)",        "codebook_sizes",        "d",    False),
        ("NDCG@5 ↑",           "ndcg5",                 ".4f",  True),
        ("NDCG@10 ↑",          "ndcg10",                ".4f",  True),
        ("Recall@5 ↑",         "recall5",               ".4f",  True),
        ("Recall@10 ↑",        "recall10",              ".4f",  True),
    ]

    col_labels = ["Metric"]
    for vk in variant_order:
        if vk in all_metrics and all_metrics[vk]["phase"]:
            col_labels.append(ABLATION_VARIANTS[vk]["short"])

    active_variants = [vk for vk in variant_order
                       if vk in all_metrics and all_metrics[vk]["phase"]]

    if not active_variants:
        print("  [Plot] Skipping summary table (no data)")
        return

    cell_text = []
    cell_colors = []

    for metric_name, metric_key, fmt, higher_better in metric_defs:
        row_text = [metric_name]
        row_colors = ["#f0f0f0"]

        values = []
        for vk in active_variants:
            m = all_metrics[vk]
            if metric_key is None:
                val = np.mean(m["utilization"][-1]) if m["utilization"] else 0
            elif metric_key == "codebook_sizes":
                val = m["codebook_sizes"][-1][0] if m["codebook_sizes"] and isinstance(
                    m["codebook_sizes"][-1], list) else 0
            else:
                vals = [v for v in m.get(metric_key, []) if v is not None]
                val = vals[-1] if vals else 0
            values.append(val)

        if values:
            best_idx = np.argmax(values) if higher_better else np.argmin(values)
        else:
            best_idx = -1

        for vi, val in enumerate(values):
            if fmt == "d":
                text = f"{int(val)}"
            elif fmt.endswith("%"):
                text = f"{val:{fmt}}"
            else:
                text = f"{val:{fmt}}"
            row_text.append(text)

            if vi == best_idx:
                row_colors.append("#d4edda")  # Green for best
            elif active_variants[vi] == "full":
                row_colors.append("#d1ecf1")  # Blue for full model
            else:
                row_colors.append("white")

        cell_text.append(row_text)
        cell_colors.append(row_colors)

    n_cols = len(col_labels)
    n_rows = len(cell_text)
    fig, ax = plt.subplots(figsize=(3 * n_cols, 0.6 * n_rows + 2.5))
    ax.axis("off")
    fig.suptitle(
        "Ablation Study: Quantitative Comparison (Last Phase)\n"
        "Green = Best Value  |  Blue = Full Model",
        fontsize=14, fontweight="bold", y=0.98
    )

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=["#e0e0e0"] * n_cols,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for j in range(n_cols):
        table[0, j].set_text_props(fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(output_dir, "ablation_summary_table.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [Plot] Saved: {path}")


def plot_ablation_pairwise(all_metrics, output_dir):
    """
    Figure 7: Pairwise comparison — Full model vs each ablated variant.
    """
    if "full" not in all_metrics or not all_metrics["full"]["phase"]:
        print("  [Plot] Skipping pairwise plot (no full model data)")
        return

    full = all_metrics["full"]
    ablation_keys = ["wo_elastic", "wo_adaptive", "wo_distill", "wo_replay", "baseline"]

    metric_defs = [
        ("Warm MSE", "warm_recon_mse", False),
        ("Collision (All)", "collision_rate_all", False),
        ("Encoder Drift", "encoder_drift_cosine", True),
        ("NDCG@10", "ndcg10", True),
    ]

    fig, axes = plt.subplots(1, len(metric_defs), figsize=(7 * len(metric_defs), 6))
    fig.suptitle(
        "Ablation Study: Full Model Improvement Over Each Variant\n"
        "Positive = Full Model is Better",
        fontsize=14, fontweight="bold"
    )

    for mi, (metric_name, metric_key, higher_better) in enumerate(metric_defs):
        ax = axes[mi]

        for ak in ablation_keys:
            if ak not in all_metrics or not all_metrics[ak]["phase"]:
                continue

            cfg = ABLATION_VARIANTS[ak]
            ablated = all_metrics[ak]
            full_vals = full.get(metric_key, [])
            abl_vals = ablated.get(metric_key, [])
            n_phases = min(len(full["phase"]), len(ablated["phase"]))

            improvements = []
            phases = []
            for pi in range(n_phases):
                fv = full_vals[pi] if pi < len(full_vals) else None
                av = abl_vals[pi] if pi < len(abl_vals) else None
                if fv is None or av is None or av == 0:
                    improvements.append(0)
                elif higher_better:
                    improvements.append((fv - av) / abs(av) * 100)
                else:
                    improvements.append((av - fv) / abs(av) * 100)
                phases.append(pi)

            ax.plot(phases, improvements,
                    color=cfg["color"], marker=cfg["marker"],
                    linestyle=cfg["linestyle"], linewidth=2.0,
                    markersize=7, label=cfg["short"])

        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="-")
        ax.set_xlabel("Phase")
        ax.set_ylabel("Improvement %")
        ax.set_title(f"{metric_name}")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(output_dir, "ablation_pairwise.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  [Plot] Saved: {path}")


# =====================================================================
# Console Summary
# =====================================================================

def print_ablation_summary(all_metrics):
    """Print a concise ablation comparison table to console."""
    print(f"\n{'=' * 110}")
    print("  ABLATION STUDY — QUANTITATIVE COMPARISON (Last Phase)")
    print(f"{'=' * 110}")

    variant_order = ["full", "wo_elastic", "wo_adaptive", "wo_distill", "wo_replay", "baseline"]
    active = [vk for vk in variant_order if vk in all_metrics and all_metrics[vk]["phase"]]

    if not active:
        print("  No data available.")
        return

    header = f"  {'Metric':<28}"
    for vk in active:
        header += f" {ABLATION_VARIANTS[vk]['short']:>14}"
    print(header)
    print(f"  {'-' * (28 + 15 * len(active))}")

    def _get_val(m, key):
        if key is None:
            return np.mean(m["utilization"][-1]) if m["utilization"] else 0
        if key == "codebook_sizes":
            return m["codebook_sizes"][-1][0] if m["codebook_sizes"] and isinstance(
                m["codebook_sizes"][-1], list) else 0
        vals = [v for v in m.get(key, []) if v is not None]
        return vals[-1] if vals else 0

    def print_row(name, key, fmt, higher_better):
        row = f"  {name:<28}"
        values = [_get_val(all_metrics[vk], key) for vk in active]
        best_idx = np.argmax(values) if higher_better else np.argmin(values)
        for vi, val in enumerate(values):
            if fmt == "d":
                text = f"{int(val):>12}"
            elif fmt.endswith("%"):
                text = f"{val:>12{fmt}}"
            else:
                text = f"{val:>12{fmt}}"
            marker = " *" if vi == best_idx else "  "
            row += text + marker
        print(row)

    print_row("Warm MSE ↓",         "warm_recon_mse",       ".6f", False)
    print_row("Drift (Cos) ↑",      "encoder_drift_cosine", ".4f", True)
    print_row("Collision (All) ↓",  "collision_rate_all",   ".4%", False)
    print_row("Collision (Warm) ↓", "collision_rate_warm",  ".4%", False)
    print_row("Collision (Cold) ↓", "collision_rate_cold",  ".4%", False)
    print_row("Avg Utilization ↑",  None,                   ".2%", True)
    print_row("CB Size (L0)",        "codebook_sizes",       "d",   False)
    print(f"  {'-' * (28 + 15 * len(active))}")
    print_row("NDCG@5 ↑",           "ndcg5",                ".4f", True)
    print_row("NDCG@10 ↑",          "ndcg10",               ".4f", True)
    print_row("Recall@5 ↑",         "recall5",              ".4f", True)
    print_row("Recall@10 ↑",        "recall10",             ".4f", True)

    print(f"  {'-' * (28 + 15 * len(active))}")
    print("  (* = best value for that metric)")
    print(f"{'=' * 110}")


# =====================================================================
# Main
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Ablation Study for Dual-Layer CL Framework")

    # Data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="/root/models/Qwen/Qwen2___5-1___5B")
    parser.add_argument("--output_dir", type=str, default="./experiments/ablation_outputs")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_phases", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--eval_test_phase", type=int, default=0,
        help="Which phase test set to use for evaluation (default: 0)"
    )

    # Full model reuse: skip training 'full' variant, load from existing checkpoint
    parser.add_argument(
        "--full_ckpt_dir", type=str, default="",
        help="Path to existing Full Model output dir (e.g. /root/Reformer/experiments/outputs). "
             "If provided, the 'full' variant will NOT be re-trained; its checkpoints are "
             "loaded directly from this directory."
    )

    # RQ-VAE defaults (will be overridden per variant)
    parser.add_argument("--rqvae_epochs", type=int, default=20000)
    parser.add_argument("--collision_threshold", type=float, default=0.03)

    # LLM defaults for full model (ablation variants use ablation_llm_epochs)
    parser.add_argument("--llm_epochs", type=int, default=50)
    parser.add_argument("--llm_lr", type=float, default=2e-5)
    parser.add_argument("--llm_batch_size", type=int, default=12)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--distill_temperature", type=float, default=2.0)

    # Time-saving: ablation variants use fewer LLM epochs
    parser.add_argument(
        "--ablation_llm_epochs", type=int, default=15,
        help="LLM training epochs for ablation variants (default 15, less than full model's 50)"
    )

    # Control
    parser.add_argument("--variants", type=str, nargs="+",
                        default=["full", "wo_elastic", "wo_adaptive",
                                 "wo_distill", "wo_replay", "baseline"],
                        help="Which ablation variants to run")
    parser.add_argument("--skip_rqvae", action="store_true",
                        help="Skip RQ-VAE training (use existing checkpoints)")
    parser.add_argument("--skip_llm", action="store_true",
                        help="Skip LLM training")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip all training, only collect metrics and plot")
    parser.add_argument("--only_plot", action="store_true",
                        help="Only generate plots from existing results")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    device = torch.device(args.device)
    ensure_dir(args.output_dir)

    # Placeholder args that will be overridden per variant
    args.old_cb_lr_ratio = 0.05
    args.distill_weight = 0.5
    args.align_weight = 0.1
    args.replay_ratio = 0.3

    print("=" * 70)
    print("  Ablation Study: Dual-Layer Continual Learning Framework")
    print("=" * 70)
    print(f"  Dataset:              {args.dataset}")
    print(f"  Phases:               {args.num_phases}")
    print(f"  Base model:           {args.base_model}")
    print(f"  Variants:             {args.variants}")
    print(f"  Output:               {args.output_dir}")
    if args.full_ckpt_dir:
        print(f"  Full model ckpt:      {args.full_ckpt_dir}  [REUSE, no re-training]")
    print(f"  Ablation LLM epochs:  {args.ablation_llm_epochs}")
    print("=" * 70)

    # Validate variant names
    for vk in args.variants:
        if vk not in ABLATION_VARIANTS:
            print(f"  [ERROR] Unknown variant: {vk}")
            print(f"  Available: {list(ABLATION_VARIANTS.keys())}")
            sys.exit(1)

    # ===== Step 1: Train all variants =====
    training_logs = {}  # variant_key -> list of phase_logs

    if not args.skip_training and not args.only_plot:
        print(f"\n{'=' * 70}")
        print("  STEP 1: Training All Ablation Variants")
        print(f"{'=' * 70}")

        for vi, vk in enumerate(args.variants):
            cfg = ABLATION_VARIANTS[vk]
            print(f"\n  [{vi+1}/{len(args.variants)}] Training variant: {cfg['name']}")

            t0 = time.time()
            phase_logs, _ = train_ablation_variant(args, vk, cfg, device)
            training_logs[vk] = phase_logs
            t1 = time.time()

            print(f"\n  [{vk}] Completed in {(t1-t0)/60:.1f} minutes")

            # Save per-variant eval results immediately (crash recovery)
            variant_dir = (args.full_ckpt_dir if (vk == "full" and args.full_ckpt_dir)
                           else os.path.join(args.output_dir, f"ablation_{vk}"))
            ensure_dir(variant_dir)
            eval_cache = os.path.join(variant_dir, "eval_results.json")
            eval_data = {}
            for pl in phase_logs:
                if "eval" in pl and isinstance(pl["eval"], dict):
                    eval_data[f"phase{pl['phase']}"] = pl["eval"]
            if eval_data:
                def _convert(obj):
                    if isinstance(obj, (np.integer,)):
                        return int(obj)
                    if isinstance(obj, (np.floating,)):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj
                with open(eval_cache, "w") as f:
                    json.dump(json.loads(json.dumps(eval_data, default=_convert)), f, indent=2)
                print(f"  [Saved] Eval results: {eval_cache}")

    # ===== Step 2: Collect metrics =====
    print(f"\n{'=' * 70}")
    print("  STEP 2: Collecting Metrics for All Variants")
    print(f"{'=' * 70}")

    results_path = os.path.join(args.output_dir, "ablation_results.json")

    if args.only_plot and os.path.exists(results_path):
        print(f"  Loading cached results from {results_path}")
        with open(results_path, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = collect_all_ablation_metrics(
            args, args.variants, training_logs, device
        )

        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            return obj

        serializable = json.loads(json.dumps(all_metrics, default=convert))
        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n  Results saved to: {results_path}")

    # ===== Step 3: Generate visualizations =====
    print(f"\n{'=' * 70}")
    print("  STEP 3: Generating Ablation Visualizations")
    print(f"{'=' * 70}")

    plot_dir = os.path.join(args.output_dir, "plots")
    ensure_dir(plot_dir)

    plot_ablation_rqvae(all_metrics, plot_dir)
    plot_ablation_rec_metrics(all_metrics, plot_dir)
    plot_ablation_codebook_growth(all_metrics, plot_dir)
    plot_ablation_radar(all_metrics, plot_dir)
    plot_ablation_delta_bars(all_metrics, plot_dir)
    plot_ablation_heatmap(all_metrics, plot_dir)
    plot_ablation_summary_table(all_metrics, plot_dir)
    plot_ablation_pairwise(all_metrics, plot_dir)

    # ===== Console Summary =====
    print_ablation_summary(all_metrics)

    # ===== Final Output =====
    print(f"\n{'=' * 70}")
    print("  ABLATION STUDY COMPLETE")
    print(f"{'=' * 70}")
    print(f"\n  Output files:")
    print(f"    {results_path}")
    print(f"    {plot_dir}/ablation_rqvae_metrics.png    (6-panel RQ-VAE comparison)")
    print(f"    {plot_dir}/ablation_rec_metrics.png      (NDCG/Recall comparison)")
    print(f"    {plot_dir}/ablation_codebook_growth.png  (Per-layer CB size growth)")
    print(f"    {plot_dir}/ablation_radar.png            (Radar overview)")
    print(f"    {plot_dir}/ablation_delta_bars.png       (Innovation impact bars)")
    print(f"    {plot_dir}/ablation_heatmap.png          (Metrics heatmap)")
    print(f"    {plot_dir}/ablation_summary_table.png    (Quantitative table)")
    print(f"    {plot_dir}/ablation_pairwise.png         (Phase-wise improvement)")
    print()
    print("  To download to local Windows:")
    print(f"    scp -P 17991 -r root@connect.nmb2.seetacloud.com:{args.output_dir} "
          f"D:\\Reformer\\experiments\\")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

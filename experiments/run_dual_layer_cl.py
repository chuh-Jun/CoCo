"""
Dual-Layer Continual Learning for LLM-based Incremental Recommendation
=======================================================================

This script implements the COMPLETE framework with 4 innovations:

  Layer 1 - RQ-VAE (Tokenization Anti-Forgetting):
    ① Elastic Codebook Fine-tuning: differential LR for old codebook (5% of base)
    ② Collision-Aware Adaptive Expansion: dynamic codebook expansion based on collision rate

  Layer 2 - LLM (Recommendation Anti-Forgetting):
    ③ Cross-Phase Semantic Distillation: teacher-student KL divergence on old item tokens
    ④ Collision-Aware Experience Replay: priority replay of high-collision-risk old samples

The script provides:
  - End-to-end training pipeline (RQ-VAE → index generation → LLM training)
  - Comprehensive evaluation metrics
  - Publication-quality visualization

"""

import argparse
import collections
import copy
import json
import logging
import math
import os
import random
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup — try multiple candidate roots so the script works regardless
# of symlinks or the actual working directory on the server.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_ROOTS = [
    os.path.join(SCRIPT_DIR, ".."),                    # relative to script location
    os.getcwd(),                                        # current working directory
    os.environ.get("PROJECT_ROOT", "/root/Reformer"),   # from env or server default
    "/root/Reformer",                                   # server default (explicit)
]
# De-duplicate while preserving order
_seen = set()
_unique_roots = []
for _r in _CANDIDATE_ROOTS:
    _ra = os.path.realpath(_r)  # resolve symlinks
    if _ra not in _seen:
        _seen.add(_ra)
        _unique_roots.append(_r)
    # Also try the non-resolved version (in case symlink target differs)
    _ra2 = os.path.abspath(_r)
    if _ra2 not in _seen:
        _seen.add(_ra2)
        _unique_roots.append(_r)

RQVAE_DIR = None
LCREC_DIR = None
for _root in _unique_roots:
    _rq = os.path.join(_root, "Reformer-TIGER", "RQ-VAE")
    _lc = os.path.join(_root, "Reformer-LC-Rec")
    if os.path.isdir(_rq):
        RQVAE_DIR = os.path.abspath(_rq)
        LCREC_DIR = os.path.abspath(_lc)
        break
    # Also try with realpath (resolves symlinks)
    _rq_real = os.path.realpath(_rq)
    if os.path.isdir(_rq_real):
        RQVAE_DIR = _rq_real
        LCREC_DIR = os.path.realpath(_lc)
        break

if RQVAE_DIR is None:
    # Print diagnostic info before raising
    print("[PATH DEBUG] SCRIPT_DIR =", SCRIPT_DIR)
    print("[PATH DEBUG] os.getcwd() =", os.getcwd())
    print("[PATH DEBUG] PROJECT_ROOT env =", os.environ.get("PROJECT_ROOT", "<not set>"))
    for _root in _unique_roots:
        _rq = os.path.join(_root, "Reformer-TIGER", "RQ-VAE")
        print(f"[PATH DEBUG]   Tried: {_rq}  exists={os.path.exists(_rq)}  realpath={os.path.realpath(_rq)}")
    raise RuntimeError(
        "Cannot locate Reformer-TIGER/RQ-VAE under any candidate root. "
        "Please ensure the directory exists or set PROJECT_ROOT env var."
    )

sys.path.insert(0, RQVAE_DIR)
sys.path.insert(0, LCREC_DIR)


# ===================================================================
# Part 0: Utility Functions
# ===================================================================

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def resolve_local_path(path: str, label: str,
                       must_exist: bool = True,
                       expect_dir: bool = False) -> str:
    """Resolve a local path across common AutoDL mount prefixes."""
    if not path:
        return path

    candidates = []

    def add_candidate(candidate: str) -> None:
        if not candidate:
            return
        normalized = os.path.expanduser(candidate)
        for item in (normalized, os.path.abspath(normalized), os.path.realpath(normalized)):
            if item not in candidates:
                candidates.append(item)

    add_candidate(path)

    if path == "/root/autodl-fs" or path.startswith("/root/autodl-fs/"):
        add_candidate(path.replace("/root/autodl-fs", "/autodl-fs", 1))
    if path == "/autodl-fs" or path.startswith("/autodl-fs/"):
        add_candidate(path.replace("/autodl-fs", "/root/autodl-fs", 1))

    for candidate in candidates:
        exists = os.path.isdir(candidate) if expect_dir else os.path.exists(candidate)
        if exists:
            if candidate != path:
                logger.info("Resolved %s from '%s' to '%s'", label, path, candidate)
            return candidate

    if must_exist:
        raise FileNotFoundError(
            f"Cannot locate {label}: '{path}'. Tried: {candidates}"
        )
    return candidates[0] if candidates else path


# ===================================================================
# Part 1: RQ-VAE Layer — Elastic Codebook + Adaptive Expansion
# ===================================================================
# (Reuses the proven implementation from run_improved_experiment.py)

# ---------------------------------------------------------------------------
# Inline Dataset classes (originally from Reformer-TIGER/RQ-VAE/datasets.py)
# to avoid import conflicts with HuggingFace `datasets` package.
# ---------------------------------------------------------------------------

def find_emb_file(data_path: str, dataset: str) -> str:
    """Locate the embedding .npy file under data_path.

    Search order:
      1. <data_path>/<dataset>.emb-llama-td.npy  (exact match)
      2. <data_path>/<base>.emb-llama-td.npy     (strip version suffix, e.g. games_11111_0.1 -> games)
      3. Any *.emb-llama-td.npy found in data_path (first match)
    Raises FileNotFoundError if nothing is found.
    """
    import glob
    # 1. Exact match
    exact = os.path.join(data_path, f"{dataset}.emb-llama-td.npy")
    if os.path.exists(exact):
        return exact
    # 2. Strip version suffix (take the part before the first '_' that looks like a version)
    base = dataset.split("_")[0]
    base_path = os.path.join(data_path, f"{base}.emb-llama-td.npy")
    if os.path.exists(base_path):
        logger.info("Embedding file not found for '%s', falling back to '%s'", dataset, base_path)
        return base_path
    # 3. Glob fallback
    candidates = glob.glob(os.path.join(data_path, "*.emb-llama-td.npy"))
    if candidates:
        logger.info("Embedding file not found for '%s', using '%s'", dataset, candidates[0])
        return candidates[0]
    raise FileNotFoundError(
        f"Cannot find embedding file for dataset '{dataset}' in '{data_path}'. "
        f"Expected '{exact}' or any '*.emb-llama-td.npy'."
    )


def load_emb_array(data_path: str, dataset: str) -> np.ndarray:
    """Load embedding array from .npy file, handling object-array wrapping.

    Some .npy files store the actual float matrix inside a 0-d object array
    (e.g. saved via np.save with a dict or nested array).  This helper
    unwraps such cases and always returns a 2-D float32 ndarray of shape
    (N, D).
    """
    path = find_emb_file(data_path, dataset)
    arr = np.load(path, allow_pickle=True)

    # Unwrap 0-d object array: arr[()] gives the contained Python object
    if arr.ndim == 0:
        arr = arr[()]

    # If the contained object is a dict, try common keys
    if isinstance(arr, dict):
        for key in ("embeddings", "emb", "features", "data"):
            if key in arr:
                arr = arr[key]
                break
        else:
            # Fall back to the first value
            arr = next(iter(arr.values()))

    # Convert list-of-arrays to a stacked ndarray
    if isinstance(arr, (list, tuple)):
        arr = np.stack([np.asarray(x, dtype=np.float32) for x in arr])

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"Embedding file '{path}' has unexpected shape {arr.shape} after unwrapping. "
            f"Expected a 2-D array (N, D)."
        )
    return arr


class EmbDataset(Dataset):
    """Embedding dataset for warm items in a given phase."""
    def __init__(self, data_path, phase=None, dataset=None):
        self.data_path = data_path
        self.embeddings = load_emb_array(data_path, dataset)
        warm = list(np.load(data_path + "/phase%s" % phase + "/warm_item.npy", allow_pickle=True).tolist())
        all_idx = np.arange(self.embeddings.shape[0])
        save = list(filter(lambda x: x in warm, all_idx))
        self.embeddings = self.embeddings[save, :]
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        return torch.FloatTensor(self.embeddings[index])

    def __len__(self):
        return len(self.embeddings)


class new_EmbDataset(Dataset):
    """Embedding dataset for cold (new) items in a given phase."""
    def __init__(self, data_path, phase, dataset):
        self.data_path = data_path
        self.embeddings = load_emb_array(data_path, dataset)
        cold = list(np.load(data_path + "/phase%s" % phase + "/cold_item.npy", allow_pickle=True).tolist())
        all_idx = np.arange(self.embeddings.shape[0])
        save = list(filter(lambda x: x in cold, all_idx))
        self.embeddings = self.embeddings[save, :]
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        return torch.FloatTensor(self.embeddings[index])

    def __len__(self):
        return len(self.embeddings)


class warm_TokenDataset(Dataset):
    """Token dataset for warm items."""
    def __init__(self, data_path, phase, dataset):
        self.data_path = data_path
        self.embeddings = load_emb_array(data_path, dataset)
        warm = list(np.load(data_path + "/phase%s" % phase + "/warm_item.npy", allow_pickle=True).tolist())
        all_idx = np.arange(self.embeddings.shape[0])
        save = list(filter(lambda x: x in warm, all_idx))
        self.embeddings = self.embeddings[save, :]
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        return torch.FloatTensor(self.embeddings[index])

    def __len__(self):
        return len(self.embeddings)


class cold_TokenDataset(Dataset):
    """Token dataset for cold items."""
    def __init__(self, data_path, phase, dataset):
        self.data_path = data_path
        self.embeddings = load_emb_array(data_path, dataset)
        cold = list(np.load(data_path + "/phase%s" % phase + "/cold_item.npy", allow_pickle=True).tolist())
        all_idx = np.arange(self.embeddings.shape[0])
        save = list(filter(lambda x: x in cold, all_idx))
        self.embeddings = self.embeddings[save, :]
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        return torch.FloatTensor(self.embeddings[index])

    def __len__(self):
        return len(self.embeddings)


# Import RQVAE model (depends on sys.path containing RQVAE_DIR)
from models.rqvae import RQVAE


def load_rqvae_from_ckpt(ckpt_path, data_dim, device, phase=0):
    """Load an RQVAE model from checkpoint.

    Uses the checkpoint's own in_dim when available so that encoder/decoder
    dimensions always match the saved weights, regardless of the caller's
    data_dim (which may come from a different dataset split).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    # Infer in_dim from checkpoint encoder weights for guaranteed consistency
    enc_first_key = [k for k in state_dict.keys() if k.startswith("encoder.") and "weight" in k]
    if enc_first_key:
        ckpt_in_dim = state_dict[enc_first_key[0]].shape[1]  # first linear layer input dim
    else:
        ckpt_in_dim = data_dim
    if ckpt_in_dim != data_dim:
        print(f"  [WARN] load_rqvae_from_ckpt: caller data_dim={data_dim} != "
              f"ckpt encoder in_dim={ckpt_in_dim}, using ckpt value")
    in_dim = ckpt_in_dim

    emb_keys = [k for k in state_dict.keys() if "codebook" in k and "weight" in k]
    layer_info = {}
    for k in emb_keys:
        parts = k.split(".")
        layer_idx = int(parts[2])
        sub_idx = int(parts[4])
        size = state_dict[k].shape[0]
        if layer_idx not in layer_info:
            layer_info[layer_idx] = {}
        layer_info[layer_idx][sub_idx] = size

    num_emb_list = []
    for li in sorted(layer_info.keys()):
        sizes = [layer_info[li][si] for si in sorted(layer_info[li].keys())]
        num_emb_list.append(sizes)

    model = RQVAE(
        in_dim=in_dim,
        num_emb_list=num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        init=args.init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
        affine_lr=0.0,
        affine_groups=1,
        replace_freq=0,
        a=args.a,
        new_a=getattr(args, "new_a", args.a),
        b=args.b,
        b_scale=args.b_scale,
        freq_policy=args.freq_policy,
        device=str(device),
        iso=getattr(args, "iso", 0),
    )
    # Use strict=False to tolerate minor buffer mismatches across phases
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [WARN] load_rqvae_from_ckpt: missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"  [WARN] load_rqvae_from_ckpt: unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    model = model.to(device)
    model.eval()
    return model, args, ckpt


class ElasticCodebookTrainer:
    """
    Innovation ①: Elastic Codebook Fine-tuning
    Old codebook parameters use a reduced learning rate (default 5% of base lr).
    """

    def __init__(self, model, args, device, old_cb_lr_ratio=0.05):
        self.model = model
        self.args = args
        self.device = device
        self.old_cb_lr_ratio = old_cb_lr_ratio
        self.lr = getattr(args, 'lr', 1e-3)
        self.weight_decay = getattr(args, 'weight_decay', 1e-4)
        self.epochs = getattr(args, 'epochs', 20000)
        self.eval_step = min(getattr(args, 'eval_step', 50), self.epochs)
        self.p = args.phase

        self.best_loss = np.inf
        self.freq_best_collision_rate = np.inf

        self.optimizer = self._build_optimizer()
        self.model = self.model.to(self.device)

        self.freq = np.sum(args.a) or np.sum(args.b)
        self._freq = None
        self._dist_scale = None
        self.start_dist_scale = torch.ones(1).to(self.device)

    def _build_optimizer(self):
        old_cb_params, new_cb_params, other_params = [], [], []
        old_ids, new_ids = set(), set()

        for vq_layer in self.model.rq.vq_layers:
            n_sub = len(vq_layer.codebook)
            for idx, cb in enumerate(vq_layer.codebook):
                if n_sub > 1 and idx < n_sub - 1:
                    old_cb_params.append(cb.weight)
                    old_ids.add(id(cb.weight))
                else:
                    new_cb_params.append(cb.weight)
                    new_ids.add(id(cb.weight))

        for name, param in self.model.named_parameters():
            if id(param) not in old_ids and id(param) not in new_ids and param.requires_grad:
                other_params.append(param)

        groups = [
            {'params': other_params, 'lr': self.lr},
            {'params': new_cb_params, 'lr': self.lr},
        ]
        if old_cb_params:
            groups.append({'params': old_cb_params, 'lr': self.lr * self.old_cb_lr_ratio})
            print(f"  [Elastic] Old CB lr={self.lr * self.old_cb_lr_ratio:.6f} "
                  f"({self.old_cb_lr_ratio:.0%}), {len(old_cb_params)} param groups")

        return optim.AdamW(groups, weight_decay=self.weight_decay)

    def _warm_up(self, warm_data_loader):
        with torch.no_grad():
            for p, iter_data in enumerate(warm_data_loader):
                for data in iter_data:
                    data = data.to(self.device)
                    self.model(data, p=p, scale=self.start_dist_scale)

            self._freq = [
                copy.deepcopy(v._freq.detach().to(self.device))
                if hasattr(v, '_freq') else None
                for v in self.model.rq.vq_layers
            ]
            self._dist_scale = [
                copy.deepcopy(v._dist_scale.detach().to(self.device))
                if hasattr(v, '_freq') else None
                for v in self.model.rq.vq_layers
            ]

    def fit(self, warm_data_loader, data_loader, ckpt_dir):
        ensure_dir(ckpt_dir)
        ckpt_path = os.path.join(ckpt_dir, "freq_best_collision_model.pth")

        if warm_data_loader:
            self._warm_up(warm_data_loader)
        else:
            self._freq = [
                torch.ones(sum(cb.weight.shape[0] for cb in v.codebook)).to(self.device)
                if hasattr(v, '_freq') else None
                for v in self.model.rq.vq_layers
            ]
            self._dist_scale = [
                torch.ones(sum(cb.weight.shape[0] for cb in v.codebook)).to(self.device)
                if hasattr(v, '_freq') else None
                for v in self.model.rq.vq_layers
            ]

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for data in data_loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                out, rq_loss, indices = self.model(data, p=self.p)
                loss, loss_recon = self.model.compute_loss(out, rq_loss, xs=data)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                for vi, vq in enumerate(self.model.rq.vq_layers):
                    if hasattr(vq, '_freq') and self._freq[vi] is not None:
                        vq._freq = copy.deepcopy(self._freq[vi]).to(self.device)

            if total_loss < self.best_loss:
                self.best_loss = total_loss

            if (epoch + 1) % self.eval_step == 0:
                cr = self._eval_collision(data_loader)
                if cr < self.freq_best_collision_rate:
                    self.freq_best_collision_rate = cr
                    self._save(epoch, cr, ckpt_path)

            if (epoch + 1) % 500 == 0:
                logger.info(f"  epoch {epoch}: loss={total_loss:.4f}, "
                            f"best_collision={self.freq_best_collision_rate:.6f}")

        return self.best_loss, self.freq_best_collision_rate

    @torch.no_grad()
    def _eval_collision(self, loader):
        self.model.eval()
        codes, n = set(), 0
        for data in loader:
            n += len(data)
            data = data.to(self.device)
            idx = self.model.get_indices(data, p=self.p, scale=self.start_dist_scale)
            for row in idx.view(-1, idx.shape[-1]).cpu().numpy():
                codes.add("-".join(str(int(x)) for x in row))
        return (n - len(codes)) / n

    def _save(self, epoch, cr, path):
        state = {
            "args": self.args, "epoch": epoch,
            "best_loss": self.best_loss, "best_collision_rate": cr,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.freq > 0:
            state["freq"] = self._freq
            state["scale"] = self._dist_scale
        torch.save(state, path, pickle_protocol=4)


def compute_adaptive_expansion(model, data_loader, device, phase,
                               threshold=0.03, min_exp=4, max_exp=64,
                               expected_dim=None):
    """Innovation ②: Collision-Aware Adaptive Expansion."""
    if phase == 0:
        return 64

    model.eval()
    all_strs = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            # Adapt data dimension to match the model's encoder in_dim
            if expected_dim is not None and data.shape[-1] != expected_dim:
                if data.shape[-1] > expected_dim:
                    data = data[..., :expected_dim]
                else:
                    pad = torch.zeros(*data.shape[:-1], expected_dim - data.shape[-1], device=device)
                    data = torch.cat([data, pad], dim=-1)
            idx = model.get_indices(data, use_sk=False, p=phase - 1)
            for row in idx.view(-1, idx.shape[-1]).cpu().numpy():
                all_strs.append("-".join(str(int(x)) for x in row))

    total = len(all_strs)
    unique = len(set(all_strs))
    cr = (total - unique) / max(total, 1)
    print(f"  [Adaptive] Phase {phase}: collision={cr:.4%}")

    if cr <= threshold:
        return min_exp

    n_col = total - unique
    exp = int(np.ceil(n_col / 3 / 0.7))
    exp = max(min_exp, min(exp, max_exp))
    exp = int(np.ceil(exp / 4)) * 4
    print(f"  [Adaptive] expansion={exp} per layer")
    return exp


def train_rqvae_phase(args, phase, prev_ckpt, device):
    """Train one phase of improved RQ-VAE (Innovation ① + ②)."""
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

    # Use warm items dim as the canonical dimension (always available & consistent)
    canonical_dim = EmbDataset(args.data_path, 0, args.dataset).dim if phase > 0 else data.dim

    # Determine expansion
    if phase == 0:
        expansion = 64
    else:
        prev_model, _, prev_ckpt_data = load_rqvae_from_ckpt(prev_ckpt, canonical_dim, device, phase - 1)
        # Infer the actual in_dim that prev_model's encoder expects from its checkpoint weights
        prev_state = prev_ckpt_data["state_dict"]
        enc_keys = [k for k in prev_state.keys() if k.startswith("encoder.") and "weight" in k]
        prev_in_dim = prev_state[enc_keys[0]].shape[1] if enc_keys else canonical_dim
        all_warm = warm_TokenDataset(args.data_path, phase, args.dataset)
        loader = DataLoader(all_warm, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
        if prev_in_dim != all_warm.dim:
            print(f"  [WARN] train_rqvae_phase: prev_model in_dim={prev_in_dim} != "
                  f"current emb dim={all_warm.dim}; will pad/truncate data in adaptive expansion")
        expansion = compute_adaptive_expansion(prev_model, loader, device, phase,
                                               threshold=args.collision_threshold,
                                               expected_dim=prev_in_dim)
        del prev_model, all_warm, loader
        torch.cuda.empty_cache()

    num_emb_list = [expansion, expansion, expansion]
    print(f"  [RQ-VAE] Phase {phase}: expansion={expansion}")

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

    # Always use canonical_dim so encoder/decoder match the warm_codebook checkpoint
    model = RQVAE(
        in_dim=canonical_dim, num_emb_list=num_emb_list,
        e_dim=32, layers=[2048, 1024, 512, 256, 128, 64],
        dropout_prob=0.0, bn=False, loss_type="mse",
        quant_loss_weight=1.0, init="kmeans", kmeans_iters=100,
        sk_epsilons=[0.0, 0.0, 0.003], sk_iters=50,
        affine_lr=0.0, affine_groups=1, replace_freq=0,
        a=[0.5, 0.0, 0.0], new_a=[0.5, 0.0, 0.0],
        b=[0.0, 0.0, 0.0], b_scale=[0.0, 0.0, 0.0],
        freq_policy="pow", warm_codebook=prev_ckpt if phase > 0 else None,
        device=str(device), iso=0, seed=args.seed,
    )

    # Adjust batch size
    l = len(data)
    sep_r = np.argmin([abs(math.ceil(l / i) - 1024) for i in range(1, 11)])
    bs = math.ceil(l / (sep_r + 1))

    warm_loader = [DataLoader(d, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True,
                              prefetch_factor=2, persistent_workers=True)
                   for d in warm_data] if warm_data else None
    data_loader = DataLoader(data, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True,
                             prefetch_factor=4, persistent_workers=True)

    trainer = ElasticCodebookTrainer(model, train_args, device, old_cb_lr_ratio=args.old_cb_lr_ratio)
    trainer.fit(warm_loader, data_loader, ckpt_dir)

    return ckpt_path


def generate_index_file(rqvae_ckpt, data_path, dataset, phase, device, output_dir,
                        prev_index_file=None, freeze_old_tokens=True):
    """Generate item index JSON from trained RQ-VAE checkpoint.

    Innovation ⑤ — Token-Stable Codebook Inheritance (TSCI):
    When ``freeze_old_tokens=True`` and ``prev_index_file`` is given, warm items
    that already exist in the previous phase's index reuse their previous tokens
    instead of being re-encoded by the new RQ-VAE. Only newly-introduced cold
    items receive freshly-generated tokens. This prevents the catastrophic
    "100% token drift" problem in which every old item gets a brand-new symbol
    after each phase, severing the LLM's accumulated token-semantics knowledge.

    Args:
        prev_index_file: Path to the previous phase's ``*.index.json`` file
            (None for phase 0 or when freezing is disabled).
        freeze_old_tokens: If True, inherit old items' tokens from the previous
            phase's index. New tokens are still generated for items absent
            from the previous index (i.e. cold items).
    """
    index_path = os.path.join(output_dir, f"phase{phase}", f"{dataset}.index.json")
    if os.path.exists(index_path):
        print(f"  [Index] Phase {phase}: index file exists, skipping")
        return index_path

    ensure_dir(os.path.join(output_dir, f"phase{phase}"))

    # ★ TSCI: load previous phase index for token inheritance
    prev_indices = {}
    if freeze_old_tokens and prev_index_file and os.path.exists(prev_index_file):
        with open(prev_index_file, 'r') as f:
            prev_indices = json.load(f)
        print(f"  [Index] TSCI enabled: inheriting tokens from {prev_index_file} "
              f"({len(prev_indices)} items)")
    elif freeze_old_tokens and phase > 0:
        print(f"  [Index] TSCI requested but prev_index_file not found; "
              f"falling back to full re-encoding")

    warm_data = warm_TokenDataset(data_path, phase, dataset)
    warm_loader = DataLoader(warm_data, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    model, _, _ = load_rqvae_from_ckpt(rqvae_ckpt, warm_data.dim, device, phase)

    # Get warm item indices
    warm_items = np.load(os.path.join(data_path, f"phase{phase}", "warm_item.npy"),
                         allow_pickle=True).tolist()

    indices_dict = {}
    all_indices = []
    with torch.no_grad():
        for data in warm_loader:
            data = data.to(device)
            idx = model.get_indices(data, p=phase)
            all_indices.append(idx.cpu())

    all_indices = torch.cat(all_indices, 0).numpy()
    n_inherited = 0
    n_fresh = 0
    for i, item_id in enumerate(warm_items):
        key = str(item_id)
        # ★ TSCI: reuse previous tokens for items already known
        if freeze_old_tokens and key in prev_indices:
            indices_dict[key] = prev_indices[key]
            n_inherited += 1
        else:
            tokens = []
            for j in range(all_indices.shape[1]):
                tokens.append(f"<item_{j}_{int(all_indices[i, j])}>")
            indices_dict[key] = tokens
            n_fresh += 1

    # Cold items: always freshly encoded by current RQ-VAE
    n_cold = 0
    if phase > 0:
        cold_items = np.load(os.path.join(data_path, f"phase{phase}", "cold_item.npy"),
                             allow_pickle=True).tolist()
        cold_data = cold_TokenDataset(data_path, phase, dataset)
        cold_loader = DataLoader(cold_data, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

        cold_indices = []
        with torch.no_grad():
            for data in cold_loader:
                data = data.to(device)
                idx = model.get_indices(data, p=phase)
                cold_indices.append(idx.cpu())

        cold_indices = torch.cat(cold_indices, 0).numpy()
        for i, item_id in enumerate(cold_items):
            key = str(item_id)
            # Cold items may also already have tokens (rare but possible if
            # warm/cold sets overlap across phases); inherit if so.
            if freeze_old_tokens and key in prev_indices:
                indices_dict[key] = prev_indices[key]
                n_inherited += 1
            else:
                tokens = []
                for j in range(cold_indices.shape[1]):
                    tokens.append(f"<item_{j}_{int(cold_indices[i, j])}>")
                indices_dict[key] = tokens
                n_cold += 1

    with open(index_path, 'w') as f:
        json.dump(indices_dict, f, indent=2)

    print(f"  [Index] Phase {phase}: total={len(indices_dict)} "
          f"(inherited={n_inherited}, warm-fresh={n_fresh}, cold-fresh={n_cold}) "
          f"-> {index_path}")
    del model
    torch.cuda.empty_cache()
    return index_path


# ===================================================================
# Part 2: LLM Layer — Semantic Distillation + Experience Replay
# ===================================================================

class ReplayAwareSeqRecDataset(Dataset):
    """
    Innovation ④: Collision-Aware Experience Replay Dataset

    When ft=True, this dataset adds replay samples from old phases.
    Replay samples are selected based on collision risk from RQ-VAE.
    """

    def __init__(self, data_path, dataset, phase, index_file, indices,
                 max_his_len=20, add_prefix=False, subseq=True,
                 replay_ratio=0.3, collision_risk_scores=None,
                 special_token=""):
        self.data_path = data_path
        self.phase = phase
        self.indices = indices
        self.max_his_len = max_his_len
        self.add_prefix = add_prefix
        self.subseq = subseq
        self.replay_ratio = replay_ratio
        self.collision_risk_scores = collision_risk_scores or {}
        self.special_token = special_token
        self.prompt = "What would user be likely to purchase next after buying items {history} ?"

        self._load_data()
        self._remap_items()
        self.inter_data = self._process_train_data()

    def _load_data(self):
        self.warm_data = {}
        self.train_data = {}
        self.replay_data = {}

        # Load all old phase data into warm_data
        for p in range(self.phase):
            train = np.load(os.path.join(self.data_path, f"phase{p}", "training_dict.npy"),
                            allow_pickle=True).item()
            for uid in train:
                if uid not in self.warm_data:
                    self.warm_data[uid] = []
                self.warm_data[uid].extend(train[uid])

            valid = np.load(os.path.join(self.data_path, f"phase{p}", "validation_dict.npy"),
                            allow_pickle=True).item()
            for uid in valid:
                if uid not in self.warm_data:
                    self.warm_data[uid] = []
                if len(valid[uid]):
                    self.warm_data[uid].extend(valid[uid])

        # Current phase data
        train = np.load(os.path.join(self.data_path, f"phase{self.phase}", "training_dict.npy"),
                        allow_pickle=True).item()
        for uid in train:
            if uid not in self.train_data:
                self.train_data[uid] = []
            self.train_data[uid].extend(train[uid])

        if self.phase == 0:
            for uid in self.train_data:
                self.warm_data[uid] = []

        # ★ Innovation ④: Build replay data from old phases
        if self.phase > 0 and self.replay_ratio > 0:
            self._build_replay_data()

    def _build_replay_data(self):
        """Select old-phase samples for replay based on collision risk."""
        # Collect all old interactions with their item IDs
        old_interactions = []
        for uid, items in self.warm_data.items():
            for item_id in items:
                risk = self.collision_risk_scores.get(str(item_id), 0.5)
                old_interactions.append((uid, item_id, risk))

        if not old_interactions:
            return

        # Sort by collision risk (higher risk = more likely to be forgotten)
        old_interactions.sort(key=lambda x: x[2], reverse=True)

        # Select top replay_ratio fraction
        n_replay = max(1, int(len(old_interactions) * self.replay_ratio))
        selected = old_interactions[:n_replay]

        # Build replay sequences
        for uid, item_id, risk in selected:
            if uid not in self.replay_data:
                self.replay_data[uid] = []
            self.replay_data[uid].append(item_id)

        print(f"  [Replay] Phase {self.phase}: selected {n_replay} replay samples "
              f"from {len(old_interactions)} old interactions "
              f"(ratio={self.replay_ratio:.0%})")

    def _remap_items(self):
        def remap(data):
            result = {}
            for uid, items in data.items():
                mapped = []
                for i in items:
                    key = str(i)
                    if key in self.indices:
                        mapped.append("".join(self.indices[key]))
                result[uid] = mapped
            return result

        self.remapped_warm = remap(self.warm_data)
        self.remapped_train = remap(self.train_data)
        self.remapped_replay = remap(self.replay_data)

    def _process_train_data(self):
        inter_data = []

        # Process current phase data (standard)
        for uid in self.remapped_train:
            warm_items = self.remapped_warm.get(uid, [])
            items = self.remapped_train[uid]
            if len(items) >= 1:
                if self.subseq:
                    for i in range(len(items)):
                        if len(warm_items) >= 1 or i >= 1:
                            one = dict()
                            one["item"] = items[i]
                            history = warm_items + items[:i]
                            if self.max_his_len > 0:
                                history = history[-self.max_his_len:]
                            if self.add_prefix:
                                history = [f"{k+1}. {h}" for k, h in enumerate(history)]
                            one["inters"] = self.prompt.format(
                                history=",".join(history)) + self.special_token
                            one["is_replay"] = False
                            inter_data.append(one)
                else:
                    one = dict()
                    one["item"] = items[-1]
                    history = warm_items + items[:-1]
                    if self.max_his_len > 0:
                        history = history[-self.max_his_len:]
                    if self.add_prefix:
                        history = [f"{k+1}. {h}" for k, h in enumerate(history)]
                    one["inters"] = self.prompt.format(
                        history=",".join(history)) + self.special_token
                    one["is_replay"] = False
                    inter_data.append(one)

        # ★ Add replay samples (old items as prediction targets)
        for uid in self.remapped_replay:
            warm_items = self.remapped_warm.get(uid, [])
            replay_items = self.remapped_replay[uid]
            for item in replay_items:
                if item and warm_items:
                    one = dict()
                    one["item"] = item
                    history = warm_items[-self.max_his_len:] if self.max_his_len > 0 else warm_items
                    if self.add_prefix:
                        history = [f"{k+1}. {h}" for k, h in enumerate(history)]
                    one["inters"] = self.prompt.format(
                        history=",".join(history)) + self.special_token
                    one["is_replay"] = True
                    inter_data.append(one)

        n_new = sum(1 for d in inter_data if not d["is_replay"])
        n_replay = sum(1 for d in inter_data if d["is_replay"])
        print(f"  [Dataset] Phase {self.phase}: {n_new} new + {n_replay} replay = {len(inter_data)} total")

        return inter_data

    def get_new_tokens(self):
        tokens = set()
        for index in self.indices.values():
            for token in index:
                tokens.add(token)
        return sorted(list(tokens)) + ["|start_of_answer|"]

    def __len__(self):
        return len(self.inter_data)

    def __getitem__(self, index):
        d = self.inter_data[index]
        return dict(input_ids=d["inters"], labels=d["item"], is_replay=d["is_replay"])


def compute_collision_risk_scores(rqvae_ckpt, data_path, dataset, phase, device):
    """
    Compute per-item collision risk scores for experience replay selection.
    Items whose token codes collide with other items get higher risk scores.
    """
    if phase == 0:
        return {}

    warm_data = warm_TokenDataset(data_path, phase, dataset)
    warm_loader = DataLoader(warm_data, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    model, _, _ = load_rqvae_from_ckpt(rqvae_ckpt, warm_data.dim, device, phase)

    warm_items = np.load(os.path.join(data_path, f"phase{phase}", "warm_item.npy"),
                         allow_pickle=True).tolist()

    all_indices = []
    with torch.no_grad():
        for data in warm_loader:
            data = data.to(device)
            idx = model.get_indices(data, p=phase)
            all_indices.append(idx.cpu())

    all_indices = torch.cat(all_indices, 0).numpy()

    # Count code occurrences
    code_to_items = {}
    item_codes = {}
    for i, item_id in enumerate(warm_items):
        code = "-".join(str(int(x)) for x in all_indices[i])
        item_codes[str(item_id)] = code
        if code not in code_to_items:
            code_to_items[code] = []
        code_to_items[code].append(item_id)

    # Collision risk = number of items sharing the same code
    risk_scores = {}
    for item_id_str, code in item_codes.items():
        n_sharing = len(code_to_items[code])
        risk_scores[item_id_str] = (n_sharing - 1) / max(len(warm_items) - 1, 1)

    n_colliding = sum(1 for v in risk_scores.values() if v > 0)
    print(f"  [CollisionRisk] Phase {phase}: {n_colliding}/{len(risk_scores)} items have collision risk > 0")

    del model
    torch.cuda.empty_cache()
    return risk_scores


class DistillationTrainer:
    """
    Innovation ③: Cross-Phase Semantic Distillation

    Custom training loop that adds KL divergence loss between teacher (old phase LLM)
    and student (current phase LLM) on old item token predictions.

    L_total = L_rec + λ₁ · L_distill + λ₂ · L_align
    """

    def __init__(self, student_model, teacher_model, tokenizer, collator,
                 train_dataset, valid_dataset,
                 distill_weight=0.5, align_weight=0.1,
                 temperature=2.0, args=None,
                 distill_mode="kl", align_mode="full",
                 old_id_token_ids=None):
        self.student = student_model
        self.teacher = teacher_model
        self.tokenizer = tokenizer
        self.collator = collator
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.distill_weight = distill_weight
        self.align_weight = align_weight
        self.temperature = temperature
        self.args = args
        # Improvement #1 / #2: distillation + alignment behaviour switches.
        self.distill_mode = distill_mode
        self.align_mode = align_mode
        if old_id_token_ids is not None and not torch.is_tensor(old_id_token_ids):
            old_id_token_ids = torch.tensor(list(old_id_token_ids), dtype=torch.long)
        self.old_id_token_ids = old_id_token_ids

        # Freeze teacher
        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

    def compute_distillation_loss(self, student_logits, teacher_logits, labels, topk=1024):
        """
        Cross-phase semantic distillation loss.

        Two modes are supported (controlled by self.distill_mode):
          * "kl":       classic temperature-softened KL divergence on Top-K logits.
          * "listwise": ListMLE ranking distillation. The teacher's Top-K ordering is
                        treated as a pseudo permutation pi, and the student is asked
                        to maximise the conditional likelihood of producing pi.
                        This directly aligns with NDCG, which is the metric we care
                        about, and tends to give noticeably better Recall / NDCG than
                        plain KL on recommendation tasks.

        Only positions where labels != -100 contribute to the loss.
        """
        if teacher_logits is None:
            return torch.tensor(0.0, device=student_logits.device)

        mask = (labels != -100).float()
        if mask.sum() == 0:
            return torch.tensor(0.0, device=student_logits.device)

        # Top-K projection (shared by both modes) ------------------------
        vocab_size = teacher_logits.shape[-1]
        if topk < vocab_size:
            _, topk_indices = teacher_logits.topk(topk, dim=-1)               # [B,S,K]
            teacher_topk = teacher_logits.gather(-1, topk_indices)            # [B,S,K]
            student_topk = student_logits.gather(-1, topk_indices)            # [B,S,K]
        else:
            teacher_topk = teacher_logits
            student_topk = student_logits

        mode = getattr(self, "distill_mode", "kl")

        if mode == "listwise":
            # --- ListMLE on the teacher's Top-K ordering ----------------
            # 1. Use teacher logits to derive the target permutation.
            perm = teacher_topk.argsort(dim=-1, descending=True)              # [B,S,K]
            # 2. Reorder the student's logits accordingly.
            s_sorted = student_topk.gather(-1, perm)                          # [B,S,K]
            # 3. ListMLE: at each rank position i, normalise over items
            #    that come at position >= i (suffix log-sum-exp).
            suffix_lse = torch.logcumsumexp(s_sorted.flip(-1), dim=-1).flip(-1)  # [B,S,K]
            # NOTE: average over the K rank positions (not sum) so that the
            # listwise loss magnitude stays comparable to KL (~5-15) regardless
            # of K. Summing over K=256 makes the loss explode and dominate the
            # rec_loss, which causes catastrophic forgetting of the new phase.
            per_pos = (suffix_lse - s_sorted).mean(dim=-1)                    # [B,S]
            return (per_pos * mask).sum() / mask.sum()

        # --- Default: temperature-softened KL ---------------------------
        T = self.temperature
        teacher_probs = F.softmax(teacher_topk / T, dim=-1)
        student_log_probs = F.log_softmax(student_topk / T, dim=-1)
        kl = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=-1)
        kl_masked = (kl * mask).sum() / mask.sum()
        return kl_masked * (T ** 2)

    def compute_alignment_loss(self, student_embeds, teacher_embeds):
        """
        Embedding-space alignment loss between student and teacher.

        Two modes (controlled by self.align_mode):
          * "full":      L2 (MSE) over the entire overlapping vocabulary. This is the
                         original behaviour but it pins down the natural-language
                         embeddings of the LLM as well, which hurts Recall on the new
                         distribution.
          * "selective": cosine-distance loss applied ONLY on item-ID tokens that were
                         already registered in the previous phase. Natural-language
                         tokens are left fully trainable, so the LLM can keep adapting
                         to the new phase while old IDs stay in roughly the same
                         direction in embedding space.
        """
        if teacher_embeds is None:
            return torch.tensor(0.0, device=student_embeds.device)

        mode = getattr(self, "align_mode", "full")
        old_ids = getattr(self, "old_id_token_ids", None)

        if mode == "selective" and old_ids is not None and len(old_ids) > 0:
            ids = old_ids.to(student_embeds.device)
            # Stay safe in case ids were generated against a slightly different vocab.
            ids = ids[(ids >= 0) & (ids < min(student_embeds.shape[0], teacher_embeds.shape[0]))]
            if ids.numel() == 0:
                return torch.tensor(0.0, device=student_embeds.device)
            s = student_embeds.index_select(0, ids)
            t = teacher_embeds.index_select(0, ids).to(student_embeds.device).to(s.dtype)
            # 1 - cos: scale-invariant, only constrains direction.
            return (1.0 - F.cosine_similarity(s, t, dim=-1)).mean()

        # Fall back to the original full-vocab MSE behaviour.
        min_size = min(student_embeds.shape[0], teacher_embeds.shape[0])
        return F.mse_loss(student_embeds[:min_size], teacher_embeds[:min_size])

    def train(self, output_dir, epochs=50, lr=2e-5, batch_size=8,
              gradient_accumulation_steps=2, warmup_steps=100,
              eval_steps=None, save_steps=None):
        """Custom training loop with distillation."""
        ensure_dir(output_dir)

        # Use num_workers>0 to parallelize data loading & tokenization on CPU,
        # so GPU does not idle waiting for the next batch.
        num_workers = 4
        prefetch_factor = 4

        # ── Teacher logits pre-caching strategy ─────────────────────────
        # When teacher is available, we pre-cache its Top-K logits ONCE,
        # then free the teacher model to reclaim ~12GB VRAM.
        # This means we can use a LARGER batch_size for student training!
        # To ensure cache alignment, we disable shuffle (data is already
        # mixed with new + replay samples, so shuffle is not critical).
        cached_teacher_logits = None
        use_shuffle = True  # default

        # Get teacher embeddings for alignment loss BEFORE caching/freeing teacher
        teacher_embeds = None
        if self.teacher is not None:
            try:
                teacher_embeds = self.teacher.model.model.embed_tokens.weight.data.detach().clone().cpu()
            except AttributeError:
                try:
                    teacher_embeds = self.teacher.model.embed_tokens.weight.data.detach().clone().cpu()
                except AttributeError:
                    pass

        if self.teacher is not None and self.distill_weight > 0:
            # First, cache teacher logits with a non-shuffled loader
            cache_loader = DataLoader(
                self.train_dataset, batch_size=batch_size,
                shuffle=False, collate_fn=self.collator,
                num_workers=num_workers, pin_memory=True,
                prefetch_factor=prefetch_factor, persistent_workers=False
            )
            # Improvement #1: bump cached Top-K from 128 -> 256 to give the
            # listwise / KL distillation a richer ranking signal. Memory cost
            # stays small (2x) since we only keep Top-K logits + indices.
            # Tunable via --teacher_topk for hyperparameter analysis.
            default_topk = 256 if self.distill_mode == "listwise" else 128
            cache_topk = int(getattr(self.args, "teacher_topk", default_topk) or default_topk)
            print(f"  [Distillation] Pre-caching teacher logits "
                  f"(mode={self.distill_mode}, Top-K={cache_topk})...")
            sys.stdout.flush()
            cached_teacher_logits = self._cache_teacher_logits(
                cache_loader, topk=cache_topk
            )
            if cached_teacher_logits is not None:
                print(f"  [Distillation] Cached {len(cached_teacher_logits)} batches of teacher logits")
                # Free teacher model to reclaim ~12GB VRAM
                del self.teacher
                self.teacher = None
                torch.cuda.empty_cache()
                import gc; gc.collect()
                print(f"  [Distillation] Teacher model freed, VRAM reclaimed")
                # Disable shuffle so batch_idx matches cached logits
                use_shuffle = False
            else:
                print(f"  [Distillation] Teacher caching failed, falling back to no distillation")
            sys.stdout.flush()
            del cache_loader

        train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size,
            shuffle=use_shuffle, collate_fn=self.collator,
            num_workers=num_workers, pin_memory=True,
            prefetch_factor=prefetch_factor, persistent_workers=True
        )
        valid_loader = DataLoader(
            self.valid_dataset, batch_size=batch_size,
            shuffle=False, collate_fn=self.collator,
            num_workers=2, pin_memory=True,
            prefetch_factor=2, persistent_workers=True
        ) if self.valid_dataset else None

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.student.parameters()),
            lr=lr, weight_decay=0.01
        )

        total_steps = len(train_loader) * epochs // gradient_accumulation_steps
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

        if eval_steps is None:
            eval_steps = max(1, len(train_loader) // 5)
        if save_steps is None:
            save_steps = eval_steps

        best_val_loss = float('inf')
        global_step = 0
        patience_counter = 0
        max_patience = 10

        # Move teacher_embeds to device if available
        if teacher_embeds is not None:
            teacher_embeds = teacher_embeds.to(self.student.device)

        print(f"\n  [Distillation] Training with distill_weight={self.distill_weight}, "
              f"align_weight={self.align_weight}, temperature={self.temperature}")
        print(f"  [Distillation] {len(train_loader)} batches/epoch, {epochs} epochs, "
              f"total_steps={total_steps}")
        print(f"  [Distillation] DataLoader: num_workers={num_workers}, "
              f"prefetch_factor={prefetch_factor}, pin_memory=True")
        sys.stdout.flush()

        # Use AMP for mixed-precision forward passes (significant speedup on RTX 3090)
        use_amp = torch.cuda.is_available()
        amp_dtype = torch.bfloat16

        for epoch in range(epochs):
            self.student.train()
            epoch_loss = 0
            epoch_rec_loss = 0
            epoch_distill_loss = 0
            epoch_align_loss = 0
            n_batches = 0
            epoch_start_time = time.time()

            for batch_idx, batch in enumerate(train_loader):
                # Move to device (non_blocking for async CPU->GPU transfer)
                input_ids = batch["input_ids"].to(self.student.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.student.device, non_blocking=True)
                labels = batch["labels"].to(self.student.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    # Student forward
                    student_outputs = self.student(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    rec_loss = student_outputs.loss

                    # Distillation loss (using cached teacher logits)
                    distill_loss = torch.tensor(0.0, device=rec_loss.device)
                    if cached_teacher_logits is not None and batch_idx < len(cached_teacher_logits):
                        try:
                            cached = cached_teacher_logits[batch_idx]
                            t_topk_logits = cached["topk_logits"].to(self.student.device, non_blocking=True)
                            t_topk_indices = cached["topk_indices"].to(self.student.device, non_blocking=True)
                            t_labels = cached["labels"].to(self.student.device, non_blocking=True)

                            # Gather student logits at teacher's top-K positions
                            s_logits = student_outputs.logits
                            # Clamp indices to student vocab size
                            t_topk_indices_clamped = t_topk_indices.clamp(
                                max=s_logits.shape[-1] - 1
                            )
                            # Handle sequence length mismatch
                            min_seq = min(s_logits.shape[1], t_topk_indices_clamped.shape[1])
                            s_topk = s_logits[:, :min_seq, :].gather(
                                -1, t_topk_indices_clamped[:, :min_seq, :]
                            )
                            t_topk = t_topk_logits[:, :min_seq, :]
                            mask = (t_labels[:, :min_seq] != -100).float()
                            if mask.sum() > 0:
                                if self.distill_mode == "listwise":
                                    # Improvement #1: ListMLE ranking distillation.
                                    # Cast to fp32 for numerical stability of
                                    # logcumsumexp, which is sensitive to precision.
                                    s_fp = s_topk.float()
                                    t_fp = t_topk.float()
                                    perm = t_fp.argsort(dim=-1, descending=True)
                                    s_sorted = s_fp.gather(-1, perm)
                                    suffix_lse = torch.logcumsumexp(
                                        s_sorted.flip(-1), dim=-1
                                    ).flip(-1)
                                    # Average over K rank positions (not sum) to
                                    # keep listwise loss in a sane magnitude;
                                    # otherwise it dominates rec_loss at K=256.
                                    per_pos = (suffix_lse - s_sorted).mean(dim=-1)
                                    distill_loss = (per_pos * mask).sum() / mask.sum()
                                else:
                                    T = self.temperature
                                    teacher_probs = F.softmax(t_topk / T, dim=-1)
                                    student_log_probs = F.log_softmax(s_topk / T, dim=-1)
                                    kl = F.kl_div(
                                        student_log_probs, teacher_probs,
                                        reduction='none'
                                    ).sum(dim=-1)
                                    distill_loss = (kl * mask).sum() / mask.sum() * (T ** 2)
                        except Exception:
                            pass

                    # Alignment loss
                    align_loss = torch.tensor(0.0, device=rec_loss.device)
                    if teacher_embeds is not None and self.align_weight > 0:
                        try:
                            student_embeds = self.student.model.model.embed_tokens.modules_to_save.default.weight
                        except AttributeError:
                            try:
                                student_embeds = self.student.model.embed_tokens.weight
                            except AttributeError:
                                student_embeds = None

                        if student_embeds is not None:
                            align_loss = self.compute_alignment_loss(student_embeds, teacher_embeds)

                    # Total loss
                    total_loss = rec_loss + self.distill_weight * distill_loss + self.align_weight * align_loss
                    total_loss = total_loss / gradient_accumulation_steps

                total_loss.backward()

                epoch_loss += total_loss.item() * gradient_accumulation_steps
                epoch_rec_loss += rec_loss.item()
                epoch_distill_loss += distill_loss.item()
                epoch_align_loss += align_loss.item()
                n_batches += 1

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), 10.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            # Epoch summary
            avg_loss = epoch_loss / max(n_batches, 1)
            avg_rec = epoch_rec_loss / max(n_batches, 1)
            avg_dist = epoch_distill_loss / max(n_batches, 1)
            avg_align = epoch_align_loss / max(n_batches, 1)
            epoch_time = time.time() - epoch_start_time
            samples_per_sec = (n_batches * batch_size) / max(epoch_time, 1e-6)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  [Epoch {epoch+1}/{epochs}] total={avg_loss:.4f}, "
                      f"rec={avg_rec:.4f}, distill={avg_dist:.4f}, align={avg_align:.4f} "
                      f"| {epoch_time:.1f}s, {samples_per_sec:.1f} samples/s")
                sys.stdout.flush()

            # Validation
            if valid_loader and (epoch + 1) % 5 == 0:
                val_loss = self._validate(valid_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._save_model(output_dir)
                    print(f"  [Val] loss={val_loss:.4f} (best, saved)")
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        print(f"  [EarlyStop] No improvement for {max_patience} evals")
                        break

        # Final save if no validation
        if valid_loader is None:
            self._save_model(output_dir)

        return best_val_loss

    @torch.no_grad()
    def _cache_teacher_logits(self, train_loader, topk=128):
        """
        Pre-cache teacher's Top-K logits for all training batches.
        This runs teacher forward pass ONCE instead of every epoch,
        eliminating ~50% of GPU compute during training.
        Stores only Top-K indices and logits on CPU to save VRAM.
        """
        self.teacher.eval()
        cached = []
        use_amp = torch.cuda.is_available()
        amp_dtype = torch.bfloat16

        # Get teacher vocab size
        try:
            teacher_vocab_size = self.teacher.model.model.embed_tokens.modules_to_save.default.weight.shape[0]
        except AttributeError:
            try:
                teacher_vocab_size = self.teacher.get_input_embeddings().weight.shape[0]
            except AttributeError:
                return None

        cache_start = time.time()
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(self.teacher.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.teacher.device, non_blocking=True)
            labels = batch["labels"].to(self.teacher.device, non_blocking=True)

            teacher_input_ids = input_ids.clamp(max=teacher_vocab_size - 1)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                teacher_outputs = self.teacher(
                    input_ids=teacher_input_ids,
                    attention_mask=attention_mask,
                )
                teacher_logits = teacher_outputs.logits

            # Store only Top-K logits and indices on CPU
            topk_logits, topk_indices = teacher_logits.topk(topk, dim=-1)
            cached.append({
                "topk_logits": topk_logits.cpu(),
                "topk_indices": topk_indices.cpu(),
                "labels": labels.cpu(),
            })

            if (batch_idx + 1) % 500 == 0:
                elapsed = time.time() - cache_start
                print(f"    [Cache] {batch_idx+1}/{len(train_loader)} batches "
                      f"({elapsed:.1f}s)")
                sys.stdout.flush()

        elapsed = time.time() - cache_start
        print(f"  [Distillation] Teacher caching done in {elapsed:.1f}s")
        sys.stdout.flush()
        return cached

    @torch.no_grad()
    def _validate(self, loader):
        self.student.eval()
        total_loss = 0
        n = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(self.student.device)
            attention_mask = batch["attention_mask"].to(self.student.device)
            labels = batch["labels"].to(self.student.device)
            outputs = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            n += 1
        return total_loss / max(n, 1)

    def _save_model(self, output_dir):
        """Save LoRA adapter weights and tokenizer."""
        try:
            self.student.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
        except Exception as e:
            print(f"  [WARN] Save error: {e}")
            # Fallback: save state dict
            torch.save(self.student.state_dict(),
                       os.path.join(output_dir, "model_state_dict.pth"))


def train_llm_phase(args, phase, prev_llm_ckpt, index_file, device):
    """
    Train one phase of LLM with Innovation ③ + ④.

    This function:
    1. Loads the base LLM + LoRA
    2. If phase > 0, loads teacher model from previous phase
    3. Creates replay-aware dataset
    4. Trains with distillation loss
    """
    output_dir = os.path.join(args.llm_ckpt_dir, f"phase{phase}")
    adapter_bin = os.path.join(output_dir, "adapter_model.bin")
    adapter_safetensors = os.path.join(output_dir, "adapter_model.safetensors")

    if os.path.exists(adapter_bin) or os.path.exists(adapter_safetensors) or os.path.exists(os.path.join(output_dir, "model_state_dict.pth")):
        print(f"  [LLM] Phase {phase}: checkpoint exists, skipping")
        return output_dir

    ensure_dir(output_dir)

    print(f"\n  [LLM] Phase {phase}: Training with Distillation + Replay")

    # Lazy imports for transformers (heavy)
    from transformers import Qwen2Tokenizer, Qwen2ForCausalLM, Qwen2Config
    from peft import (TaskType, LoraConfig, get_peft_model,
                      prepare_model_for_kbit_training, set_peft_model_state_dict)

    # Load tokenizer
    if prev_llm_ckpt and os.path.exists(os.path.join(prev_llm_ckpt, "tokenizer.json")):
        tokenizer = Qwen2Tokenizer.from_pretrained(prev_llm_ckpt,
                                                    model_max_length=2048,
                                                    padding_side="left")
    else:
        tokenizer = Qwen2Tokenizer.from_pretrained(args.base_model,
                                                    model_max_length=2048,
                                                    padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load index
    with open(index_file, 'r') as f:
        indices = json.load(f)

    # Compute collision risk scores for replay
    collision_risks = {}
    if phase > 0:
        rqvae_ckpt = os.path.join(args.rqvae_ckpt_dir, f"phase{phase}",
                                   "freq_best_collision_model.pth")
        collision_risks = compute_collision_risk_scores(
            rqvae_ckpt, args.data_path, args.dataset, phase, device
        )

    # Build replay-aware dataset
    train_dataset = ReplayAwareSeqRecDataset(
        data_path=args.data_path, dataset=args.dataset,
        phase=phase, index_file=index_file, indices=indices,
        max_his_len=20, subseq=True,
        replay_ratio=args.replay_ratio if phase > 0 else 0.0,
        collision_risk_scores=collision_risks,
        special_token="|start_of_answer|"
    )

    # Add new tokens
    new_tokens = train_dataset.get_new_tokens()
    print(f"  [LLM] Before add tokens: {len(tokenizer)}")
    tokenizer.add_tokens(new_tokens)
    print(f"  [LLM] After add tokens: {len(tokenizer)}")

    # Build validation dataset (simple, no replay)
    valid_dataset = None  # Use training loss for simplicity

    # Collator
    from collator import Collator_DecoderOnly_manual
    collator_args = argparse.Namespace(only_train_response=True)
    collator = Collator_DecoderOnly_manual(collator_args, tokenizer)

    # Load student model
    # NOTE: Patch cached_file in both hub and peft modules to avoid HuggingFace
    # repo-id validation for local paths (peft.maybe_load_adapters imports cached_file directly).
    from transformers import BitsAndBytesConfig
    import transformers.utils.hub as _hub_utils_s
    import transformers.integrations.peft as _peft_utils_s
    _orig_cached_file_s = _hub_utils_s.cached_file

    def _patched_cached_file_s(path_or_repo_id, filename, **kwargs):
        if os.path.isdir(path_or_repo_id):
            full = os.path.join(path_or_repo_id, filename)
            return full if os.path.exists(full) else None
        return _orig_cached_file_s(path_or_repo_id, filename, **kwargs)

    dtype = torch.bfloat16
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    _hub_utils_s.cached_file = _patched_cached_file_s
    _peft_utils_s.cached_file = _patched_cached_file_s
    try:
        student = Qwen2ForCausalLM.from_pretrained(
            args.base_model, torch_dtype=dtype,
            quantization_config=quantization_config, device_map="auto",
        )
    finally:
        _hub_utils_s.cached_file = _orig_cached_file_s
        _peft_utils_s.cached_file = _orig_cached_file_s
    student.resize_token_embeddings(len(tokenizer))
    student = prepare_model_for_kbit_training(student)

    lora_config = LoraConfig(
        r=64, lora_alpha=128,
        target_modules=["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
        lora_dropout=0.05, bias="none",
        inference_mode=False, task_type=TaskType.CAUSAL_LM,
    )
    student = get_peft_model(student, lora_config)

    # Load previous LoRA weights if available
    prev_adapter_bin = os.path.join(prev_llm_ckpt, "adapter_model.bin") if prev_llm_ckpt else ""
    prev_adapter_safetensors = os.path.join(prev_llm_ckpt, "adapter_model.safetensors") if prev_llm_ckpt else ""
    if prev_llm_ckpt and (os.path.exists(prev_adapter_bin) or os.path.exists(prev_adapter_safetensors)):
        print(f"  [LLM] Loading previous LoRA from {prev_llm_ckpt}")
        if os.path.exists(prev_adapter_safetensors):
            from safetensors.torch import load_file
            adapters_weights = load_file(prev_adapter_safetensors)
        else:
            checkpoint_name = prev_adapter_bin
            adapters_weights = torch.load(checkpoint_name, map_location="cpu", weights_only=False)

        # For modules_to_save (embed_tokens, lm_head), the saved weights may have
        # smaller vocab size than current model. We need to replace them with
        # full-sized weights (old weights + random init for new tokens) so that
        # set_peft_model_state_dict can load them without dimension mismatch.
        for key in list(adapters_weights.keys()):
            if "embed_tokens" in key:
                old_weight = adapters_weights[key]
                new_weight = student.model.model.embed_tokens.modules_to_save.default.weight.data.clone().cpu()
                new_weight[:old_weight.shape[0]] = old_weight
                adapters_weights[key] = new_weight
            elif "lm_head" in key:
                old_weight = adapters_weights[key]
                new_weight = student.model.lm_head.modules_to_save.default.weight.data.clone().cpu()
                new_weight[:old_weight.shape[0]] = old_weight
                adapters_weights[key] = new_weight

        set_peft_model_state_dict(student, adapters_weights)
        print(f"  [LLM] Loaded LoRA weights from previous phase ({len(adapters_weights)} keys)")
        del adapters_weights

    # Freeze original modules
    for n, p in student.named_parameters():
        if "original_module" in n and any(m in n for m in ["embed_tokens", "lm_head"]):
            p.requires_grad = False

    student.print_trainable_parameters()

    # Load teacher model (previous phase LLM) for distillation
    teacher = None
    if phase > 0 and prev_llm_ckpt and args.distill_weight > 0:
        print(f"  [LLM] Loading teacher model from {prev_llm_ckpt}")
        try:
            import transformers.utils.hub as _hub_utils_t
            import transformers.integrations.peft as _peft_utils_t
            _orig_cached_file_t = _hub_utils_t.cached_file

            def _patched_cached_file_t(path_or_repo_id, filename, **kwargs):
                if os.path.isdir(path_or_repo_id):
                    full = os.path.join(path_or_repo_id, filename)
                    return full if os.path.exists(full) else None
                return _orig_cached_file_t(path_or_repo_id, filename, **kwargs)

            _hub_utils_t.cached_file = _patched_cached_file_t
            _peft_utils_t.cached_file = _patched_cached_file_t
            try:
                teacher = Qwen2ForCausalLM.from_pretrained(
                    args.base_model, torch_dtype=dtype,
                    quantization_config=quantization_config, device_map="auto",
                )
            finally:
                _hub_utils_t.cached_file = _orig_cached_file_t
                _peft_utils_t.cached_file = _orig_cached_file_t
            # Resize to previous tokenizer size
            prev_tokenizer = Qwen2Tokenizer.from_pretrained(prev_llm_ckpt)
            teacher.resize_token_embeddings(len(prev_tokenizer))
            teacher = prepare_model_for_kbit_training(teacher)
            teacher = get_peft_model(teacher, LoraConfig(
                r=64, lora_alpha=128,
                target_modules=["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
                modules_to_save=["embed_tokens", "lm_head"],
                lora_dropout=0.05, bias="none",
                inference_mode=True, task_type=TaskType.CAUSAL_LM,
            ))

            teacher_ckpt_bin = os.path.join(prev_llm_ckpt, "adapter_model.bin")
            teacher_ckpt_safetensors = os.path.join(prev_llm_ckpt, "adapter_model.safetensors")
            if os.path.exists(teacher_ckpt_safetensors):
                from safetensors.torch import load_file
                tw = load_file(teacher_ckpt_safetensors)
            elif os.path.exists(teacher_ckpt_bin):
                tw = torch.load(teacher_ckpt_bin, map_location="cpu", weights_only=False)
            else:
                tw = None
            if tw is not None:
                # For teacher, vocab size should match saved weights since we
                # resize to prev_tokenizer size. But handle dimension mismatch
                # just in case, same approach as student.
                for key in list(tw.keys()):
                    if "embed_tokens" in key:
                        old_weight = tw[key]
                        cur_weight = teacher.model.model.embed_tokens.modules_to_save.default.weight.data
                        if old_weight.shape[0] != cur_weight.shape[0]:
                            new_weight = cur_weight.clone().cpu()
                            new_weight[:old_weight.shape[0]] = old_weight
                            tw[key] = new_weight
                    elif "lm_head" in key:
                        old_weight = tw[key]
                        cur_weight = teacher.model.lm_head.modules_to_save.default.weight.data
                        if old_weight.shape[0] != cur_weight.shape[0]:
                            new_weight = cur_weight.clone().cpu()
                            new_weight[:old_weight.shape[0]] = old_weight
                            tw[key] = new_weight

                set_peft_model_state_dict(teacher, tw)
                del tw

            teacher.eval()
            print(f"  [LLM] Teacher model loaded successfully")
        except Exception as e:
            print(f"  [WARN] Failed to load teacher: {e}")
            teacher = None

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    # Improvement #2: collect the set of *old* item-ID token ids registered in
    # phase-(k-1)'s tokenizer. Selective alignment will only constrain these
    # token embeddings, leaving natural-language tokens fully trainable.
    old_id_token_ids = None
    if phase > 0 and prev_llm_ckpt is not None:
        try:
            prev_tok = Qwen2Tokenizer.from_pretrained(prev_llm_ckpt)
            base_tok_size = len(Qwen2Tokenizer.from_pretrained(args.base_model))
            # IDs that were added on top of the base vocab in earlier phases
            # (i.e. the <a_*>, <b_*>, <c_*> tokens already registered).
            id_range = list(range(base_tok_size, len(prev_tok)))
            # Map them through the current tokenizer to be safe.
            cur_ids = []
            for tid in id_range:
                tok_str = prev_tok.convert_ids_to_tokens(tid)
                cur_id = tokenizer.convert_tokens_to_ids(tok_str)
                if cur_id is not None and cur_id != tokenizer.unk_token_id:
                    cur_ids.append(cur_id)
            old_id_token_ids = cur_ids
            print(f"  [LLM] Selective-align: {len(cur_ids)} old ID tokens "
                  f"will be constrained (out of {len(tokenizer)} vocab).")
        except Exception as e:
            print(f"  [WARN] Could not build old_id_token_ids ({e}); "
                  f"selective alignment falls back to full alignment.")
            old_id_token_ids = None

    # Train with distillation
    distill_trainer = DistillationTrainer(
        student_model=student,
        teacher_model=teacher,
        tokenizer=tokenizer,
        collator=collator,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        distill_weight=args.distill_weight if phase > 0 else 0.0,
        align_weight=args.align_weight if phase > 0 else 0.0,
        temperature=args.distill_temperature,
        args=args,
        distill_mode=getattr(args, "distill_mode", "kl"),
        align_mode=getattr(args, "align_mode", "full"),
        old_id_token_ids=old_id_token_ids,
    )

    distill_trainer.train(
        output_dir=output_dir,
        epochs=args.llm_epochs,
        lr=args.llm_lr,
        batch_size=args.llm_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=100,
    )

    # Cleanup
    del student, teacher
    torch.cuda.empty_cache()

    return output_dir


# ===================================================================
# Part 3: Evaluation & Visualization
# ===================================================================


def resolve_eval_test_phase(phase, eval_test_phase):
    """Resolve which phase should provide the evaluation test set."""
    if eval_test_phase is None or eval_test_phase < 0:
        return phase
    if eval_test_phase > phase:
        raise ValueError(
            f"eval_test_phase ({eval_test_phase}) cannot exceed current phase ({phase})"
        )
    return eval_test_phase


class TestSeqRecDataset(Dataset):
    """Test dataset for beam search evaluation."""

    def __init__(self, data_path, dataset, phase, indices, max_his_len=20,
                 special_token="", eval_test_phase=-1):
        self.data_path = data_path
        self.phase = phase
        self.eval_test_phase = resolve_eval_test_phase(phase, eval_test_phase)
        self.indices = indices
        self.max_his_len = max_his_len
        self.special_token = special_token
        self.prompt = "What would user be likely to purchase next after buying items {history} ?"

        self._load_data()
        self._remap_items()
        self.test_data = self._process_test_data()

    def _load_data(self):
        self.warm_data = {}
        # Load history snapshot up to the evaluation test phase
        for p in range(self.eval_test_phase + 1):
            train = np.load(os.path.join(self.data_path, f"phase{p}", "training_dict.npy"),
                            allow_pickle=True).item()
            for uid in train:
                if uid not in self.warm_data:
                    self.warm_data[uid] = []
                self.warm_data[uid].extend(train[uid])

            valid = np.load(os.path.join(self.data_path, f"phase{p}", "validation_dict.npy"),
                            allow_pickle=True).item()
            for uid in valid:
                if uid not in self.warm_data:
                    self.warm_data[uid] = []
                if len(valid[uid]):
                    self.warm_data[uid].extend(valid[uid])

        # Load test data for the evaluation phase
        self.test_items = {}
        test = np.load(os.path.join(self.data_path, f"phase{self.eval_test_phase}", "testing_dict.npy"),
                       allow_pickle=True).item()
        for uid in test:
            if len(test[uid]) > 0:
                self.test_items[uid] = test[uid]

    def _remap_items(self):
        def remap(data):
            result = {}
            for uid, items in data.items():
                mapped = []
                for i in items:
                    key = str(i)
                    if key in self.indices:
                        mapped.append("".join(self.indices[key]))
                    else:
                        mapped.append(str(i))
                result[uid] = mapped
            return result
        self.remapped_warm = remap(self.warm_data)
        self.remapped_test = remap(self.test_items)

    def _process_test_data(self):
        test_data = []
        skipped = 0
        for uid in self.test_items:
            if uid not in self.remapped_warm or len(self.remapped_warm[uid]) == 0:
                skipped += 1
                continue
            history = self.remapped_warm[uid]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]
            # Only keep targets that have valid token indices
            raw_targets = self.test_items[uid]
            valid_targets = []
            for item_id in raw_targets:
                key = str(item_id)
                if key in self.indices:
                    valid_targets.append("".join(self.indices[key]))
            if not valid_targets:
                skipped += 1
                continue
            input_text = self.prompt.format(history=",".join(history)) + self.special_token
            test_data.append({
                "input_text": input_text,
                "target": valid_targets,
            })
        print(f"  [TestDataset] Train phase {self.phase}, test phase {self.eval_test_phase}: "
              f"{len(test_data)} test samples ({skipped} skipped due to no history or no valid targets)")
        return test_data

    def get_all_items(self):
        """Get item token sequences for constrained decoding from the evaluation phase."""
        items = set()
        phase_dir = os.path.join(self.data_path, f"phase{self.eval_test_phase}")
        warm_path = os.path.join(phase_dir, "warm_item.npy")
        cold_path = os.path.join(phase_dir, "cold_item.npy")
        phase_item_ids = set()
        if os.path.exists(warm_path):
            phase_item_ids.update(
                np.load(warm_path, allow_pickle=True).tolist()
            )
        if os.path.exists(cold_path):
            phase_item_ids.update(
                np.load(cold_path, allow_pickle=True).tolist()
            )
        for iid in phase_item_ids:
            key = str(iid)
            if key in self.indices:
                items.add("".join(self.indices[key]))
        return sorted(list(items))

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, index):
        return self.test_data[index]


class Trie(object):
    """Trie for constrained beam search decoding."""
    def __init__(self, sequences=None):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                self._add_to_trie(sequence, self.trie_dict)
                self.len += 1

    @staticmethod
    def _add_to_trie(sequence, trie_dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    def get(self, prefix_sequence):
        return Trie._get_from_trie(prefix_sequence, self.trie_dict)

    @staticmethod
    def _get_from_trie(prefix_sequence, trie_dict):
        if len(prefix_sequence) == 0:
            return list(trie_dict.keys())
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(prefix_sequence[1:], trie_dict[prefix_sequence[0]])
        else:
            return []


def compute_rec_metrics(ground_truth, predictions, topN_list):
    """Compute Hit@K, NDCG@K, Recall@K metrics."""
    results = {}
    for k in topN_list:
        hit_count = 0
        ndcg_sum = 0.0
        recall_sum = 0.0
        user_count = 0

        for i in range(len(predictions)):
            if len(ground_truth[i]) == 0:
                continue
            user_count += 1
            pred_k = predictions[i][:k]

            hit = any(p in ground_truth[i] for p in pred_k)
            hit_count += int(hit)

            dcg = sum(1.0 / math.log2(j + 2) for j, p in enumerate(pred_k) if p in ground_truth[i])
            idcg = sum(1.0 / math.log2(j + 2) for j in range(min(len(ground_truth[i]), k)))
            if idcg > 0:
                ndcg_sum += dcg / idcg

            hit_items = sum(1 for p in pred_k if p in ground_truth[i])
            recall_sum += hit_items / len(ground_truth[i])

        if user_count > 0:
            results[f"Hit@{k}"] = round(hit_count / user_count, 4)
            results[f"NDCG@{k}"] = round(ndcg_sum / user_count, 4)
            results[f"Recall@{k}"] = round(recall_sum / user_count, 4)
        else:
            results[f"Hit@{k}"] = 0.0
            results[f"NDCG@{k}"] = 0.0
            results[f"Recall@{k}"] = 0.0

    return results


def evaluate_recommendation(args, phase, llm_ckpt, index_file, device):
    """Evaluate recommendation performance using beam search with constrained decoding.

    Computes NDCG@5, NDCG@10, Recall@5, Recall@10 for the given phase.
    """
    eval_phase = resolve_eval_test_phase(phase, args.eval_test_phase)
    print(f"\n  [Eval] Train phase {phase}, test phase {eval_phase}: evaluating recommendation performance")

    from transformers import Qwen2Tokenizer, Qwen2ForCausalLM, BitsAndBytesConfig
    from peft import TaskType, LoraConfig, get_peft_model, set_peft_model_state_dict

    # Load tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(llm_ckpt,
                                                model_max_length=2048,
                                                padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load index
    with open(index_file, 'r') as f:
        indices = json.load(f)

    # Build test dataset
    test_dataset = TestSeqRecDataset(
        data_path=args.data_path, dataset=args.dataset,
        phase=phase, indices=indices, max_his_len=20,
        special_token="|start_of_answer|",
        eval_test_phase=eval_phase,
    )

    if len(test_dataset) == 0:
        print(f"  [Eval] Phase {phase}: no test data, skipping")
        return {"phase": phase, "note": "no test data"}

    # Load model
    # NOTE: Newer transformers versions call maybe_load_adapters() inside from_pretrained(),
    # which tries to validate the model path as a HuggingFace repo id and fails for local
    # absolute paths. We patch cached_file in transformers.utils.hub to skip the HF lookup
    # when the path_or_repo_id is a local directory.
    dtype = torch.bfloat16
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    import transformers.utils.hub as _hub_utils
    import transformers.integrations.peft as _peft_utils
    _orig_cached_file = _hub_utils.cached_file

    def _patched_cached_file(path_or_repo_id, filename, **kwargs):
        if os.path.isdir(path_or_repo_id):
            full = os.path.join(path_or_repo_id, filename)
            return full if os.path.exists(full) else None
        return _orig_cached_file(path_or_repo_id, filename, **kwargs)

    # Patch both the hub module and the peft integration module,
    # because peft.maybe_load_adapters imports cached_file directly.
    _hub_utils.cached_file = _patched_cached_file
    _peft_utils.cached_file = _patched_cached_file
    try:
        model = Qwen2ForCausalLM.from_pretrained(
            args.base_model, torch_dtype=dtype,
            quantization_config=quantization_config, device_map="auto",
        )
    finally:
        _hub_utils.cached_file = _orig_cached_file
        _peft_utils.cached_file = _orig_cached_file
    model.resize_token_embeddings(len(tokenizer))

    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=64, lora_alpha=128,
        target_modules=["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        modules_to_save=["embed_tokens", "lm_head"],
        lora_dropout=0.05, bias="none",
        inference_mode=True, task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # Load LoRA weights
    adapter_safetensors = os.path.join(llm_ckpt, "adapter_model.safetensors")
    adapter_bin = os.path.join(llm_ckpt, "adapter_model.bin")
    if os.path.exists(adapter_safetensors):
        from safetensors.torch import load_file
        adapters_weights = load_file(adapter_safetensors)
    elif os.path.exists(adapter_bin):
        adapters_weights = torch.load(adapter_bin, map_location="cpu", weights_only=False)
    else:
        print(f"  [Eval] Phase {phase}: no adapter weights found, skipping")
        del model
        torch.cuda.empty_cache()
        return {"phase": phase, "note": "no adapter weights"}

    # Handle dimension mismatch for embed_tokens / lm_head
    for key in list(adapters_weights.keys()):
        if "embed_tokens" in key:
            old_w = adapters_weights[key]
            cur_w = model.model.model.embed_tokens.modules_to_save.default.weight.data
            if old_w.shape[0] != cur_w.shape[0]:
                new_w = cur_w.clone().cpu()
                new_w[:old_w.shape[0]] = old_w
                adapters_weights[key] = new_w
        elif "lm_head" in key:
            old_w = adapters_weights[key]
            cur_w = model.model.lm_head.modules_to_save.default.weight.data
            if old_w.shape[0] != cur_w.shape[0]:
                new_w = cur_w.clone().cpu()
                new_w[:old_w.shape[0]] = old_w
                adapters_weights[key] = new_w

    set_peft_model_state_dict(model, adapters_weights)
    del adapters_weights
    model.eval()

    # Build constrained decoding trie
    # For decoder-only models (Qwen), the trie stores only the candidate token sequences
    # (without bos prefix). The prefix_allowed_tokens_fn strips the input prompt tokens
    # before querying the trie, and adds eos_token_id as a valid ending token.
    all_items = test_dataset.get_all_items()
    print(f"  [Eval] Building trie with {len(all_items)} candidate items")
    candidate_trie = Trie([
        tokenizer.encode(candidate) + [tokenizer.eos_token_id]
        for candidate in all_items
    ])

    # ── Improvement #5: prepare history-aware popularity prior for rerank ──
    # We collect, for every item-id-string, the number of times it appears in
    # any user's training/validation history of the eval phase. log(1+pop) is
    # then mixed into the beam score together with a TOKEN-level Jaccard
    # overlap (e.g. <a_12><b_5><c_30> -> {<a_12>,<b_5>,<c_30>}) against the
    # user's recent history. Both contributions can be turned off individually
    # via --rerank_alpha / --rerank_beta = 0.0.
    #
    # NOTE: defaults are 0.0 (rerank OFF) so that the baseline beam-search
    # behaviour is preserved. Set them explicitly to enable the rerank.
    rerank_alpha = float(getattr(args, "rerank_alpha", 0.0))
    rerank_beta = float(getattr(args, "rerank_beta", 0.0))
    # Token regex matches <a_12>, <b_5>, <c_30>, ... (item-ID atoms).
    _TOK_RE = re.compile(r"<[a-z]_\d+>")
    item_pop = {}
    if rerank_alpha > 0.0 or rerank_beta > 0.0:
        for uid, hist in test_dataset.remapped_warm.items():
            for s in hist:
                item_pop[s] = item_pop.get(s, 0) + 1
        print(f"  [Eval] Rerank enabled: alpha={rerank_alpha}, beta={rerank_beta}, "
              f"|item_pop|={len(item_pop)}")

    # Build a per-test-sample reference history for Jaccard reranking.
    # We re-derive it from remapped_warm because TestSeqRecDataset stores
    # only the prompt text, which is harder to parse back.
    sample_history = []
    if rerank_alpha > 0.0:
        # Iterate users in the same order test_dataset was built so indices align.
        for uid in test_dataset.test_items:
            if uid not in test_dataset.remapped_warm or len(test_dataset.remapped_warm[uid]) == 0:
                continue
            raw_targets = test_dataset.test_items[uid]
            if not any(str(i) in indices for i in raw_targets):
                continue
            hist = test_dataset.remapped_warm[uid]
            if test_dataset.max_his_len > 0:
                hist = hist[-test_dataset.max_his_len:]
            sample_history.append(hist)
        if len(sample_history) != len(test_dataset):
            # Fall back to disabling rerank entirely if alignment is off,
            # otherwise popularity prior would still pollute results.
            print(f"  [Eval] WARN: history alignment mismatch "
                  f"({len(sample_history)} vs {len(test_dataset)}); "
                  f"disabling rerank entirely.")
            sample_history = []
            rerank_alpha = 0.0
            rerank_beta = 0.0

    # Mutable container to hold current batch's input length
    _cur_input_len = [0]

    def prefix_allowed_tokens_fn(batch_id, sentence):
        # For decoder-only models, sentence = input_prompt_tokens + generated_tokens
        # Strip the input prompt part so we only match against the trie
        generated = sentence.tolist()[_cur_input_len[0]:]
        allowed = candidate_trie.get(generated)
        if not allowed:
            # If no continuation found, allow eos to terminate gracefully
            allowed = [tokenizer.eos_token_id]
        return allowed

    # Evaluate with beam search.
    # ``num_beams`` and ``eval_batch_size`` are exposed as CLI flags so they
    # can be swept during hyper-parameter analysis (e.g. {10,20,40} beams).
    num_beams = int(getattr(args, "num_beams", 20) or 20)
    all_pred_list = []
    all_gold_list = []

    eval_batch_size = int(getattr(args, "eval_batch_size", 4) or 4)
    use_amp = torch.cuda.is_available()
    amp_dtype = torch.bfloat16

    print(f"  [Eval] Running beam search (num_beams={num_beams}, "
          f"batch_size={eval_batch_size}, samples={len(test_dataset)})")
    sys.stdout.flush()

    eval_start = time.time()
    for start_idx in range(0, len(test_dataset), eval_batch_size):
        end_idx = min(start_idx + eval_batch_size, len(test_dataset))
        batch = [test_dataset[i] for i in range(start_idx, end_idx)]

        input_texts = [b["input_text"] for b in batch]
        targets = [b["target"] for b in batch]

        inputs = tokenizer(
            input_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=2048
        ).to(model.device)

        # Update input length for prefix_allowed_tokens_fn
        _cur_input_len[0] = inputs["input_ids"].shape[1]

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            output = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=10,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                output_scores=True,
                return_dict_in_generate=True,
                early_stopping=True,
            )

        output_ids = output["sequences"]
        scores = output["sequences_scores"].cpu().tolist()
        # Decode only the generated part (after input)
        input_len = inputs["input_ids"].shape[1]
        gen_ids = output_ids[:, input_len:]
        decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        decoded = [s.strip().replace(" ", "") for s in decoded]

        B = len(targets)
        for b in range(B):
            batch_seqs = decoded[b * num_beams: (b + 1) * num_beams]
            batch_scores = scores[b * num_beams: (b + 1) * num_beams]

            # ── Improvement #5: history + popularity rerank ──
            # IMPORTANT: Jaccard is computed on TOKEN sets parsed by _TOK_RE,
            # NOT on raw character sets. Char-level overlap is meaningless for
            # ID strings like "<a_12><b_5><c_30>" because every candidate
            # shares the punctuation/digit characters and the resulting score
            # is essentially noise that overrides the LLM's own ranking.
            global_idx = start_idx + b
            if (rerank_alpha > 0.0 or rerank_beta > 0.0) and len(batch_seqs) > 0:
                hist = sample_history[global_idx] if (rerank_alpha > 0.0
                                                       and global_idx < len(sample_history)) else []
                hist_tok_sets = [set(_TOK_RE.findall(h)) for h in hist[-5:]]
                hist_tok_sets = [s for s in hist_tok_sets if len(s) > 0]

                # Optional: only the top-K beams (by LLM score) participate in
                # rerank. Remaining beams keep their original order and are
                # appended after the reranked ones. K=0 means rerank-all.
                rerank_topk = int(getattr(args, "rerank_topk", 0) or 0)
                paired = list(zip(batch_seqs, batch_scores))
                if rerank_topk > 0 and rerank_topk < len(paired):
                    paired_sorted = sorted(paired, key=lambda x: x[1], reverse=True)
                    head = paired_sorted[:rerank_topk]
                    tail = paired_sorted[rerank_topk:]
                else:
                    head = paired
                    tail = []

                rescored = []
                for seq, sc in head:
                    bonus = 0.0
                    if rerank_alpha > 0.0 and len(hist_tok_sets) > 0:
                        seq_toks = set(_TOK_RE.findall(seq))
                        if len(seq_toks) > 0:
                            jacs = []
                            for hs in hist_tok_sets:
                                union = len(seq_toks | hs)
                                if union > 0:
                                    jacs.append(len(seq_toks & hs) / union)
                            if jacs:
                                bonus += rerank_alpha * max(jacs)
                    if rerank_beta > 0.0:
                        bonus += rerank_beta * math.log(1.0 + item_pop.get(seq, 0))
                    rescored.append((seq, sc + bonus))
                head_sorted = sorted(rescored, key=lambda x: x[1], reverse=True)
                pairs = head_sorted + tail
            else:
                pairs = sorted(zip(batch_seqs, batch_scores), key=lambda x: x[1], reverse=True)

            pred = [p[0] for p in pairs]
            all_pred_list.append(pred)
            all_gold_list.append(targets[b])

        if (start_idx // eval_batch_size + 1) % 50 == 0:
            elapsed = time.time() - eval_start
            done = start_idx + eval_batch_size
            print(f"    [Eval] {done}/{len(test_dataset)} samples ({elapsed:.1f}s)")
            sys.stdout.flush()

    elapsed = time.time() - eval_start
    print(f"  [Eval] Beam search done in {elapsed:.1f}s")

    # Compute metrics
    results = compute_rec_metrics(all_gold_list, all_pred_list, [5, 10])
    results["phase"] = phase
    results["eval_test_phase"] = eval_phase

    print(f"  [Eval] Train phase {phase}, test phase {eval_phase} Results:")
    for k, v in results.items():
        if k != "phase":
            print(f"    {k}: {v}")

    del model
    torch.cuda.empty_cache()
    return results


def collect_rqvae_metrics(ckpt_dir, data_path, dataset, num_phases, device):
    """Collect RQ-VAE layer metrics across all phases."""
    metrics = {
        "phase": [], "warm_recon_mse": [], "collision_rate_all": [],
        "collision_rate_warm": [], "collision_rate_cold": [],
        "codebook_sizes": [], "utilization": [],
        "encoder_drift_cosine": [],
    }

    ref_enc = None
    for phase in range(num_phases):
        ckpt_path = os.path.join(ckpt_dir, f"phase{phase}", "freq_best_collision_model.pth")
        if not os.path.exists(ckpt_path):
            continue

        warm_data = warm_TokenDataset(data_path, phase, dataset)
        loader = DataLoader(warm_data, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
        model, _, _ = load_rqvae_from_ckpt(ckpt_path, warm_data.dim, device, phase)

        # Warm items
        all_enc, all_quant, all_idx, all_recon = [], [], [], []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                enc = model.encoder(data)
                out, _, indices = model.rq(enc, use_freq=False, use_sk=False, p=phase)
                recon = model.decoder(out)
                recon_loss = F.mse_loss(recon, data, reduction="none").mean(dim=-1)
                all_enc.append(enc.cpu())
                all_quant.append(out.cpu())
                all_idx.append(indices.cpu())
                all_recon.append(recon_loss.cpu())

        enc = torch.cat(all_enc, 0)
        idx = torch.cat(all_idx, 0)
        recon = torch.cat(all_recon, 0)

        warm_mse = recon.mean().item()

        # Encoder drift
        if ref_enc is None:
            ref_enc = enc.clone()
            drift_cos = 1.0
        else:
            n = min(ref_enc.shape[0], enc.shape[0])
            drift_cos = F.cosine_similarity(ref_enc[:n], enc[:n], dim=-1).mean().item()

        # Collision rates
        warm_strs = ["-".join(str(int(x)) for x in row) for row in idx.numpy()]
        warm_cr = (len(warm_strs) - len(set(warm_strs))) / max(len(warm_strs), 1)

        cold_strs = []
        if phase > 0:
            try:
                cold_data = cold_TokenDataset(data_path, phase, dataset)
                cold_loader = DataLoader(cold_data, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
                cold_idx_list = []
                with torch.no_grad():
                    for data in cold_loader:
                        data = data.to(device)
                        cidx = model.get_indices(data, p=phase)
                        cold_idx_list.append(cidx.cpu())
                cold_idx = torch.cat(cold_idx_list, 0)
                cold_strs = ["-".join(str(int(x)) for x in row) for row in cold_idx.numpy()]
            except Exception:
                pass

        cold_cr = (len(cold_strs) - len(set(cold_strs))) / max(len(cold_strs), 1) if cold_strs else 0
        all_strs = warm_strs + cold_strs
        all_cr = (len(all_strs) - len(set(all_strs))) / max(len(all_strs), 1)

        # Codebook sizes and utilization
        sizes = []
        utils = []
        for li in range(idx.shape[-1]):
            cb_size = sum(cb.weight.shape[0] for cb in model.rq.vq_layers[li].codebook)
            unique = torch.unique(idx[:, li]).shape[0]
            sizes.append(cb_size)
            utils.append(unique / cb_size)

        metrics["phase"].append(phase)
        metrics["warm_recon_mse"].append(warm_mse)
        metrics["collision_rate_all"].append(all_cr)
        metrics["collision_rate_warm"].append(warm_cr)
        metrics["collision_rate_cold"].append(cold_cr)
        metrics["codebook_sizes"].append(sizes)
        metrics["utilization"].append(utils)
        metrics["encoder_drift_cosine"].append(drift_cos)

        del model
        torch.cuda.empty_cache()

    return metrics


def plot_dual_layer_results(rqvae_metrics, training_log, output_dir):
    """Generate comprehensive visualization of the dual-layer CL framework."""
    ensure_dir(output_dir)

    phases = rqvae_metrics["phase"]
    if not phases:
        print("  [Plot] No metrics to plot")
        return

    # ===== Figure 1: RQ-VAE Layer Performance =====
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Dual-Layer CL Framework: RQ-VAE Layer Performance\n"
                 "Innovation ① Elastic Codebook + Innovation ② Adaptive Expansion",
                 fontsize=15, fontweight="bold")

    # (a) Warm Recon MSE
    ax = axes[0, 0]
    ax.plot(phases, rqvae_metrics["warm_recon_mse"], "s-",
            color="tab:blue", linewidth=2.5, markersize=8, label="Dual-Layer CL")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Warm Recon MSE")
    ax.set_title("(a) Warm Item Reconstruction MSE ↓")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # (b) Encoder Drift
    ax = axes[0, 1]
    ax.plot(phases, rqvae_metrics["encoder_drift_cosine"], "s-",
            color="tab:blue", linewidth=2.5, markersize=8, label="Dual-Layer CL")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Cosine Similarity to Phase 0")
    ax.set_title("(b) Encoder Drift (Cosine) ↑ = Less Drift")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # (c) Codebook Size
    ax = axes[0, 2]
    sizes_l0 = [s[0] for s in rqvae_metrics["codebook_sizes"]]
    ax.plot(phases, sizes_l0, "s-",
            color="tab:blue", linewidth=2.5, markersize=8, label="Layer 0")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Codebook Size")
    ax.set_title("(c) Adaptive Codebook Growth")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # (d) Collision Rates
    ax = axes[1, 0]
    ax.plot(phases, rqvae_metrics["collision_rate_all"], "s-",
            color="tab:blue", linewidth=2.5, markersize=8, label="All Items")
    ax.plot(phases, rqvae_metrics["collision_rate_warm"], "D--",
            color="tab:green", linewidth=2, markersize=7, label="Warm Items")
    ax.plot(phases, rqvae_metrics["collision_rate_cold"], "^:",
            color="tab:orange", linewidth=2, markersize=7, label="Cold Items")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Collision Rate")
    ax.set_title("(d) Collision Rates ↓ = Better")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # (e) Utilization
    ax = axes[1, 1]
    avg_util = [np.mean(u) for u in rqvae_metrics["utilization"]]
    ax.plot(phases, avg_util, "s-",
            color="tab:blue", linewidth=2.5, markersize=8, label="Avg Utilization")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Utilization")
    ax.set_title("(e) Codebook Utilization ↑ = Better")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # (f) Summary text
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = "Dual-Layer CL Framework Summary\n"
    summary_text += "=" * 40 + "\n\n"
    summary_text += "RQ-VAE Layer (Tokenization):\n"
    summary_text += "  ① Elastic Codebook Fine-tuning\n"
    summary_text += "     Old CB lr = 5% × base_lr\n\n"
    summary_text += "  ② Adaptive Expansion\n"
    summary_text += "     Collision-driven sizing\n\n"
    summary_text += "LLM Layer (Recommendation):\n"
    summary_text += "  ③ Semantic Distillation\n"
    summary_text += "     KL(teacher ∥ student)\n\n"
    summary_text += "  ④ Experience Replay\n"
    summary_text += "     Collision-aware sampling\n"
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()
    path = os.path.join(output_dir, "dual_layer_rqvae.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # ===== Figure 2: LLM Layer Training Curves =====
    if training_log:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Dual-Layer CL Framework: LLM Layer Training\n"
                     "Innovation ③ Semantic Distillation + Innovation ④ Experience Replay",
                     fontsize=15, fontweight="bold")

        # (a) Training loss per phase
        ax = axes[0]
        for phase_data in training_log:
            p = phase_data["phase"]
            ax.bar(p, phase_data.get("final_loss", 0),
                   color="tab:blue", alpha=0.8, edgecolor="black")
        ax.set_xlabel("Phase")
        ax.set_ylabel("Final Training Loss")
        ax.set_title("(a) LLM Training Loss per Phase")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # (b) Distillation loss
        ax = axes[1]
        for phase_data in training_log:
            p = phase_data["phase"]
            ax.bar(p, phase_data.get("distill_loss", 0),
                   color="tab:purple", alpha=0.8, edgecolor="black")
        ax.set_xlabel("Phase")
        ax.set_ylabel("Distillation Loss")
        ax.set_title("(b) KL Distillation Loss")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # (c) Replay statistics
        ax = axes[2]
        n_new = [d.get("n_new_samples", 0) for d in training_log]
        n_replay = [d.get("n_replay_samples", 0) for d in training_log]
        x = [d["phase"] for d in training_log]
        ax.bar(x, n_new, label="New Samples", color="tab:blue", alpha=0.8)
        ax.bar(x, n_replay, bottom=n_new, label="Replay Samples",
               color="tab:orange", alpha=0.8)
        ax.set_xlabel("Phase")
        ax.set_ylabel("Number of Samples")
        ax.set_title("(c) Training Data Composition")
        ax.legend()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()
        path = os.path.join(output_dir, "dual_layer_llm.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")

    # ===== Figure 3: Framework Architecture Diagram =====
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Title
    ax.text(5, 9.5, "Dual-Layer Continual Learning Framework",
            ha='center', fontsize=18, fontweight='bold')
    ax.text(5, 9.0, "for LLM-based Incremental Recommendation",
            ha='center', fontsize=14, style='italic', color='gray')

    # RQ-VAE Layer box
    rqvae_box = plt.Rectangle((0.5, 5.5), 4, 3, fill=True,
                               facecolor='#FFE0B2', edgecolor='#E65100', linewidth=2)
    ax.add_patch(rqvae_box)
    ax.text(2.5, 8.2, "Layer 1: RQ-VAE Tokenizer", ha='center',
            fontsize=13, fontweight='bold', color='#E65100')
    ax.text(2.5, 7.5, "① Elastic Codebook Fine-tuning", ha='center', fontsize=11)
    ax.text(2.5, 7.0, "   Old CB lr = 5% × base_lr", ha='center', fontsize=9, color='gray')
    ax.text(2.5, 6.3, "② Adaptive Expansion", ha='center', fontsize=11)
    ax.text(2.5, 5.8, "   Collision-driven dynamic sizing", ha='center', fontsize=9, color='gray')

    # LLM Layer box
    llm_box = plt.Rectangle((5.5, 5.5), 4, 3, fill=True,
                              facecolor='#BBDEFB', edgecolor='#1565C0', linewidth=2)
    ax.add_patch(llm_box)
    ax.text(7.5, 8.2, "Layer 2: LLM Recommender", ha='center',
            fontsize=13, fontweight='bold', color='#1565C0')
    ax.text(7.5, 7.5, "③ Semantic Distillation", ha='center', fontsize=11)
    ax.text(7.5, 7.0, "   KL(P_teacher ∥ P_student)", ha='center', fontsize=9, color='gray')
    ax.text(7.5, 6.3, "④ Experience Replay", ha='center', fontsize=11)
    ax.text(7.5, 5.8, "   Collision-aware priority sampling", ha='center', fontsize=9, color='gray')

    # Arrow between layers
    ax.annotate("", xy=(5.5, 7.0), xytext=(4.5, 7.0),
                arrowprops=dict(arrowstyle="->", lw=2, color='black'))
    ax.text(5.0, 7.3, "Token IDs\n+ Collision\nRisk", ha='center', fontsize=8, color='gray')

    # Loss function box
    loss_box = plt.Rectangle((2, 2), 6, 2.5, fill=True,
                              facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(loss_box)
    ax.text(5, 4.2, "Loss Function", ha='center', fontsize=13, fontweight='bold', color='#2E7D32')
    ax.text(5, 3.5, "L_total = L_rec + λ₁·L_distill + λ₂·L_align", ha='center', fontsize=12)
    ax.text(5, 2.8, "L_rec: Next-item prediction  |  L_distill: KL divergence  |  L_align: Embedding alignment",
            ha='center', fontsize=9, color='gray')
    ax.text(5, 2.3, "RQ-VAE: L_recon + β·L_quant (with elastic LR)", ha='center', fontsize=9, color='gray')

    # Arrows to loss
    ax.annotate("", xy=(3.5, 4.5), xytext=(2.5, 5.5),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='#E65100', ls='--'))
    ax.annotate("", xy=(6.5, 4.5), xytext=(7.5, 5.5),
                arrowprops=dict(arrowstyle="->", lw=1.5, color='#1565C0', ls='--'))

    # Output
    ax.text(5, 1.2, "🎯 Recommendation Output: Next-Item Prediction",
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E1BEE7', edgecolor='#6A1B9A'))

    ax.annotate("", xy=(5, 1.7), xytext=(5, 2.0),
                arrowprops=dict(arrowstyle="->", lw=2, color='#6A1B9A'))

    plt.tight_layout()
    path = os.path.join(output_dir, "dual_layer_architecture.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ===================================================================
# Part 4: Main Pipeline
# ===================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Dual-Layer CL Framework")

    # Data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="/root/models/Qwen/Qwen2___5-1___5B")
    parser.add_argument("--output_dir", type=str, default="./experiments/dual_layer_outputs")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_phases", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_test_phase", type=int, default=0,
                        help="Fixed test phase for evaluation. Use -1 to evaluate the current phase.")

    # RQ-VAE (Innovation ① + ②)
    parser.add_argument("--rqvae_epochs", type=int, default=5000)
    parser.add_argument("--old_cb_lr_ratio", type=float, default=0.05,
                        help="Elastic codebook LR ratio (Innovation ①)")
    parser.add_argument("--collision_threshold", type=float, default=0.03,
                        help="Adaptive expansion threshold (Innovation ②)")

    # LLM (Innovation ③ + ④)
    parser.add_argument("--llm_epochs", type=int, default=15)
    parser.add_argument("--llm_lr", type=float, default=2e-5)
    parser.add_argument("--llm_batch_size", type=int, default=12)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--distill_weight", type=float, default=0.3,
                        help="λ₁: distillation loss weight (Innovation ③). "
                             "Default 0.3 is tuned for listwise mode; raise to "
                             "0.5 if you switch back to --distill_mode kl.")
    parser.add_argument("--align_weight", type=float, default=0.1,
                        help="λ₂: embedding alignment loss weight (Innovation ③)")
    parser.add_argument("--distill_temperature", type=float, default=2.0,
                        help="Temperature for distillation (Innovation ③)")
    parser.add_argument("--teacher_topk", type=int, default=256,
                        help="Top-K of teacher logits cached for distillation. "
                             "Default 256 (listwise) / 128 (kl). Tunable for "
                             "hyperparameter analysis, e.g. {64,128,256,512}.")
    parser.add_argument("--replay_ratio", type=float, default=0.3,
                        help="Fraction of old data to replay (Innovation ④)")

    # ── Inference / Beam-search hyper-parameters (for hyper-param analysis) ──
    parser.add_argument("--num_beams", type=int, default=20,
                        help="Number of beams used at evaluation. "
                             "Tunable for hyperparameter analysis, e.g. {10,20,40}.")
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="Batch size used during beam-search evaluation.")
    parser.add_argument("--rerank_topk", type=int, default=0,
                        help="If >0, only the top-K beams (by LLM score) are kept "
                             "before applying history/popularity rerank. 0 means "
                             "rerank all returned beams (= num_beams). Tunable for "
                             "hyperparameter analysis.")

    # Improvement #5 (Innovation ⑤): Token-Stable Codebook Inheritance (TSCI)
    parser.add_argument("--freeze_old_tokens", action="store_true", default=True,
                        help="Inherit old items' tokens from previous phase's index "
                             "to prevent 100%% token drift. Default True. "
                             "Use --no_freeze_old_tokens to disable for ablation.")
    parser.add_argument("--no_freeze_old_tokens", dest="freeze_old_tokens",
                        action="store_false",
                        help="Disable Token-Stable Codebook Inheritance "
                             "(reverts to legacy behavior, for ablation only).")

    # Improvement #1: distillation mode (kl | listwise)
    parser.add_argument("--distill_mode", type=str, default="listwise",
                        choices=["kl", "listwise"],
                        help="Distillation objective. 'listwise' is rank-aware (ListMLE) "
                             "and is the default; 'kl' falls back to vanilla KL.")
    # Improvement #2: alignment mode (full | selective)
    parser.add_argument("--align_mode", type=str, default="selective",
                        choices=["full", "selective"],
                        help="Embedding alignment scope. 'selective' restricts the "
                             "constraint to old ID tokens via cosine; 'full' is the "
                             "original full-vocab MSE.")
    # Improvement #5: beam-rerank weights.
    # Defaults are 0.0 (rerank OFF). The previous defaults (0.3 / 0.1) were
    # found to overpower LLM beam scores and HURT Hit/NDCG/Recall, so rerank
    # is now opt-in -- enable it explicitly with --rerank_alpha / --rerank_beta
    # only after verifying it helps on a held-out phase.
    parser.add_argument("--rerank_alpha", type=float, default=0.0,
                        help="Weight of history token-Jaccard bonus in beam "
                             "rerank (0 disables, applied at eval time only).")
    parser.add_argument("--rerank_beta", type=float, default=0.0,
                        help="Weight of log-popularity bonus in beam rerank "
                             "(0 disables, applied at eval time only).")

    # Control
    parser.add_argument("--skip_rqvae", action="store_true",
                        help="Skip RQ-VAE training (use existing checkpoints)")
    parser.add_argument("--skip_llm", action="store_true",
                        help="Skip LLM training (only RQ-VAE + metrics)")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip all training, only collect metrics and plot")
    parser.add_argument("--start_phase", type=int, default=0,
                        help="Resume training from this phase (0-indexed). "
                             "Checkpoints from previous phases must already exist.")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    args.data_path = resolve_local_path(args.data_path, "data_path", must_exist=True, expect_dir=True)
    args.output_dir = resolve_local_path(args.output_dir, "output_dir", must_exist=False, expect_dir=False)
    args.base_model = resolve_local_path(args.base_model, "base_model", must_exist=True, expect_dir=True)

    device = torch.device(args.device)
    ensure_dir(args.output_dir)

    # Directory structure
    args.rqvae_ckpt_dir = os.path.join(args.output_dir, "rqvae_ckpts")
    args.llm_ckpt_dir = os.path.join(args.output_dir, "llm_ckpts")
    args.index_dir = os.path.join(args.output_dir, "indices")

    print("=" * 70)
    print("  Dual-Layer Continual Learning Framework")
    print("  for LLM-based Incremental Recommendation")
    print("=" * 70)
    print(f"  Dataset:           {args.dataset}")
    print(f"  Phases:            {args.num_phases}")
    print(f"  Base model:        {args.base_model}")
    print(f"  Eval test phase:   {args.eval_test_phase}")
    print()
    print("  Innovation ① Elastic Codebook:    old_cb_lr_ratio = {:.0%}".format(args.old_cb_lr_ratio))
    print("  Innovation ② Adaptive Expansion:  threshold = {:.2%}".format(args.collision_threshold))
    print("  Innovation ③ Semantic Distill:     λ₁ = {}, T = {}".format(
        args.distill_weight, args.distill_temperature))
    print("  Innovation ④ Experience Replay:    ratio = {:.0%}".format(args.replay_ratio))
    print("=" * 70)

    training_log = []

    if not args.skip_training:
        # ===== Phase-by-phase training =====
        prev_rqvae_ckpt = None
        prev_llm_ckpt = None
        prev_index_file = None  # ★ TSCI: track previous phase index for token inheritance

        # Resume from start_phase: restore checkpoints of the previous phase
        if args.start_phase > 0:
            prev_phase = args.start_phase - 1
            prev_rqvae_ckpt = os.path.join(args.rqvae_ckpt_dir, f"phase{prev_phase}",
                                           "freq_best_collision_model.pth")
            prev_llm_ckpt = os.path.join(args.llm_ckpt_dir, f"phase{prev_phase}")
            prev_index_file = os.path.join(args.index_dir, f"phase{prev_phase}",
                                           f"{args.dataset}.index.json")
            print(f"  [Resume] Starting from Phase {args.start_phase}, "
                  f"loading checkpoints from Phase {prev_phase}")
            print(f"  [Resume] RQ-VAE ckpt: {prev_rqvae_ckpt}")
            print(f"  [Resume] LLM   ckpt:  {prev_llm_ckpt}")
            print(f"  [Resume] Index file: {prev_index_file}")

        for phase in range(args.start_phase, args.num_phases):
            print(f"\n{'#' * 70}")
            print(f"  PHASE {phase}")
            print(f"{'#' * 70}")

            phase_log = {"phase": phase}

            # Step 1: Train RQ-VAE (Innovation ① + ②)
            if not args.skip_rqvae:
                print(f"\n  === Step 1: RQ-VAE Training (Phase {phase}) ===")
                prev_rqvae_ckpt = train_rqvae_phase(args, phase, prev_rqvae_ckpt, device)
            else:
                prev_rqvae_ckpt = os.path.join(args.rqvae_ckpt_dir, f"phase{phase}",
                                                "freq_best_collision_model.pth")

            # Step 2: Generate index file
            print(f"\n  === Step 2: Generate Index (Phase {phase}) ===")
            index_file = generate_index_file(
                prev_rqvae_ckpt, args.data_path, args.dataset,
                phase, device, args.index_dir,
                prev_index_file=prev_index_file,
                freeze_old_tokens=args.freeze_old_tokens,
            )
            prev_index_file = index_file  # ★ TSCI: pass to next phase

            # Step 3: Train LLM (Innovation ③ + ④)
            if not args.skip_llm:
                print(f"\n  === Step 3: LLM Training (Phase {phase}) ===")
                prev_llm_ckpt = train_llm_phase(
                    args, phase, prev_llm_ckpt, index_file, device
                )
                phase_log["llm_ckpt"] = prev_llm_ckpt
            else:
                prev_llm_ckpt = os.path.join(args.llm_ckpt_dir, f"phase{phase}")

            # Step 4: Evaluate recommendation metrics
            if not args.skip_llm:
                print(f"\n  === Step 4: Evaluation (Phase {phase}) ===")
                eval_results = evaluate_recommendation(
                    args, phase, prev_llm_ckpt, index_file, device
                )
                phase_log["eval"] = eval_results

            training_log.append(phase_log)

    # ===== Collect metrics =====
    print(f"\n{'=' * 70}")
    print("  Collecting Metrics")
    print(f"{'=' * 70}")

    rqvae_metrics = collect_rqvae_metrics(
        args.rqvae_ckpt_dir, args.data_path, args.dataset,
        args.num_phases, device
    )

    # ===== Evaluate all phases (if skipped during training) =====
    if args.skip_training or args.skip_llm:
        for phase in range(args.num_phases):
            llm_ckpt = os.path.join(args.llm_ckpt_dir, f"phase{phase}")
            index_file = os.path.join(args.index_dir, f"phase{phase}", f"{args.dataset}.index.json")
            if os.path.exists(llm_ckpt) and os.path.exists(index_file):
                eval_results = evaluate_recommendation(args, phase, llm_ckpt, index_file, device)
                # Find or create phase_log
                found = False
                for pl in training_log:
                    if pl.get("phase") == phase:
                        pl["eval"] = eval_results
                        found = True
                        break
                if not found:
                    training_log.append({"phase": phase, "eval": eval_results})

    # ===== Generate visualizations =====
    print(f"\n{'=' * 70}")
    print("  Generating Visualizations")
    print(f"{'=' * 70}")

    plot_dual_layer_results(rqvae_metrics, training_log, args.output_dir)

    # ===== Save results =====
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

    # Collect per-phase evaluation summary
    eval_summary = {}
    for pl in training_log:
        if "eval" in pl and isinstance(pl["eval"], dict):
            p = pl["phase"]
            eval_summary[f"phase{p}"] = {k: v for k, v in pl["eval"].items() if k != "phase"}

    results = {
        "rqvae_metrics": rqvae_metrics,
        "eval_summary": eval_summary,
        "training_log": training_log,
        "config": {
            "old_cb_lr_ratio": args.old_cb_lr_ratio,
            "collision_threshold": args.collision_threshold,
            "distill_weight": args.distill_weight,
            "align_weight": args.align_weight,
            "distill_temperature": args.distill_temperature,
            "teacher_topk": getattr(args, "teacher_topk", None),
            "replay_ratio": args.replay_ratio,
            "distill_mode": getattr(args, "distill_mode", None),
            "align_mode": getattr(args, "align_mode", None),
            "freeze_old_tokens": getattr(args, "freeze_old_tokens", None),
            "rerank_alpha": getattr(args, "rerank_alpha", None),
            "rerank_beta": getattr(args, "rerank_beta", None),
            "rerank_topk": getattr(args, "rerank_topk", None),
            "num_beams": getattr(args, "num_beams", None),
            "eval_batch_size": getattr(args, "eval_batch_size", None),
            "llm_epochs": args.llm_epochs,
            "llm_lr": args.llm_lr,
            "llm_batch_size": args.llm_batch_size,
            "rqvae_epochs": args.rqvae_epochs,
            "num_phases": args.num_phases,
            "seed": args.seed,
        }
    }

    results_path = os.path.join(args.output_dir, "dual_layer_results.json")
    with open(results_path, "w") as f:
        json.dump(json.loads(json.dumps(results, default=convert)), f, indent=2)

    # ===== Print Summary =====
    print(f"\n{'=' * 70}")
    print("  DUAL-LAYER CL FRAMEWORK — COMPLETE")
    print(f"{'=' * 70}")

    if rqvae_metrics["phase"]:
        print(f"\n  RQ-VAE Layer Metrics (per phase):")
        print(f"    {'Phase':<8} {'Warm MSE':<12} {'Enc Drift':<12} {'Collision':<12} {'Utilization':<12}")
        print(f"    {'-'*56}")
        for i, p in enumerate(rqvae_metrics['phase']):
            print(f"    {p:<8} {rqvae_metrics['warm_recon_mse'][i]:<12.6f} "
                  f"{rqvae_metrics['encoder_drift_cosine'][i]:<12.4f} "
                  f"{rqvae_metrics['collision_rate_all'][i]:<12.4%} "
                  f"{np.mean(rqvae_metrics['utilization'][i]):<12.2%}")

    if eval_summary:
        print(f"\n  Recommendation Metrics (per phase):")
        header_keys = None
        for pkey in sorted(eval_summary.keys()):
            ev = eval_summary[pkey]
            metric_keys = [k for k in ev.keys() if k not in ("note",)]
            if header_keys is None and metric_keys:
                header_keys = metric_keys
                header = f"    {'Phase':<8}" + "".join(f"{k:<12}" for k in header_keys)
                print(header)
                print(f"    {'-'*( 8 + 12*len(header_keys))}")
            vals = "".join(f"{ev.get(k, 'N/A'):<12}" for k in (header_keys or []))
            print(f"    {pkey:<8}{vals}")

    print(f"\n  Output files:")
    print(f"    {args.output_dir}/dual_layer_rqvae.png        (RQ-VAE layer metrics)")
    print(f"    {args.output_dir}/dual_layer_llm.png          (LLM layer training)")
    print(f"    {args.output_dir}/dual_layer_architecture.png (Framework diagram)")
    print(f"    {args.output_dir}/dual_layer_results.json     (All results)")
    print(f"    {args.rqvae_ckpt_dir}/                        (RQ-VAE checkpoints)")
    print(f"    {args.llm_ckpt_dir}/                          (LLM checkpoints)")
    print(f"    {args.index_dir}/                             (Item index files)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

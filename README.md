```markdown
# CoCo: Collision-aware Continual Learning for Generative Recommendation
Official implementation of **CoCo (Collision-aware Continual Learning)** proposed in the paper *Dynamic Update Mechanism for Generative Recommendation*.

### Project Overview
Existing LLM-driven generative recommendation suffers three core incremental drawbacks: rigid old codebook induced by RQ-VAE stop-gradient constraint, redundant storage from fixed-size linear codebook expansion, and collision-focused catastrophic forgetting under LLM incremental fine-tuning.
We design a two-layer collaborative continual learning framework CoCo:
1. **RQ-VAE Layer**: Elastic codebook fine-tuning with differentiated learning rates, plus collision-rate-based adaptive sub-codebook expansion to cut redundant parameters;
2. **LLM Layer**: Top-K approximate cross-stage semantic distillation and collision-risk sorted experience replay to mitigate item forgetting;
3. Real-time item collision metric serves as cross-layer linkage to unify the optimization of two submodules.

### Datasets
Two standard Amazon review benchmarks are adopted:
- Amazon Games
- Amazon Software
All items split chronologically into four incremental stages (Phase 0 ~ Phase3) by their first interaction timestamp to simulate real-world new-item streaming.
Raw processed data is stored under `CoCo-data/`.

### Environment Requirements
```
python >=3.9
torch >=2.0
transformers >=4.38
peft >=0.9.0
numpy, pandas, scikit-learn
```
- Base LLM: Qwen2.5-1.5B with 8-bit quantization & LoRA tuning
- Training Hardware: Single RTX3090 (24GB)

### Key Hyperparameters
| Param | Value | Description |
|-------|-------|-------------|
| $\gamma$ | 0.05 | Old codebook learning rate discount factor |
| $\tau$ | 0.03 | Collision threshold for adaptive expansion |
| $\varsigma$ | 0.3 | High-risk replay sampling proportion |
| $\lambda_1$ | 0.5 | Distillation loss weight |
| $\lambda_2$ | 0.1 | Embedding alignment loss weight |
| $T$ | 2.0 | Distillation temperature |
| $K$ | 128 | Top-K logits for KL calculation |

### Run Commands to Reproduce Full Results
```bash
# Train on Software Dataset
nohup python experiments/run_dual_layer_cl.py \
--data_path CoCo-data/Software_11111_0.1 \
--dataset Software_11111_0.1 \
--base_model ~/CoCo/models/Qwen/Qwen2___5-1___5B \
--device cuda:0 --num_phases 4 --replay_ratio 0.3 --start_phase 2 \
--output_dir ./experiments/software_replay0.3 > run_soft2.log 2>&1 &

# Train on Games Dataset
nohup python experiments/run_dual_layer_cl.py \
--data_path CoCo-data/Games_11111_0.1 \
--dataset Games_11111_0.1 \
--base_model ~/CoCo/models/Qwen/Qwen2___5-1___5B \
--device cuda:0 --num_phases 4 --replay_ratio 0.3 --start_phase 2 \
--output_dir ./experiments/games_replay0.3 > run_game.log 2>&1 &
```

## Main Experimental Results
### 1. RQ-VAE Layer Metrics (MSE$\times10^{-2}$ / Encoder Similarity / Collision Rate)
| Dataset | Model | Phase0 | Phase1 | Phase2 | Phase3 |
|--------|-------|--------|--------|--------|--------|
| Games | Reformer | 0.376/1.0/0.2234 | 0.380/0.9212/0.0473 | 0.383/0.8450/0.0235 | 0.384/0.7947/0.0312 |
| Games | CoCo | 0.067/1.0/0.1207 | 0.230/0.9267/0.1451 | 0.317/0.7410/0.1099 | 0.368/0.5736/0.1029 |
| Software | Reformer | 0.190/1.0/0.0795 | 0.152/0.9425/0.0723 | 0.225/0.8631/0.0412 | 0.249/0.8127/0.0416 |
| Software | CoCo | 0.050/1.0/0.0532 | 0.169/0.9565/0.0829 | 0.239/0.8893/0.0687 | 0.265/0.8068/0.0616 |

### 2. Phase3 Recommendation Metrics (NDCG@5 / NDCG@10 / Recall@5 / Recall@10)
| Dataset | Reformer | CoCo |
|---------|---------|------|
| Games | 0.0112 / 0.0145 / 0.0102 / 0.0189 | 0.0119 / 0.0151 / 0.0106 / 0.0199 |
| Software | 0.0266 / 0.0354 / 0.0313 / 0.0598 | 0.0268 / 0.0384 / 0.0329 / 0.0638 |

### 3. Ablation Results on Games (Phase3)
| Variant | NDCG@5 | NDCG@10 | Recall@5 | Recall@10 |
|---------|--------|---------|----------|-----------|
| Full CoCo | 0.0119 | 0.0151 | 0.0106 | 0.0199 |
| w/o Elastic CB | 0.0051 | 0.0066 | 0.0054 | 0.0096 |
| w/o Adaptive Exp | 0.0087 | 0.0111 | 0.0082 | 0.0148 |
| w/o Distillation | 0.0096 | 0.0120 | 0.0093 | 0.0161 |
| w/o Replay | 0.0091 | 0.0098 | 0.0069 | 0.0112 |
| Baseline(Reformer) | 0.0112 | 0.0145 | 0.0102 | 0.0189 |


```
```

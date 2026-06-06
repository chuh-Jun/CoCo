"""
Generate Item Embeddings for Games Dataset.

Provides multiple embedding generation methods:
  1. collaborative: SVD on user-item interaction matrix (recommended, no text needed)
  2. sentence_transformer: Text-based embeddings using Sentence-Transformers
  3. llama: Text-based embeddings using LLaMA model
  4. random: Random embeddings (for quick testing only)

The embedding file is required by the RQ-VAE training pipeline (TIGER, Reformer, LSAT).

"""

import argparse
import os
import json
import numpy as np
import torch
from tqdm import tqdm
from scipy import sparse
from scipy.sparse.linalg import svds


# ---------------------------------------------------------------------------
# Item info loading
# ---------------------------------------------------------------------------

def load_item_map(data_path):
    """Load item_map and return {internal_id: original_id} mapping and num_items."""
    item_map_path = os.path.join(data_path, "item_map.npy")
    item_map = np.load(item_map_path, allow_pickle=True).item()

    sample_key = list(item_map.keys())[0]
    sample_val = list(item_map.values())[0]

    if isinstance(sample_val, int) or (isinstance(sample_val, str) and sample_val.isdigit()):
        # item_map is {original_id: internal_id}
        num_items = max(int(v) for v in item_map.values()) + 1
        id_to_original = {}
        for orig, iid in item_map.items():
            id_to_original[int(iid)] = str(orig)
    else:
        # item_map is {internal_id: original_id}
        num_items = max(int(k) for k in item_map.keys()) + 1
        id_to_original = {int(k): str(v) for k, v in item_map.items()}

    return id_to_original, num_items


def load_metadata(metadata_path):
    """Load Amazon product metadata from JSON/JSONL file.

    Supports both .json (list) and .jsonl (one JSON per line) formats.
    Returns {asin: {"title": ..., "description": ..., "categories": ...}}.
    """
    if metadata_path is None or not os.path.exists(metadata_path):
        return {}

    print(f"  Loading metadata from {metadata_path} ...")
    meta = {}
    ext = os.path.splitext(metadata_path)[1].lower()

    if ext in (".jsonl", ".jl"):
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Reading metadata"):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                asin = obj.get("asin") or obj.get("parent_asin") or obj.get("id", "")
                if asin:
                    meta[str(asin)] = obj
    elif ext == ".json":
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for obj in tqdm(data, desc="Reading metadata"):
                asin = obj.get("asin") or obj.get("parent_asin") or obj.get("id", "")
                if asin:
                    meta[str(asin)] = obj
        elif isinstance(data, dict):
            meta = data
    elif ext == ".gz":
        import gzip
        with gzip.open(metadata_path, "rt", encoding="utf-8") as f:
            for line in tqdm(f, desc="Reading metadata (gzip)"):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                asin = obj.get("asin") or obj.get("parent_asin") or obj.get("id", "")
                if asin:
                    meta[str(asin)] = obj

    print(f"  Loaded metadata for {len(meta)} products")
    return meta


def build_text_descriptions(id_to_original, num_items, metadata):
    """Build text descriptions for each item.

    Priority: metadata title+description > original_id as text > generic placeholder.
    """
    texts = []
    stats = {"meta_full": 0, "meta_title": 0, "id_only": 0, "placeholder": 0}

    for i in range(num_items):
        orig_id = id_to_original.get(i, None)

        if orig_id and orig_id in metadata:
            obj = metadata[orig_id]
            title = obj.get("title", "")
            # Try multiple description field names
            desc = (obj.get("description", "") or obj.get("desc", "")
                    or obj.get("feature", ""))
            if isinstance(desc, list):
                desc = " ".join(str(d) for d in desc)
            categories = obj.get("categories", obj.get("category", ""))
            if isinstance(categories, list):
                if categories and isinstance(categories[0], list):
                    categories = " > ".join(categories[0])
                else:
                    categories = ", ".join(str(c) for c in categories)

            parts = []
            if title:
                parts.append(title)
            if categories:
                parts.append(f"Category: {categories}")
            if desc:
                # Truncate long descriptions
                parts.append(desc[:200])

            if parts:
                text = " | ".join(parts)
                if desc:
                    stats["meta_full"] += 1
                else:
                    stats["meta_title"] += 1
            else:
                text = f"Game product {orig_id}"
                stats["id_only"] += 1
        elif orig_id:
            text = f"Game product {orig_id}"
            stats["id_only"] += 1
        else:
            text = f"Game item {i}"
            stats["placeholder"] += 1

        texts.append(text)

    print(f"  Text stats: {stats}")
    print(f"  Sample texts: {texts[:3]}")
    return texts


# ---------------------------------------------------------------------------
# Collaborative filtering embeddings (SVD)
# ---------------------------------------------------------------------------

def generate_collaborative_embeddings(data_path, num_items, emb_dim=256):
    """Generate item embeddings via SVD on user-item interaction matrix.

    This method captures collaborative filtering signals from user behavior,
    producing high-quality embeddings without requiring any text information.
    """
    print(f"\n  [Collaborative] Building user-item interaction matrix ...")

    # Collect interactions from all available phases
    all_interactions = {}  # {user_id: set(item_ids)}
    for phase_dir_name in sorted(os.listdir(data_path)):
        phase_path = os.path.join(data_path, phase_dir_name)
        if not os.path.isdir(phase_path) or not phase_dir_name.startswith("phase"):
            continue

        for fname in ["training_dict.npy", "validation_dict.npy", "testing_dict.npy"]:
            fpath = os.path.join(phase_path, fname)
            if not os.path.exists(fpath):
                continue
            d = np.load(fpath, allow_pickle=True).item()
            for uid, items in d.items():
                if uid not in all_interactions:
                    all_interactions[uid] = set()
                if isinstance(items, (list, np.ndarray)):
                    all_interactions[uid].update(int(x) for x in items)

    # Also load training_list if available (some datasets use this format)
    for phase_dir_name in sorted(os.listdir(data_path)):
        phase_path = os.path.join(data_path, phase_dir_name)
        if not os.path.isdir(phase_path) or not phase_dir_name.startswith("phase"):
            continue
        tl_path = os.path.join(phase_path, "training_list.npy")
        if os.path.exists(tl_path):
            tl = np.load(tl_path, allow_pickle=True)
            if isinstance(tl, np.ndarray) and tl.ndim == 1:
                # training_list is typically a list of [user, item, ...] entries
                for entry in tl:
                    if isinstance(entry, (list, np.ndarray)) and len(entry) >= 2:
                        uid, iid = int(entry[0]), int(entry[1])
                        if uid not in all_interactions:
                            all_interactions[uid] = set()
                        all_interactions[uid].add(iid)

    num_users = max(all_interactions.keys()) + 1 if all_interactions else 0
    total_inters = sum(len(v) for v in all_interactions.values())
    print(f"  [Collaborative] {num_users} users, {num_items} items, {total_inters} interactions")

    if total_inters == 0:
        print("  [Collaborative] WARNING: No interactions found, falling back to random")
        return generate_random_embeddings(num_items, emb_dim)

    # Build sparse interaction matrix
    rows, cols, vals = [], [], []
    for uid, items in all_interactions.items():
        for iid in items:
            if iid < num_items:
                rows.append(uid)
                cols.append(iid)
                vals.append(1.0)

    interaction_matrix = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(num_users, num_items),
        dtype=np.float32,
    )

    # Apply log-popularity weighting (TF-IDF style)
    item_popularity = np.array(interaction_matrix.sum(axis=0)).flatten() + 1
    idf_weights = np.log(num_users / item_popularity)
    idf_diag = sparse.diags(idf_weights)
    weighted_matrix = interaction_matrix @ idf_diag

    # SVD decomposition
    actual_dim = min(emb_dim, min(weighted_matrix.shape) - 1)
    print(f"  [Collaborative] Running SVD with k={actual_dim} ...")
    U, sigma, Vt = svds(weighted_matrix, k=actual_dim)

    # Item embeddings = Vt^T * diag(sigma)
    # Sort by singular values (svds returns in ascending order)
    idx = np.argsort(-sigma)
    sigma = sigma[idx]
    Vt = Vt[idx, :]

    item_embeddings = (Vt * sigma[:, np.newaxis]).T  # (num_items, actual_dim)
    item_embeddings = item_embeddings.astype(np.float32)

    # L2 normalize
    norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    item_embeddings = item_embeddings / norms

    print(f"  [Collaborative] Embeddings shape: {item_embeddings.shape}")
    print(f"  [Collaborative] Top-5 singular values: {sigma[:5]}")
    return item_embeddings


# ---------------------------------------------------------------------------
# Text-based embeddings
# ---------------------------------------------------------------------------

def generate_llama_embeddings(texts, llama_path, device, batch_size=32):
    """Generate embeddings using LLaMA model."""
    from transformers import LlamaTokenizer, LlamaModel

    print(f"Loading LLaMA model from {llama_path}...")
    tokenizer = LlamaTokenizer.from_pretrained(llama_path)
    model = LlamaModel.from_pretrained(
        llama_path,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating LLaMA embeddings"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling of last hidden state
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            all_embeddings.append(embeddings.cpu().float().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"LLaMA embeddings shape: {embeddings.shape}")
    return embeddings


def generate_sentence_transformer_embeddings(texts, device,
                                              model_name="all-MiniLM-L6-v2",
                                              model_path=None):
    """Generate embeddings using Sentence-Transformers (lightweight alternative).

    Args:
        texts: List of text strings to encode.
        device: Device to use (e.g., 'cuda:0').
        model_name: HuggingFace model name (used when model_path is None).
        model_path: Local path to a pre-downloaded model directory.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Installing sentence-transformers...")
        import subprocess
        subprocess.check_call(["pip", "install", "sentence-transformers"])
        from sentence_transformers import SentenceTransformer

    load_target = model_path if model_path else model_name
    print(f"Loading Sentence-Transformer model: {load_target}...")
    model = SentenceTransformer(load_target, device=device)

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print(f"Sentence-Transformer embeddings shape: {embeddings.shape}")
    return embeddings


def generate_random_embeddings(num_items, emb_dim=256):
    """Generate random embeddings (for quick testing only)."""
    print(f"Generating random embeddings: ({num_items}, {emb_dim})")
    np.random.seed(42)
    embeddings = np.random.randn(num_items, emb_dim).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate item embeddings")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to dataset directory (e.g., Reformer-data/games_11111_0.1)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (e.g., games_11111_0.1)")
    parser.add_argument("--method", type=str, default="collaborative",
                        choices=["llama", "sentence_transformer", "collaborative", "random"],
                        help="Embedding generation method (default: collaborative)")
    parser.add_argument("--llama_path", type=str, default=None,
                        help="Path to LLaMA model (required if method=llama)")
    parser.add_argument("--metadata_path", type=str, default=None,
                        help="Path to Amazon metadata file (.json/.jsonl/.gz) for text-based methods")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output path (default: <data_path>/combine_tdcb_maps.npy)")
    parser.add_argument("--output_format", type=str, default="both",
                        choices=["dict", "array", "both"],
                        help="Output format: dict ({id: emb}), array (N x D), or both")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--emb_dim", type=int, default=256,
                        help="Embedding dimension (for collaborative/random methods)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Local path to pre-downloaded sentence-transformer model directory")
    args = parser.parse_args()

    print("=" * 70)
    print("  Generate Item Embeddings")
    print("=" * 70)
    print(f"  Data path:  {args.data_path}")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Method:     {args.method}")
    print(f"  Emb dim:    {args.emb_dim}")
    print(f"  Device:     {args.device}")
    if args.metadata_path:
        print(f"  Metadata:   {args.metadata_path}")
    print()

    # Load item map
    id_to_original, num_items = load_item_map(args.data_path)
    print(f"  Total items: {num_items}")

    # Generate embeddings
    if args.method == "collaborative":
        embeddings = generate_collaborative_embeddings(
            args.data_path, num_items, args.emb_dim
        )

    elif args.method == "sentence_transformer":
        metadata = load_metadata(args.metadata_path)
        texts = build_text_descriptions(id_to_original, num_items, metadata)

        # Check if texts are mostly just ASIN codes (low quality)
        asin_count = sum(1 for t in texts if t.startswith("Game product "))
        if asin_count > num_items * 0.8 and not metadata:
            print(f"\n  WARNING: {asin_count}/{num_items} items have only ASIN codes (no metadata).")
            print("  Text-based embeddings will have low quality.")
            print("  Consider using --method collaborative or providing --metadata_path")
            print("  Proceeding anyway...\n")

        embeddings = generate_sentence_transformer_embeddings(
            texts, args.device, model_path=args.model_path
        )

    elif args.method == "llama":
        if args.llama_path is None:
            raise ValueError("--llama_path is required for LLaMA method")
        metadata = load_metadata(args.metadata_path)
        texts = build_text_descriptions(id_to_original, num_items, metadata)
        embeddings = generate_llama_embeddings(
            texts, args.llama_path, args.device, args.batch_size
        )

    elif args.method == "random":
        embeddings = generate_random_embeddings(num_items, args.emb_dim)

    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Save in requested format(s)
    saved_files = []

    if args.output_format in ("dict", "both"):
        # Save as dict format (combine_tdcb_maps.npy) - used by baseline scripts
        dict_path = args.output_path or os.path.join(args.data_path, "combine_tdcb_maps.npy")
        emb_dict = {i: embeddings[i] for i in range(len(embeddings))}
        np.save(dict_path, emb_dict)
        saved_files.append(("dict", dict_path))

    if args.output_format in ("array", "both"):
        # Save as array format ({dataset}.emb-llama-td.npy) - used by original TIGER
        array_path = os.path.join(args.data_path, f"{args.dataset}.emb-llama-td.npy")
        np.save(array_path, embeddings)
        saved_files.append(("array", array_path))

    print(f"\n{'=' * 70}")
    print(f"  Embedding generation complete!")
    print(f"  Shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    for fmt, path in saved_files:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  Saved ({fmt}): {path}  ({size_mb:.1f} MB)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

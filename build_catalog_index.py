"""
Step 1: Build DINOv2 Catalog Embedding Index
=============================================
Pre-computes DINOv2 embeddings for all 28,094 catalog products and
saves them to a FAISS index for fast nearest-neighbour retrieval.

Run:
    ./venv/bin/python3 build_catalog_index.py
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

# ── Config ───────────────────────────────────────────────────────────
CATALOG_JSONL = "dataset/product_catelog.jsonl"
PRODUCTS_DIR  = Path("data/products")
INDEX_PATH    = "index/catalog.faiss"     # FAISS index
SIGS_PATH     = "index/catalog_sigs.json" # Ordered list of product signatures
BATCH_SIZE    = 64
MODEL_ID      = "facebook/dinov2-small"   # 384-dim, fast & accurate
# ─────────────────────────────────────────────────────────────────────


def load_model():
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Loading {MODEL_ID} on device: {device}")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model     = AutoModel.from_pretrained(MODEL_ID, local_files_only=True).to(device)
    model.eval()
    return processor, model, device


def embed_batch(images: list[Image.Image], processor, model, device) -> np.ndarray:
    """Return L2-normalised [CLS] embeddings for a list of PIL images."""
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        out  = model(**inputs)
        embs = out.last_hidden_state[:, 0, :]      # [CLS] token
        embs = F.normalize(embs, p=2, dim=1)       # L2 normalise
    return embs.cpu().float().numpy()


def build_index():
    os.makedirs("index", exist_ok=True)

    # Load unique catalog signatures from JSONL
    seen, ordered_sigs = set(), []
    with open(CATALOG_JSONL) as f:
        for line in f:
            sig = json.loads(line.strip()).get("product", "").strip()
            if sig and sig not in seen:
                seen.add(sig)
                ordered_sigs.append(sig)

    print(f"Unique catalog products to index: {len(ordered_sigs)}")

    # Filter to images actually on disk (sanity check)
    valid_sigs = [s for s in ordered_sigs if (PRODUCTS_DIR / f"{s}.jpg").is_file()]
    missing = len(ordered_sigs) - len(valid_sigs)
    print(f"Found on disk: {len(valid_sigs)}  |  Missing (skip): {missing}")

    processor, model, device = load_model()

    # DINOv2-small produces 384-dim embeddings
    DIM = 384
    index = faiss.IndexFlatIP(DIM)   # Inner Product (cosine after L2-norm)

    all_embs, indexed_sigs = [], []
    t0 = time.time()

    for start in tqdm(range(0, len(valid_sigs), BATCH_SIZE), desc="Embedding catalog"):
        batch_sigs = valid_sigs[start : start + BATCH_SIZE]
        images = []
        good_sigs = []
        for sig in batch_sigs:
            try:
                img = Image.open(PRODUCTS_DIR / f"{sig}.jpg").convert("RGB")
                images.append(img)
                good_sigs.append(sig)
            except Exception:
                pass  # skip corrupt files

        if not images:
            continue

        embs = embed_batch(images, processor, model, device)
        all_embs.append(embs)
        indexed_sigs.extend(good_sigs)

    all_embs_np = np.vstack(all_embs)
    index.add(all_embs_np)

    # Save index and signature mapping
    faiss.write_index(index, INDEX_PATH)
    with open(SIGS_PATH, "w") as f:
        json.dump(indexed_sigs, f)

    elapsed = time.time() - t0
    print(f"\nIndex saved  → {INDEX_PATH}")
    print(f"Sigs saved   → {SIGS_PATH}")
    print(f"Total indexed: {index.ntotal} products")
    print(f"Embedding dim: {DIM}")
    print(f"Time elapsed:  {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    build_index()

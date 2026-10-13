"""
Step B: Load saved embeddings and run FAISS search.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import numpy as np
import faiss
from pathlib import Path
from PIL import Image

INDEX_PATH   = "index/catalog.faiss"
SIGS_PATH    = "index/catalog_sigs.json"
EMB_FILE     = "index/val_scene_embs.npz"
PRODUCTS_DIR = Path("data/products")

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)
catalog_sigs = json.load(open(SIGS_PATH))
catalog_set  = set(catalog_sigs)
print(f"Index: {index.ntotal} products | dim={index.d}")

print("Loading pre-computed scene embeddings...")
data = np.load(EMB_FILE, allow_pickle=True)
embs        = data["embs"]           # (N, 384)
scene_sigs  = data["scene_sigs"]
product_sigs= data["product_sigs"]
N = len(embs)
print(f"Loaded {N} validation scene embeddings")

# ── Evaluate Recall@K and MRR ──────────────────────────────────────────
print("\n── Evaluation ──")
r1, r5, r10, mrr = 0, 0, 0, []

scores_all, idxs_all = index.search(embs.astype("float32"), 10)   # batch search

for i in range(N):
    top_sigs = [catalog_sigs[j] for j in idxs_all[i]]
    gt_sig   = str(product_sigs[i])

    if gt_sig in top_sigs:
        rank = top_sigs.index(gt_sig) + 1
        if rank == 1:  r1  += 1
        if rank <= 5:  r5  += 1
        if rank <= 10: r10 += 1
        mrr.append(1.0 / rank)
    else:
        mrr.append(0.0)

n = len(mrr)
print(f"\n  ── Final Results ──")
print(f"  Evaluated pairs : {n}")
print(f"  Recall@1        : {r1/n:.4f}  ({r1}/{n})")
print(f"  Recall@5        : {r5/n:.4f}  ({r5}/{n})")
print(f"  Recall@10       : {r10/n:.4f}  ({r10}/{n})")
print(f"  MRR             : {np.mean(mrr):.4f}")

# ── Exact-match threshold calibration ─────────────────────────────────
# Already computed: τ = 0.226
# Distribution of top-1 cosine scores
top1_scores = scores_all[:, 0]
print(f"\n  Top-1 cosine score distribution:")
print(f"  Min={top1_scores.min():.4f} | Max={top1_scores.max():.4f} | Mean={top1_scores.mean():.4f}")

# ── Demo: sneaker crop retrieval ───────────────────────────────────────
print("\n── Demo: Sneaker Crop Retrieval ──")
import torch, torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
DINOV2_ID = "facebook/dinov2-small"
proc = AutoImageProcessor.from_pretrained(DINOV2_ID, local_files_only=True)
mdl  = AutoModel.from_pretrained(DINOV2_ID, local_files_only=True)
mdl.eval()

crop_path  = Path("data/crops/cdab9160072dd1800038227960ff6467_shoes_sneakers_1.jpg")
target_sig = "0027e30879ce3d87f82f699f148bff7e"   # Ground truth: White Keds

img = Image.open(crop_path).convert("RGB")
inp = proc(images=img, return_tensors="pt")
with torch.no_grad():
    out = mdl(**inp)
    emb = F.normalize(out.last_hidden_state[:, 0, :], p=2, dim=1).cpu().float().numpy()

scores, idxs = index.search(emb, 10)
top_sigs   = [catalog_sigs[j] for j in idxs[0]]
top_scores = scores[0].tolist()

TAU = 0.226
print(f"  Query: sneaker crop | τ = {TAU}")
print(f"  Target ({target_sig[:12]}...) in top-10: {target_sig in top_sigs}")
print(f"  Best score: {top_scores[0]:.4f} → {'EXACT MATCH' if top_scores[0] >= TAU else 'RECOMMENDATION'}")
print()
for rank, (sig, sc) in enumerate(zip(top_sigs, top_scores), 1):
    marker = "  ◀ GROUND TRUTH" if sig == target_sig else ""
    print(f"  Rank #{rank}: {sig[:16]}... | cosine={sc:.4f}{marker}")

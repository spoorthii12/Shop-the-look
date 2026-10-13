"""
Standalone Evaluation Script - minimal dependencies to avoid segfault
Compute Recall@1/5/10 and MRR on validation set using the FAISS index.
"""

import gc
import json
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# ── Config ───────────────────────
INDEX_PATH   = "index/catalog.faiss"
SIGS_PATH    = "index/catalog_sigs.json"
SCENES_DIR   = Path("data/scenes")
PRODUCTS_DIR = Path("data/products")
VAL_JSONL    = "dataset/validation.jsonl"
DINOV2_ID    = "facebook/dinov2-small"
# ─────────────────────────────────

print("Loading DINOv2 on cpu...")
processor = AutoImageProcessor.from_pretrained(DINOV2_ID, local_files_only=True)
model = AutoModel.from_pretrained(DINOV2_ID, local_files_only=True)
model.eval()

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)
catalog_sigs = json.load(open(SIGS_PATH))
catalog_set = set(catalog_sigs)
print(f"Index: {index.ntotal} products | dim={index.d}")


def embed(img: Image.Image) -> np.ndarray:
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
        emb = out.last_hidden_state[:, 0, :]
        emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().float().numpy()


# ── Calibrate τ ──────────────────────────────
print("\n── Calibrating τ ──")
val_records = [json.loads(l) for l in open(VAL_JSONL)]
all_in  = [r for r in val_records if r.get("product","") in catalog_set]
all_out = [r for r in val_records if r.get("product","") not in catalog_set]
sample  = all_in[:len(all_out)] + all_out

in_scores, out_scores = [], []
for rec in sample:
    scene_path = SCENES_DIR   / f"{rec['scene']}.jpg"
    prod_path  = PRODUCTS_DIR / f"{rec['product']}.jpg"
    if not scene_path.is_file() or not prod_path.is_file():
        continue
    s_emb = embed(Image.open(scene_path).convert("RGB"))
    p_emb = embed(Image.open(prod_path).convert("RGB"))
    score = float(np.dot(s_emb[0], p_emb[0]))
    if rec["product"] in catalog_set:
        in_scores.append(score)
    else:
        out_scores.append(score)
    del s_emb, p_emb
    gc.collect()

mu_pos = np.mean(in_scores)  if in_scores  else 0.0
mu_neg = np.mean(out_scores) if out_scores else 0.0
tau    = round((mu_pos + mu_neg) / 2, 3)
print(f"  In-catalog  (n={len(in_scores)}):  mean cosine = {mu_pos:.4f}")
print(f"  Out-of-catalog (n={len(out_scores)}): mean cosine = {mu_neg:.4f}")
print(f"  ✓ Calibrated τ = {tau}")


# ── Evaluation ───────────────────────────────
print("\n── Running Evaluation ──")
in_pairs = [(r["scene"], r["product"])
            for r in val_records
            if r.get("product","") in catalog_set
            and (SCENES_DIR / f"{r['scene']}.jpg").is_file()]

print(f"  In-catalog pairs: {len(in_pairs)}")

r1, r5, r10, mrr = 0, 0, 0, []

for i, (scene_sig, product_sig) in enumerate(in_pairs):
    scene_img = Image.open(SCENES_DIR / f"{scene_sig}.jpg").convert("RGB")
    q_emb = embed(scene_img)
    _, idxs = index.search(q_emb, 10)
    top_sigs = [catalog_sigs[j] for j in idxs[0]]

    if product_sig in top_sigs:
        rank = top_sigs.index(product_sig) + 1
        if rank == 1:  r1  += 1
        if rank <= 5:  r5  += 1
        if rank <= 10: r10 += 1
        mrr.append(1.0 / rank)
    else:
        mrr.append(0.0)

    del scene_img, q_emb
    gc.collect()

    if (i + 1) % 50 == 0 or (i + 1) == len(in_pairs):
        print(f"  [{i+1}/{len(in_pairs)}]  R@1={r1}  R@5={r5}  R@10={r10}")

n = len(mrr)
print(f"\n  ── Final Results ──")
print(f"  Evaluated pairs : {n}")
print(f"  Recall@1        : {r1/n:.4f}  ({r1}/{n})")
print(f"  Recall@5        : {r5/n:.4f}  ({r5}/{n})")
print(f"  Recall@10       : {r10/n:.4f}  ({r10}/{n})")
print(f"  MRR             : {np.mean(mrr):.4f}")

# ── Demo: sneaker crop retrieval ──────────────────────────────────────
print("\n── Demo: Sneaker Crop Retrieval ──")
crop_path = Path("data/crops/cdab9160072dd1800038227960ff6467_shoes_sneakers_1.jpg")
if crop_path.is_file():
    crop_emb  = embed(Image.open(crop_path).convert("RGB"))
    scores, idxs = index.search(crop_emb, 5)
    top_sigs  = [catalog_sigs[j] for j in idxs[0]]
    top_scores = scores[0].tolist()
    target_sig = "0027e30879ce3d87f82f699f148bff7e"  # White Keds (ground truth)

    print(f"  Query: sneaker crop | τ = {tau}")
    print(f"  Target in top-5: {target_sig in top_sigs}")
    for rank, (sig, sc) in enumerate(zip(top_sigs, top_scores), 1):
        match_marker = "  ◀ EXACT MATCH!" if sig == target_sig else ""
        print(f"  Rank #{rank}: {sig} | cosine={sc:.4f}{match_marker}")

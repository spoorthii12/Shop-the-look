"""
Step A: Pre-compute and save validation scene embeddings to disk.
Run this first, then run eval_search.py.
"""
import gc, json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

VAL_JSONL  = "dataset/validation.jsonl"
SCENES_DIR = Path("data/scenes")
OUT_FILE   = "index/val_scene_embs.npz"
DINOV2_ID  = "facebook/dinov2-small"

print("Loading DINOv2 on cpu...")
processor = AutoImageProcessor.from_pretrained(DINOV2_ID, local_files_only=True)
model = AutoModel.from_pretrained(DINOV2_ID, local_files_only=True)
model.eval()

val_records = [json.loads(l) for l in open(VAL_JSONL)]
catalog_sigs = set(json.load(open("index/catalog_sigs.json")))

# Only embed in-catalog pairs (scenes with a known catalog product)
pairs = [(r["scene"], r["product"])
         for r in val_records
         if r.get("product","") in catalog_sigs
         and (SCENES_DIR / f"{r['scene']}.jpg").is_file()]

print(f"Embedding {len(pairs)} validation scenes...")

scene_sigs, product_sigs, embs = [], [], []

for i, (scene_sig, product_sig) in enumerate(pairs):
    img = Image.open(SCENES_DIR / f"{scene_sig}.jpg").convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
        emb = out.last_hidden_state[:, 0, :]
        emb = F.normalize(emb, p=2, dim=1)
    embs.append(emb.cpu().float().numpy()[0])
    scene_sigs.append(scene_sig)
    product_sigs.append(product_sig)
    del img, inputs, out, emb
    gc.collect()
    if (i+1) % 50 == 0 or (i+1) == len(pairs):
        print(f"  Embedded {i+1}/{len(pairs)}")

embs_arr = np.stack(embs)
np.savez(OUT_FILE, embs=embs_arr,
         scene_sigs=np.array(scene_sigs),
         product_sigs=np.array(product_sigs))
print(f"Saved {embs_arr.shape} to {OUT_FILE}")

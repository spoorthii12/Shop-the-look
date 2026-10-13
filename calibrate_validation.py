"""
Calibrate and Evaluate Exact Match Decision Boundary on Validation Set
========================================================================
1. Evaluates Grounding DINO + DINOv2 retrieval performance (Recall@K, MRR).
2. Measures empirical similarity score distributions for:
   - Positive pairs: True matching crop vs. Ground-truth catalog packshot.
   - Negative pairs: Out-of-catalog queries (true item not in catalog) & false-match retrievals.
3. Performs threshold sweep tau in [0.20, 0.90] to calibrate Precision, Recall, and F1.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from pipeline import ShopTheLookPipeline

VAL_PATH = "dataset/validation.jsonl"
SIGS_PATH = "index/catalog_sigs.json"
MATRIX_PATH = "index/catalog_matrix.npy"
SCENES_DIR = Path("data/scenes")
PRODUCTS_DIR = Path("data/products")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def run_calibration(num_in_catalog: int = 50, num_out_catalog: int = 42):
    print("=" * 70)
    print("CALIBRATING EXACT MATCH DECISION BOUNDARY & EVALUATING RETRIEVAL")
    print("=" * 70)

    # 1. Initialize Pipeline
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipeline = ShopTheLookPipeline(device=device)

    # 2. Load Validation Dataset
    with open(VAL_PATH) as f:
        val_pairs = [json.loads(line) for line in f]

    with open(SIGS_PATH) as f:
        catalog_sigs = json.load(f)
    catalog_sig_to_idx = {sig: idx for idx, sig in enumerate(catalog_sigs)}

    in_catalog_pairs = [p for p in val_pairs if p["product"] in catalog_sig_to_idx]
    out_catalog_pairs = [p for p in val_pairs if p["product"] not in catalog_sig_to_idx]

    print(f"Total Validation Pairs: {len(val_pairs)}")
    print(f"In-Catalog Pairs: {len(in_catalog_pairs)} (Evaluating {min(num_in_catalog, len(in_catalog_pairs))})")
    print(f"Out-of-Catalog Pairs: {len(out_catalog_pairs)} (Evaluating {min(num_out_catalog, len(out_catalog_pairs))})")

    eval_in = in_catalog_pairs[:num_in_catalog]
    eval_out = out_catalog_pairs[:num_out_catalog]

    # 3. Evaluate In-Catalog Queries (Positive Match & Rank Retrieval)
    print("\n--- 1. Evaluating In-Catalog Pairs (True Matches Exist) ---")
    pos_gt_scores = []         # Direct similarity between target packshot and best matching crop
    in_cat_top1_scores = []     # Top-1 retrieved catalog score
    in_cat_ranks = []          # Rank of ground-truth product
    mrr_scores = []
    r1, r5, r10, r20 = 0, 0, 0, 0

    for item in tqdm(eval_in, desc="In-Catalog"):
        scene_path = SCENES_DIR / f"{item['scene']}.jpg"
        prod_path = PRODUCTS_DIR / f"{item['product']}.jpg"
        target_sig = item["product"]
        target_cat_idx = catalog_sig_to_idx[target_sig]

        if not scene_path.exists() or not prod_path.exists():
            continue

        # Target packshot embedding
        target_emb = pipeline.catalog_matrix[target_cat_idx]

        # Scene Garment Detections
        scene_img = Image.open(scene_path).convert("RGB")
        detections = pipeline.detect_garments(scene_img, threshold=0.25)
        if not detections:
            in_cat_ranks.append(99999)
            mrr_scores.append(0.0)
            continue

        # Find which detected crop best matches the target garment
        crop_embs = [pipeline.embed_image(d["crop"]).flatten() for d in detections]
        crop_gt_sims = [float(np.dot(ce, target_emb)) for ce in crop_embs]
        best_crop_idx = int(np.argmax(crop_gt_sims))
        best_crop_emb = crop_embs[best_crop_idx]
        pos_gt_scores.append(crop_gt_sims[best_crop_idx])

        # Search the entire 28,094 catalog with this best garment crop
        all_scores = np.dot(pipeline.catalog_matrix, best_crop_emb)
        top1_score = float(np.max(all_scores))
        in_cat_top1_scores.append(top1_score)

        # Determine Rank of Target Product
        target_score = all_scores[target_cat_idx]
        rank = int(np.sum(all_scores > target_score)) + 1
        in_cat_ranks.append(rank)

        if rank == 1:
            r1 += 1
        if rank <= 5:
            r5 += 1
        if rank <= 10:
            r10 += 1
        if rank <= 20:
            r20 += 1
        mrr_scores.append(1.0 / rank)

    total_in = len(in_cat_ranks)
    print(f"\nIn-Catalog Retrieval Metrics (N={total_in}):")
    print(f"  Recall@1:  {r1/total_in*100:.2f}% ({r1}/{total_in})")
    print(f"  Recall@5:  {r5/total_in*100:.2f}% ({r5}/{total_in})")
    print(f"  Recall@10: {r10/total_in*100:.2f}% ({r10}/{total_in})")
    print(f"  Recall@20: {r20/total_in*100:.2f}% ({r20}/{total_in})")
    print(f"  MRR:       {np.mean(mrr_scores):.4f}")

    # 4. Evaluate Out-of-Catalog Queries (True Match Does NOT Exist in Catalog)
    print("\n--- 2. Evaluating Out-of-Catalog Pairs (True Match Absent) ---")
    out_cat_top1_scores = []    # Top-1 retrieved catalog score when product does not exist

    for item in tqdm(eval_out, desc="Out-of-Catalog"):
        scene_path = SCENES_DIR / f"{item['scene']}.jpg"
        if not scene_path.exists():
            continue

        scene_img = Image.open(scene_path).convert("RGB")
        detections = pipeline.detect_garments(scene_img, threshold=0.25)
        if not detections:
            continue

        # Embed crops and query catalog
        for d in detections:
            c_emb = pipeline.embed_image(d["crop"]).flatten()
            scores = np.dot(pipeline.catalog_matrix, c_emb)
            out_cat_top1_scores.append(float(np.max(scores)))

    # 5. Measure Negative Non-Matching Catalog Samples
    print("\n--- 3. Sampling Non-Matching Background Pairs ---")
    random_neg_scores = []
    # Take random crops against 500 random catalog products
    if pos_gt_scores:
        sample_crops = [pipeline.embed_image(detections[0]["crop"]).flatten()] if detections else []
        for _ in range(500):
            rand_idx = np.random.randint(0, len(catalog_sigs))
            rand_emb = pipeline.catalog_matrix[rand_idx]
            random_neg_scores.append(float(np.dot(best_crop_emb, rand_emb)))

    # 6. Empirical Score Summary
    print("\n" + "=" * 70)
    print("SCORE DISTRIBUTION COMPARISON")
    print("=" * 70)
    print(f"Positive Ground-Truth Pairs (N={len(pos_gt_scores)}):")
    print(f"  Mean: {np.mean(pos_gt_scores):.4f} | Median: {np.median(pos_gt_scores):.4f} | Range: [{np.min(pos_gt_scores):.4f}, {np.max(pos_gt_scores):.4f}]")

    print(f"\nOut-of-Catalog Top-1 False Matches (N={len(out_cat_top1_scores)}):")
    print(f"  Mean: {np.mean(out_cat_top1_scores):.4f} | Median: {np.median(out_cat_top1_scores):.4f} | Range: [{np.min(out_cat_top1_scores):.4f}, {np.max(out_cat_top1_scores):.4f}]")

    print(f"\nRandom Background Catalog Items (N={len(random_neg_scores)}):")
    print(f"  Mean: {np.mean(random_neg_scores):.4f} | Median: {np.median(random_neg_scores):.4f} | Range: [{np.min(random_neg_scores):.4f}, {np.max(random_neg_scores):.4f}]")

    # 7. Threshold Calibration Sweep
    print("\n" + "=" * 70)
    print("CALIBRATING EXACT MATCH THRESHOLD (τ) SWEEP")
    print("=" * 70)
    print(f"{'Threshold τ':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'FP Rate (Out-of-Cat)':<22}")
    print("-" * 72)

    sweep_results = []
    # For each tau, define:
    # TP: in-catalog query where rank == 1 and top1_score >= tau
    # FP_in: in-catalog query where rank > 1 and top1_score >= tau
    # FP_out: out-of-catalog query where top1_score >= tau
    # FN: in-catalog query where target was matched (rank==1) but top1_score < tau, or not top 1

    for tau_cand in np.arange(0.20, 0.95, 0.05):
        tp = sum(1 for rank, sc in zip(in_cat_ranks, in_cat_top1_scores) if rank == 1 and sc >= tau_cand)
        fp_in = sum(1 for rank, sc in zip(in_cat_ranks, in_cat_top1_scores) if rank > 1 and sc >= tau_cand)
        fp_out = sum(1 for sc in out_cat_top1_scores if sc >= tau_cand)
        fp = fp_in + fp_out

        fn = sum(1 for rank, sc in zip(in_cat_ranks, in_cat_top1_scores) if not (rank == 1 and sc >= tau_cand))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / total_in if total_in > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        fp_rate_out = fp_out / len(out_cat_top1_scores) if out_cat_top1_scores else 0.0

        marker = " ◀ (Provisional)" if abs(tau_cand - 0.226) < 0.03 else ""
        print(f"τ = {tau_cand:.2f}     | {prec*100:>6.2f}%   | {rec*100:>6.2f}%   | {f1:>8.4f} | {fp_rate_out*100:>6.1f}% ({fp_out}/{len(out_cat_top1_scores)}){marker}")
        sweep_results.append({
            "tau": round(float(tau_cand), 2),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "fp_out_rate": round(fp_rate_out, 4)
        })

    # Save Results
    out_payload = {
        "num_in_catalog": total_in,
        "num_out_catalog": len(out_cat_top1_scores),
        "recall_at_1": r1 / total_in,
        "recall_at_5": r5 / total_in,
        "recall_at_10": r10 / total_in,
        "recall_at_20": r20 / total_in,
        "mrr": float(np.mean(mrr_scores)),
        "pos_gt_scores_summary": {
            "mean": float(np.mean(pos_gt_scores)),
            "median": float(np.median(pos_gt_scores)),
            "min": float(np.min(pos_gt_scores)),
            "max": float(np.max(pos_gt_scores))
        },
        "out_cat_scores_summary": {
            "mean": float(np.mean(out_cat_top1_scores)),
            "median": float(np.median(out_cat_top1_scores)),
            "min": float(np.min(out_cat_top1_scores)),
            "max": float(np.max(out_cat_top1_scores))
        },
        "sweep_results": sweep_results
    }

    with open(RESULTS_DIR / "calibration_report.json", "w") as f:
        json.dump(out_payload, f, indent=2)

    print(f"\nCalibration report saved to {RESULTS_DIR / 'calibration_report.json'}")

if __name__ == "__main__":
    run_calibration(num_in_catalog=40, num_out_catalog=20)

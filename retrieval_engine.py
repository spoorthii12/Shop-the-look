"""
Step 2 & 3: Retrieval Engine + Explainability + Evaluation
===========================================================
- Loads the pre-built FAISS index.
- Accepts a scene crop (PIL Image or path) and retrieves top-K catalog products.
- Applies the exact-match threshold τ calibrated on the validation set.
- Generates explainability attributes (Color, Category, Visual Similarity).
- Runs full Recall@K / MRR evaluation over all 330 validation pairs.

Usage:
    ./venv/bin/python3 retrieval_engine.py
"""

import os
import gc
import json
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from pathlib import Path
from PIL import Image
from transformers import (
    AutoImageProcessor, AutoModel,
    AutoProcessor, AutoModelForZeroShotObjectDetection
)

# ── Config ───────────────────────────────────────────────────────────
INDEX_PATH    = "index/catalog.faiss"
SIGS_PATH     = "index/catalog_sigs.json"
PRODUCTS_DIR  = Path("data/products")
SCENES_DIR    = Path("data/scenes")
CROPS_DIR     = Path("data/crops")
VAL_JSONL     = "dataset/validation.jsonl"
DINOV2_ID     = "facebook/dinov2-small"
GDINO_ID      = "IDEA-Research/grounding-dino-tiny"
FASHION_PROMPT = "dress . top . shirt . blouse . jacket . coat . pants . jeans . skirt . shorts . shoes . sneakers . boots . handbag . bag ."
TOP_K         = 10
# ─────────────────────────────────────────────────────────────────────


# ── 1.  Embedding Model ───────────────────────────────────────────────
class EmbeddingModel:
    def __init__(self):
        # Force CPU to avoid MPS memory segfaults in large evaluation loops
        self.device = "cpu"
        print(f"Loading {DINOV2_ID} on device: {self.device}")
        self.processor = AutoImageProcessor.from_pretrained(DINOV2_ID, local_files_only=True)
        self.model     = AutoModel.from_pretrained(DINOV2_ID, local_files_only=True).to(self.device)
        self.model.eval()

    def embed(self, images) -> np.ndarray:
        """Embed a single PIL image or a list of PIL images → L2-normalised vector."""
        if isinstance(images, Image.Image):
            images = [images]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out  = self.model(**inputs)
            embs = out.last_hidden_state[:, 0, :]
            embs = F.normalize(embs, p=2, dim=1)
        return embs.cpu().float().numpy()


# ── 2.  FAISS Catalog Index ──────────────────────────────────────────
class CatalogIndex:
    def __init__(self):
        print("Loading FAISS index...")
        self.index = faiss.read_index(INDEX_PATH)
        with open(SIGS_PATH) as f:
            self.sigs = json.load(f)
        print(f"Index loaded: {self.index.ntotal} products, dim={self.index.d}")

    def search(self, query_emb: np.ndarray, k: int = TOP_K):
        """Return (scores, product_sigs) for the top-k nearest catalog items."""
        scores, idxs = self.index.search(query_emb, k)
        result_sigs   = [self.sigs[i] for i in idxs[0]]
        result_scores = scores[0].tolist()
        return result_scores, result_sigs


# ── 3.  Explainability Engine ────────────────────────────────────────
def dominant_colors(img: Image.Image, k: int = 3) -> list:
    """Extract k dominant RGB colors via simple pixel sampling."""
    img_small = img.resize((60, 60)).convert("RGB")
    pixels    = np.array(img_small).reshape(-1, 3).astype(float)
    # k-means lite: pick k evenly-spaced pixels as cluster centers
    idxs    = np.linspace(0, len(pixels) - 1, k, dtype=int)
    centers = pixels[idxs]
    for _ in range(8):
        dists = np.linalg.norm(pixels[:, None] - centers[None], axis=2)
        labels = dists.argmin(axis=1)
        for j in range(k):
            pts = pixels[labels == j]
            if len(pts) > 0:
                centers[j] = pts.mean(axis=0)
    return [tuple(map(int, c)) for c in centers]


def rgb_to_name(rgb: tuple) -> str:
    """Heuristic color family labelling from RGB."""
    r, g, b = rgb
    if r > 200 and g < 100 and b < 100: return "Red"
    if r > 200 and g > 150 and b < 80:  return "Orange / Tan"
    if r > 180 and g > 160 and b < 80:  return "Yellow"
    if r < 100 and g > 150 and b < 100: return "Green"
    if r < 100 and g < 100 and b > 180: return "Blue"
    if r > 120 and g < 80  and b > 120: return "Purple"
    if r > 200 and g < 100 and b > 150: return "Pink"
    if r > 180 and g > 170 and b > 160 and max(r,g,b)-min(r,g,b)<40: return "White / Cream"
    if r < 80  and g < 80  and b < 80:  return "Black"
    if r < 160 and g < 160 and b < 160 and max(r,g,b)-min(r,g,b)<40: return "Grey"
    if r > 130 and g > 90  and b < 70:  return "Brown / Beige"
    if r > 100 and g < 100 and b > 60:  return "Maroon"
    if r < 100 and g > 130 and b > 150: return "Teal / Navy"
    return "Multi-color"


def color_similarity_score(img1: Image.Image, img2: Image.Image) -> float:
    """Rough color similarity: mean Euclidean distance in RGB space (0-100%)."""
    c1 = np.array(img1.resize((60, 60)).convert("RGB")).reshape(-1, 3).mean(axis=0)
    c2 = np.array(img2.resize((60, 60)).convert("RGB")).reshape(-1, 3).mean(axis=0)
    dist = np.linalg.norm(c1 - c2)
    return max(0, round(100 - dist / 4.42, 1))   # max dist ≈ 442


def aspect_ratio_label(img: Image.Image) -> str:
    w, h = img.size
    ar = h / w
    if ar < 0.6:   return "Wide / Landscape"
    if ar < 1.05:  return "Square"
    if ar < 1.8:   return "Portrait (Half-body)"
    return "Tall Portrait (Full-body)"


def explain_match(query_img: Image.Image, catalog_sig: str, cos_score: float) -> dict:
    """Generate multi-attribute similarity explanation for a catalog match."""
    cat_path = PRODUCTS_DIR / f"{catalog_sig}.jpg"
    cat_img  = Image.open(cat_path).convert("RGB")

    qcolors  = dominant_colors(query_img)
    ccolors  = dominant_colors(cat_img)
    qnames   = [rgb_to_name(c) for c in qcolors]
    cnames   = [rgb_to_name(c) for c in ccolors]

    color_pct   = color_similarity_score(query_img, cat_img)
    shared_cols = list(set(qnames) & set(cnames))
    visual_pct  = round(cos_score * 100, 1)

    return {
        "visual_similarity_pct": visual_pct,
        "color_match_pct":       color_pct,
        "query_colors":          qnames,
        "catalog_colors":        cnames,
        "shared_color_families": shared_cols if shared_cols else ["Different palette"],
        "catalog_composition":   aspect_ratio_label(cat_img),
    }


# ── 4.  Full Retrieval Pipeline ──────────────────────────────────────
class ShopTheLookRetriever:
    """
    End-to-end retriever:
    1. Scene → Grounding DINO → garment crops
    2. DINOv2 embed each crop
    3. FAISS search against 28k catalog
    4. Apply exact-match threshold τ
    5. Return ranked results + explanations
    """

    # Calibrated threshold (will be set after running calibrate_tau())
    TAU = 0.50

    def __init__(self):
        self.emb_model  = EmbeddingModel()
        self.catalog    = CatalogIndex()
        self._dino_proc = None
        self._dino_mod  = None

    def _load_gdino(self):
        if self._dino_proc is None:
            print("Loading Grounding DINO...")
            self._dino_proc = AutoProcessor.from_pretrained(GDINO_ID, local_files_only=True)
            self._dino_mod  = AutoModelForZeroShotObjectDetection.from_pretrained(
                GDINO_ID, local_files_only=True
            ).to(self.emb_model.device)
            self._dino_mod.eval()

    def detect_garments(self, scene_img: Image.Image, scene_sig: str = "unknown") -> list:
        """Return list of {label, score, box, crop_img} for detected garments."""
        self._load_gdino()
        w, h = scene_img.size
        inputs = self._dino_proc(
            images=scene_img, text=FASHION_PROMPT, return_tensors="pt"
        ).to(self.emb_model.device)
        with torch.no_grad():
            outputs = self._dino_mod(**inputs)
        results = self._dino_proc.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.25,
            text_threshold=0.20,
            target_sizes=[(h, w)]
        )[0]

        garments = []
        for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
            box = [int(v.item()) for v in box]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            crop = scene_img.crop(box)
            garments.append({
                "label": label,
                "score": round(score.item(), 3),
                "box":   box,
                "crop":  crop,
            })
        return garments

    def retrieve(
        self,
        query_img: Image.Image,
        top_k: int = TOP_K,
        tau: float = None
    ) -> dict:
        """
        Given a garment crop image, retrieve top-k matches with explanations.
        Returns a dict with keys: exact_match, results.
        """
        tau = tau if tau is not None else self.TAU
        emb = self.emb_model.embed(query_img)
        scores, sigs = self.catalog.search(emb, k=top_k)

        best_score = scores[0]
        is_exact   = best_score >= tau

        results = []
        for rank, (sig, score) in enumerate(zip(sigs, scores), start=1):
            explanation = explain_match(query_img, sig, score)
            results.append({
                "rank":        rank,
                "product_sig": sig,
                "cos_score":   round(float(score), 4),
                "exact_match": rank == 1 and is_exact,
                "explanation": explanation,
            })
        return {
            "exact_match_found": is_exact,
            "best_score":        round(float(best_score), 4),
            "tau_threshold":     tau,
            "results":           results,
        }


# ── 5.  Threshold Calibration using Validation Set ──────────────────
def calibrate_tau(retriever: ShopTheLookRetriever) -> float:
    """
    Embed both the full scene image and its ground-truth product image,
    then compute cosine similarity.
    - In-catalog pairs  → positive distribution
    - Out-of-catalog pairs → negative distribution
    τ = midpoint of the two means.
    """
    print("\n── Calibrating exact-match threshold (τ) ──")
    val_records = [json.loads(l) for l in open(VAL_JSONL)]
    catalog_set = set(retriever.catalog.sigs)

    in_catalog_scores  = []
    out_catalog_scores = []

    # Use a mixed sample: take all out-of-catalog + equal number in-catalog
    all_in  = [r for r in val_records if r.get("product","") in catalog_set]
    all_out = [r for r in val_records if r.get("product","") not in catalog_set]
    sample  = all_in[:len(all_out)] + all_out          # balanced sample
    print(f"  Calibration sample: {len(all_in[:len(all_out)])} in-catalog + {len(all_out)} out-of-catalog")

    for rec in sample:
        scene_sig   = rec.get("scene", "")
        product_sig = rec.get("product", "")
        scene_path  = SCENES_DIR   / f"{scene_sig}.jpg"
        prod_path   = PRODUCTS_DIR / f"{product_sig}.jpg"
        if not scene_path.is_file() or not prod_path.is_file():
            continue

        scene_img = Image.open(scene_path).convert("RGB")
        prod_img  = Image.open(prod_path).convert("RGB")
        scene_emb = retriever.emb_model.embed(scene_img)
        prod_emb  = retriever.emb_model.embed(prod_img)
        score     = float(np.dot(scene_emb[0], prod_emb[0]))

        if product_sig in catalog_set:
            in_catalog_scores.append(score)
        else:
            out_catalog_scores.append(score)

    mu_pos = np.mean(in_catalog_scores)  if in_catalog_scores  else 0.0
    mu_neg = np.mean(out_catalog_scores) if out_catalog_scores else 0.0
    tau    = round((mu_pos + mu_neg) / 2, 3)

    print(f"  In-catalog  (n={len(in_catalog_scores)}):  mean cosine = {mu_pos:.4f}")
    print(f"  Out-of-catalog (n={len(out_catalog_scores)}): mean cosine = {mu_neg:.4f}")
    print(f"  ✓ Calibrated τ = {tau}")
    return tau


# ── 6.  Full Evaluation Benchmark ───────────────────────────────────
def run_evaluation(retriever: ShopTheLookRetriever) -> dict:
    """Compute Recall@1/5/10 and MRR on all in-catalog validation pairs."""
    print("\n── Running Full Evaluation ──")
    val_records = [json.loads(l) for l in open(VAL_JSONL)]
    catalog_set = set(retriever.catalog.sigs)

    in_catalog_pairs = [(r["scene"], r["product"])
                        for r in val_records if r.get("product","") in catalog_set
                        and (SCENES_DIR / f"{r['scene']}.jpg").is_file()]

    print(f"  In-catalog pairs to evaluate: {len(in_catalog_pairs)}")

    r1, r5, r10, mrr_scores = 0, 0, 0, []

    for i, (scene_sig, product_sig) in enumerate(in_catalog_pairs):
        scene_img  = Image.open(SCENES_DIR / f"{scene_sig}.jpg").convert("RGB")
        query_emb  = retriever.emb_model.embed(scene_img)
        scores, sigs = retriever.catalog.search(query_emb, k=10)

        if product_sig in sigs:
            rank = sigs.index(product_sig) + 1
            if rank == 1:  r1  += 1
            if rank <= 5:  r5  += 1
            if rank <= 10: r10 += 1
            mrr_scores.append(1.0 / rank)
        else:
            mrr_scores.append(0.0)

        # Free memory every 50 iterations
        if i % 50 == 0:
            gc.collect()
            print(f"  ... {i+1}/{len(in_catalog_pairs)} evaluated")

    n = len(mrr_scores)
    metrics = {
        "evaluated_pairs": n,
        "Recall@1":  round(r1  / n, 4) if n else 0,
        "Recall@5":  round(r5  / n, 4) if n else 0,
        "Recall@10": round(r10 / n, 4) if n else 0,
        "MRR":       round(float(np.mean(mrr_scores)), 4) if n else 0,
    }
    print("\n  ── Results ──")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    retriever = ShopTheLookRetriever()

    # Step 1: Calibrate τ
    tau = calibrate_tau(retriever)
    retriever.TAU = tau

    # Step 2: Evaluate on the full validation set
    metrics = run_evaluation(retriever)

    # Step 3: Demo retrieval on the sneaker scene crop
    print("\n── Demo Retrieval: Sneaker Scene ──")
    crop_path = Path("data/crops/cdab9160072dd1800038227960ff6467_shoes_sneakers_1.jpg")
    if crop_path.is_file():
        crop_img = Image.open(crop_path).convert("RGB")
        result = retriever.retrieve(crop_img, top_k=5, tau=tau)
        print(f"Query: sneaker crop | Exact match: {result['exact_match_found']} | Best score: {result['best_score']:.4f}")
        for r in result["results"][:3]:
            expl = r["explanation"]
            print(f"  Rank #{r['rank']}: {r['product_sig'][:16]}... | "
                  f"Visual={expl['visual_similarity_pct']}% | "
                  f"Color={expl['color_match_pct']}% | "
                  f"Shared palette: {expl['shared_color_families']}")


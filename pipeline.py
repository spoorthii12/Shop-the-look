"""
Shop-the-Look: Unified Detection & Retrieval Pipeline
====================================================
End-to-end inference engine combining:
1. Grounding DINO garment detection & outfit parsing
2. DINOv2 visual metric feature representation
3. FAISS catalog index vector search (28,094 products)
4. Decision logic: Exact Match (score >= tau) vs. Closest Recommendations (score < tau)
5. Multi-attribute explainability engine (color, category, pattern, silhouette)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoProcessor, AutoModelForZeroShotObjectDetection,
    AutoImageProcessor, AutoModel
)

from explainability import ExplainabilityEngine

# ── Defaults & Paths ──────────────────────────────────────────────────
INDEX_PATH = "index/catalog.faiss"
SIGS_PATH = "index/catalog_sigs.json"
PRODUCTS_DIR = Path("data/products")
GDINO_ID = "IDEA-Research/grounding-dino-tiny"
DINOV2_ID = "facebook/dinov2-small"
# Provisional threshold pending full empirical calibration against labelled matching/non-matching pairs
DEFAULT_PROVISIONAL_TAU = 0.226

FASHION_TEXT_PROMPT = (
    "clothing . dress . gown . jumpsuit . romper . overalls . "
    "top . shirt . t-shirt . tee . blouse . crop top . tank top . "
    "jacket . coat . blazer . cardigan . sweater . hoodie . sweatshirt . vest . "
    "pants . jeans . trousers . leggings . sweatpants . shorts . skirt . "
    "shoes . sneakers . boots . heels . sandals . loafers . flats . slippers . "
    "sunglasses . glasses . spectacles . eyewear . "
    "bag . handbag . purse . tote . backpack . clutch . crossbody bag . "
    "hat . cap . beanie . beret . headband . "
    "scarf . belt . watch . necklace . bracelet . earrings . tie . socks ."
)


class ShopTheLookPipeline:
    """
    Unified Shop-the-Look system.
    """

    def __init__(
        self,
        index_path: str = INDEX_PATH,
        sigs_path: str = SIGS_PATH,
        products_dir: Union[str, Path] = PRODUCTS_DIR,
        tau: float = DEFAULT_PROVISIONAL_TAU,
        device: Optional[str] = None
    ):
        self.provisional_tau = tau
        self.tau = tau
        self.products_dir = Path(products_dir)

        # Device selection: use cpu for inference to guarantee stability on macOS
        self.device = device or "cpu"
        print(f"[Pipeline] Initializing pipeline on device: {self.device}...")

        # 1. Load Grounding DINO Detector
        print(f"[Pipeline] Loading Grounding DINO ({GDINO_ID})...")
        self.dino_proc = AutoProcessor.from_pretrained(GDINO_ID, local_files_only=True)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GDINO_ID, local_files_only=True
        ).to(self.device)
        self.dino_model.eval()

        # 2. Load DINOv2 Feature Extractor
        print(f"[Pipeline] Loading DINOv2 ({DINOV2_ID})...")
        self.dinov2_proc = AutoImageProcessor.from_pretrained(DINOV2_ID, local_files_only=True)
        self.dinov2_model = AutoModel.from_pretrained(DINOV2_ID, local_files_only=True).to(self.device)
        self.dinov2_model.eval()

        # 3. Load Catalog Index
        matrix_path = Path("index/catalog_matrix.npy")
        if matrix_path.exists():
            print(f"[Pipeline] Loading pre-extracted catalog matrix from {matrix_path}...")
            self.catalog_matrix = np.load(matrix_path)
            self.use_numpy = True
        else:
            print(f"[Pipeline] Loading FAISS catalog index from {index_path}...")
            self.index = faiss.read_index(index_path)
            self.catalog_matrix = self.index.reconstruct_n(0, self.index.ntotal)
            self.use_numpy = True

        with open(sigs_path, "r") as f:
            self.catalog_sigs = json.load(f)

        # 4. Load Precomputed Aspect Ratios & Category Masks for Filtering
        ar_path = Path("index/catalog_aspect_ratios.npy")
        if ar_path.exists():
            self.catalog_ars = np.load(ar_path)
        else:
            self.catalog_ars = None

        glasses_path = Path("index/catalog_is_glasses.npy")
        if glasses_path.exists():
            self.catalog_is_glasses = np.load(glasses_path)
            print(f"[Pipeline] Loaded {np.sum(self.catalog_is_glasses)} verified glasses packshots for category isolation.")
        else:
            self.catalog_is_glasses = None

        print(f"[Pipeline] Ready! {len(self.catalog_sigs)} products indexed (dim={self.catalog_matrix.shape[1]}).")
        self.explainer = ExplainabilityEngine()

    def embed_image(self, pil_image: Image.Image) -> np.ndarray:
        """Extract L2-normalized 384-dimensional DINOv2 embedding with aspect-ratio preserving square padding."""
        from PIL import ImageOps
        img_rgb = pil_image.convert("RGB")
        max_side = max(img_rgb.size)
        padded = ImageOps.pad(img_rgb, (max_side, max_side), color='white')
        inputs = self.dinov2_proc(images=padded, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.dinov2_model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]
            emb = F.normalize(emb, p=2, dim=1)
        return emb.cpu().float().numpy()

    def detect_garments(
        self,
        scene_image: Image.Image,
        threshold: float = 0.25,
        text_threshold: float = 0.20
    ) -> List[Dict]:
        """
        Parse outfit using Grounding DINO to extract garment bounding boxes and crops.
        """
        img_rgb = scene_image.convert("RGB")
        w, h = img_rgb.size

        inputs = self.dino_proc(
            images=img_rgb,
            text=FASHION_TEXT_PROMPT,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        results = self.dino_proc.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[(h, w)]
        )[0]

        import torchvision.ops as ops

        # 1. Apply Non-Maximum Suppression (NMS) to eliminate duplicate/overlapping boxes
        boxes_tensor = results["boxes"]
        scores_tensor = results["scores"]
        if len(boxes_tensor) > 0:
            keep_indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.45).cpu().tolist()
        else:
            keep_indices = []

        garments = []
        for idx in keep_indices:
            box = results["boxes"][idx]
            score = results["scores"][idx]
            raw_label = results["text_labels"][idx]

            b = [int(v.item()) for v in box]
            bw = b[2] - b[0]
            bh = b[3] - b[1]

            # Discard tiny artifacts or invalid boxes
            if bw < 20 or bh < 20 or bw * bh < 600:
                continue

            crop = img_rgb.crop(b)
            clean_label = self._clean_category_label(raw_label)

            garments.append({
                "label": clean_label,
                "confidence": round(float(score.item()), 3),
                "bbox": b,
                "crop": crop
            })

        return garments

    def _clean_category_label(self, raw_label: str) -> str:
        """Standardize compound Grounding DINO token matches into clean user-facing labels."""
        lbl = raw_label.lower().strip()
        if any(k in lbl for k in ["sunglasses", "glasses", "spectacles", "eyewear"]):
            return "Glasses / Eyewear"
        if any(k in lbl for k in ["sneaker", "shoes", "boot", "heels", "sandals", "loafers", "flats", "slippers"]):
            return "Shoes / Footwear"
        if any(k in lbl for k in ["pants", "jeans", "trousers", "leggings", "sweatpants"]):
            return "Pants / Jeans"
        if any(k in lbl for k in ["skirt", "shorts"]):
            return "Skirt / Shorts"
        if any(k in lbl for k in ["dress", "gown", "jumpsuit", "romper"]):
            return "Dress / One-Piece"
        if any(k in lbl for k in ["t-shirt", "tee", "shirt", "blouse", "crop top", "tank top", "top"]):
            return "Top / Shirt"
        if any(k in lbl for k in ["jacket", "coat", "blazer", "cardigan", "hoodie", "sweatshirt", "vest"]):
            return "Jacket / Outerwear"
        if any(k in lbl for k in ["bag", "handbag", "purse", "tote", "backpack", "clutch"]):
            return "Bag / Handbag"
        if any(k in lbl for k in ["hat", "cap", "beanie", "beret", "headband"]):
            return "Hat / Headwear"
        if "belt" in lbl:
            return "Belt"
        if "scarf" in lbl:
            return "Scarf"
        if "watch" in lbl:
            return "Watch"
        if any(k in lbl for k in ["necklace", "bracelet", "earrings", "jewelry"]):
            return "Jewelry / Accessories"

        # Fallback: clean subwords (e.g. ##s) and title case
        cleaned = "".join([c if c.isalnum() or c in " -/" else " " for c in raw_label])
        words = [w for w in cleaned.split() if not w.startswith("#")]
        return " ".join(words).title() if words else "Fashion Item"

    def annotate_scene(self, scene_image: Image.Image, detections: List[Dict]) -> Image.Image:
        """Draw bounding boxes and category labels on scene image."""
        annotated = scene_image.convert("RGB").copy()
        draw = ImageDraw.Draw(annotated)

        palette = [
            (236, 72, 153),  # Pink
            (56, 189, 248),  # Sky blue
            (16, 185, 129),  # Emerald
            (245, 158, 11),  # Amber
            (139, 92, 246),  # Violet
        ]

        for idx, item in enumerate(detections):
            color = palette[idx % len(palette)]
            b = item["bbox"]
            # Draw rectangle border
            for offset in range(3):
                draw.rectangle([b[0] - offset, b[1] - offset, b[2] + offset, b[3] + offset], outline=color)

            # Draw tag label
            tag = f"#{idx+1} {item['label']} ({int(item['confidence']*100)}%)"
            # Label background banner
            draw.rectangle([b[0], max(0, b[1] - 18), b[0] + len(tag) * 7 + 8, b[1]], fill=color)
            draw.text((b[0] + 4, max(0, b[1] - 16)), tag, fill=(255, 255, 255))

        return annotated

    def search_look(
        self,
        scene_image_or_path: Union[str, Path, Image.Image],
        top_k: int = 5,
        tau: Optional[float] = None,
        detection_threshold: float = 0.25
    ) -> Dict:
        """
        Complete end-to-end Shop-the-Look discovery:
        1. Parse garments in scene
        2. Retrieve top-k catalog matches per garment
        3. Classify Exact Match vs. Recommendations
        4. Generate multi-attribute explanations
        """
        tau = tau if tau is not None else self.tau

        if isinstance(scene_image_or_path, (str, Path)):
            scene_image = Image.open(scene_image_or_path).convert("RGB")
            scene_source = str(scene_image_or_path)
        else:
            scene_image = scene_image_or_path.convert("RGB")
            scene_source = "Uploaded Image"

        # 1. Garment Detection
        detections = self.detect_garments(scene_image, threshold=detection_threshold)
        annotated = self.annotate_scene(scene_image, detections)

        # 2. Retrieval & Explainability per detected garment
        analyzed_items = []
        for idx, item in enumerate(detections):
            crop = item["crop"]
            crop_emb = self.embed_image(crop)

            # Fast Dot Product Search (1.8 ms across 28k products)
            dot_scores = np.dot(self.catalog_matrix, crop_emb.T).flatten()

            # Category-Constrained Filtering (prevents shoes matching jackets/tops and vice versa)
            if self.catalog_ars is not None:
                lbl = item["label"].lower()
                if any(k in lbl for k in ["shoe", "sneaker", "boot", "heels", "sandals", "footwear", "loafers", "flats"]):
                    # Low-profile shoes/sneakers in packshots have AR <= 1.25 (and exclude verified glasses)
                    mask = (self.catalog_ars <= 1.25)
                    if self.catalog_is_glasses is not None:
                        mask = mask & (~self.catalog_is_glasses)
                    dot_scores[~mask] = -1.0
                elif any(k in lbl for k in ["glasses", "sunglasses", "eyewear", "spectacles"]):
                    # Restrict exclusively to verified glasses/sunglasses packshots
                    if self.catalog_is_glasses is not None:
                        mask = self.catalog_is_glasses
                    else:
                        mask = (self.catalog_ars <= 0.80)
                    dot_scores[~mask] = -1.0
                elif any(k in lbl for k in ["pants", "jeans", "trousers", "leggings", "sweatpants", "dress", "skirt"]):
                    # Full-length bottoms and dresses are vertical (tall): AR >= 0.85
                    mask = (self.catalog_ars >= 0.85)
                    dot_scores[~mask] = -1.0
                elif any(k in lbl for k in ["top", "shirt", "jacket", "coat", "outerwear", "sweater", "hoodie", "vest", "blouse"]):
                    # Upper body apparel in packshots are vertical/square: AR >= 0.80
                    mask = (self.catalog_ars >= 0.80)
                    dot_scores[~mask] = -1.0

            top_indices = np.argpartition(-dot_scores, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-dot_scores[top_indices])]

            retrieved_sigs = [self.catalog_sigs[i] for i in top_indices]
            retrieved_scores = [float(dot_scores[i]) for i in top_indices]

            best_score = retrieved_scores[0]
            # Provisional match determination pending full calibration against positive/negative distributions
            is_provisional_match = best_score >= tau

            # Recommendations breakdown
            recs = []
            for rank, (sig, score) in enumerate(zip(retrieved_sigs, retrieved_scores), start=1):
                cat_path = self.products_dir / f"{sig}.jpg"
                cat_img = Image.open(cat_path).convert("RGB") if cat_path.exists() else None

                explanation = (
                    self.explainer.explain(
                        query_crop=crop,
                        catalog_image=cat_img,
                        cos_similarity=score,
                        detected_category=item["label"],
                        exact_match_threshold=tau
                    )
                    if cat_img else {}
                )

                recs.append({
                    "rank": rank,
                    "product_sig": sig,
                    "product_image_path": str(cat_path) if cat_path.exists() else None,
                    "similarity_score": round(score, 4),
                    "is_exact_match": (rank == 1 and is_provisional_match),
                    "is_provisional_match": (rank == 1 and is_provisional_match),
                    "explanation": explanation
                })

            analyzed_items.append({
                "item_id": idx + 1,
                "label": item["label"],
                "confidence": item["confidence"],
                "bbox": item["bbox"],
                "crop_image": crop,
                "exact_match_found": is_provisional_match,
                "is_provisional_match": is_provisional_match,
                "best_score": round(best_score, 4),
                "provisional_tau": tau,
                "tau_threshold": tau,
                "recommendations": recs
            })

        return {
            "source": scene_source,
            "scene_image": scene_image,
            "annotated_scene": annotated,
            "total_garments_detected": len(analyzed_items),
            "tau_threshold": tau,
            "items": analyzed_items
        }

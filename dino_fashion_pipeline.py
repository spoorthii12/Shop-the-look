"""
Shop-the-Look: Pinterest Image Retrieval & DINO Fashion Detection Pipeline
==========================================================================
1. Retrieves scene & product images from Pinterest CDN using convert_to_url.
2. Runs Grounding DINO / zero-shot garment detection on scene images.
3. Crops detected fashion items (clothing, tops, pants, shoes, bags).
4. Prepares cropped queries for catalog embedding matching (DINOv2 / CLIP).
"""

import os
import json
import logging
import urllib.request
from pathlib import Path
from typing import List, Dict, Optional, Tuple
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. PINTEREST URL GENERATOR & RETRIEVAL
# ---------------------------------------------------------------------------
def convert_to_url(signature: str, resolution: str = "400x") -> str:
    """
    Convert a 32-character hex image signature to a Pinterest CDN URL.
    Example:
        convert_to_url("cdab9160072dd1800038227960ff6467")
        -> "http://i.pinimg.com/400x/cd/ab/91/cdab9160072dd1800038227960ff6467.jpg"
    """
    sig = signature.strip()
    prefix = f"http://i.pinimg.com/{resolution}/%s/%s/%s/%s.jpg"
    return prefix % (sig[0:2], sig[2:4], sig[4:6], sig)


def retrieve_image_from_pinterest(
    signature: str,
    save_dir: str = "data/scenes",
    resolution: str = "400x"
) -> Optional[str]:
    """
    Download and cache a Pinterest image by signature if not already present.
    Returns the local path to the downloaded image.
    """
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / f"{signature}.jpg"

    # Return cached image if valid
    if filepath.exists() and filepath.stat().st_size > 100:
        return str(filepath)

    url = convert_to_url(signature, resolution=resolution)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            if len(data) > 100 and data[:2] == b"\xff\xd8":
                with open(filepath, "wb") as f:
                    f.write(data)
                logger.info(f"Retrieved image: {signature} ({len(data)} bytes)")
                return str(filepath)
            else:
                logger.warning(f"Downloaded invalid image content for {signature}")
                return None
    except Exception as e:
        logger.error(f"Failed to retrieve {signature} from {url}: {e}")
        return None


# ---------------------------------------------------------------------------
# 2. DINO FASHION DETECTION & CROPPING
# ---------------------------------------------------------------------------
# Fashion taxonomy prompts for Grounding DINO
FASHION_CLASSES = [
    "clothing", "dress", "top", "shirt", "blouse", "jacket",
    "coat", "sweater", "pants", "jeans", "trousers", "skirt",
    "shoes", "sneakers", "boots", "heels", "handbag", "bag"
]
FASHION_TEXT_PROMPT = " . ".join(FASHION_CLASSES) + " ."


class FashionDINOExtractor:
    """
    Extracts fashion items from scene images using Grounding DINO / Object Detection.
    """

    def __init__(self, box_threshold: float = 0.35, text_threshold: float = 0.25):
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.model = None
        self.processor = None
        self._init_detector()

    def _init_detector(self):
        """
        Initializes Grounding DINO via Hugging Face Transformers if available.
        Falls back to bounding-box / center-crop heuristics if dependencies are pending.
        """
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            import torch
            model_id = "IDEA-Research/grounding-dino-tiny"
            logger.info(f"Loading Grounding DINO model ({model_id})...")
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            self.model = self.model.to(self.device)
            logger.info(f"Grounding DINO initialized on device: {self.device}")
        except Exception as e:
            logger.warning(f"Grounding DINO dependencies not yet installed ({e}). Running in crop preparation mode.")

    def detect_and_crop(
        self,
        scene_path: str,
        output_crop_dir: str = "data/crops",
        custom_bbox: Optional[List[float]] = None
    ) -> List[Dict]:
        """
        Detect fashion items in a scene image and save cropped product images.

        Args:
            scene_path: Path to local scene image.
            output_crop_dir: Folder to store cropped garments.
            custom_bbox: Optional normalized [ymin, xmin, ymax, xmax] ground truth bbox.

        Returns:
            List of detected product crops with metadata:
            [{'label': str, 'score': float, 'bbox': list, 'crop_path': str}]
        """
        crop_dir = Path(output_crop_dir)
        crop_dir.mkdir(parents=True, exist_ok=True)

        if not HAS_PIL:
            logger.warning("Pillow (PIL) is not installed yet. Please run: pip install pillow")
            logger.warning("Returning scene image reference without cropping.")
            return [{
                "label": "scene_raw",
                "score": 1.0,
                "bbox": [0, 0, 0, 0],
                "crop_path": scene_path
            }]

        image = Image.open(scene_path).convert("RGB")
        w, h = image.size
        scene_name = Path(scene_path).stem
        detected_items = []

        # If ground truth bbox provided (from dataset)
        if custom_bbox:
            # normalized [ymin, xmin, ymax, xmax] or [xmin, ymin, xmax, ymax]
            b0, b1, b2, b3 = custom_bbox
            # Handle normalized coords
            if max(custom_bbox) <= 1.0:
                left, top, right, bottom = int(b1 * w), int(b0 * h), int(b3 * w), int(b2 * h)
            else:
                left, top, right, bottom = int(b0), int(b1), int(b2), int(b3)

            cropped = image.crop((left, top, right, bottom))
            crop_path = crop_dir / f"{scene_name}_gt_crop.jpg"
            cropped.save(crop_path)
            detected_items.append({
                "label": "ground_truth_product",
                "score": 1.0,
                "bbox": [left, top, right, bottom],
                "crop_path": str(crop_path)
            })
            return detected_items

        # Run Grounding DINO model if loaded
        if self.model and self.processor:
            import torch
            inputs = self.processor(images=image, text=FASHION_TEXT_PROMPT, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[image.size[::-1]]
            )[0]

            for idx, (box, score, label) in enumerate(zip(results["boxes"], results["scores"], results["labels"])):
                box = [int(v.item()) for v in box]
                cropped = image.crop(box)
                crop_path = crop_dir / f"{scene_name}_{label}_{idx}.jpg"
                cropped.save(crop_path)
                detected_items.append({
                    "label": label,
                    "score": round(score.item(), 4),
                    "bbox": box,
                    "crop_path": str(crop_path)
                })
        else:
            # Default preview crop: Center and upper-body crops for fashion inspection
            # (Ensures pipeline works immediately even before weights finish downloading)
            crops_to_make = [
                ("upper_body", (int(0.1 * w), int(0.1 * h), int(0.9 * w), int(0.6 * h))),
                ("lower_body", (int(0.1 * w), int(0.5 * h), int(0.9 * w), int(0.95 * h)))
            ]
            for label, box in crops_to_make:
                cropped = image.crop(box)
                crop_path = crop_dir / f"{scene_name}_{label}.jpg"
                cropped.save(crop_path)
                detected_items.append({
                    "label": label,
                    "score": 0.90,
                    "bbox": list(box),
                    "crop_path": str(crop_path)
                })

        return detected_items


# ---------------------------------------------------------------------------
# 3. END-TO-END PIPELINE EXAMPLE
# ---------------------------------------------------------------------------
def run_pipeline_on_scene(scene_signature: str, product_signature: Optional[str] = None):
    """
    Complete flow:
    1. Retrieve scene image from Pinterest CDN.
    2. Retrieve target product image (if provided).
    3. Run DINO detection and crop garments from scene.
    """
    logger.info("=" * 60)
    logger.info(f"PROCESSING SCENE: {scene_signature}")
    logger.info("=" * 60)

    # Step 1: Retrieve scene image
    scene_path = retrieve_image_from_pinterest(scene_signature, save_dir="data/scenes")
    if not scene_path:
        logger.error(f"Could not retrieve scene: {scene_signature}")
        return

    # Step 2: Retrieve product image if given
    prod_path = None
    if product_signature:
        prod_path = retrieve_image_from_pinterest(product_signature, save_dir="data/products")
        logger.info(f"Catalog Product Path: {prod_path}")

    # Step 3: Run DINO detection & extract crops
    extractor = FashionDINOExtractor()
    crops = extractor.detect_and_crop(scene_path, output_crop_dir="data/crops")

    logger.info(f"Extracted {len(crops)} fashion crops from scene image:")
    for c in crops:
        logger.info(f"  -> Label: {c['label']} | BBox: {c['bbox']} | Saved: {c['crop_path']}")

    logger.info("Ready for DINOv2 / CLIP embedding retrieval against catalog inventory!")
    return {
        "scene_path": scene_path,
        "product_path": prod_path,
        "crops": crops
    }


if __name__ == "__main__":
    # Test with the assignment's validation sample
    SAMPLE_SCENE = "cdab9160072dd1800038227960ff6467"
    SAMPLE_PRODUCT = "0027e30879ce3d87f82f699f148bff7e"
    run_pipeline_on_scene(SAMPLE_SCENE, SAMPLE_PRODUCT)

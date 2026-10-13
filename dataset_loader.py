"""
Shop-the-Look Dataset Loader & Utilities
=========================================
Module for parsing, URL generation, multi-threaded downloading,
and loading of Pinterest Shop-the-Look scene and catalog product images.
"""

import os
import json
import csv
import time
import socket
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Pinterest image URL prefix template
PINTEREST_URL_PREFIX = "http://i.pinimg.com/400x/%s/%s/%s/%s.jpg"


def convert_to_url(signature: str, resolution: str = "400x") -> str:
    """
    Convert a 32-character hexadecimal image signature into a Pinterest CDN URL.

    Args:
        signature: 32-character MD5 hex string (scene or product ID).
        resolution: CDN image width (e.g., '400x', '736x', or 'originals').

    Returns:
        Full HTTP URL pointing to the JPEG image on Pinterest CDN.
    """
    signature = signature.strip()
    if len(signature) < 6:
        raise ValueError(f"Invalid signature length ({len(signature)}): {signature}")

    prefix = f"http://i.pinimg.com/{resolution}/%s/%s/%s/%s.jpg"
    return prefix % (
        signature[0:2],
        signature[2:4],
        signature[4:6],
        signature
    )


def is_valid_jpeg(filepath: Union[str, Path]) -> bool:
    """
    Check if a file exists, is non-empty, and has valid JPEG magic bytes (0xFF, 0xD8).
    """
    path = Path(filepath)
    if not path.is_file() or path.stat().st_size < 100:
        return False
    try:
        with open(path, "rb") as f:
            header = f.read(2)
            return header == b"\xff\xd8"
    except Exception:
        return False


class ImageDownloader:
    """
    High-performance, multi-threaded image downloader with exponential
    backoff retry, verification, and resume capabilities.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        max_workers: int = 16,
        timeout: int = 15,
        max_retries: int = 3,
        user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    ):
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_retries = max_retries
        self.headers = {"User-Agent": user_agent}

    def download_single(self, signature: str, category: str = "images") -> Tuple[str, bool, Optional[str]]:
        """
        Download a single image by signature if not already present.

        Args:
            signature: 32-char hex image signature.
            category: Subfolder name (e.g. 'scenes', 'products').

        Returns:
            Tuple of (signature, success, local_filepath_or_error)
        """
        target_dir = self.output_dir / category
        target_dir.mkdir(parents=True, exist_ok=True)
        dest_path = target_dir / f"{signature}.jpg"

        # Check if valid file already exists
        if is_valid_jpeg(dest_path):
            return signature, True, str(dest_path)

        url = convert_to_url(signature)
        req = urllib.request.Request(url, headers=self.headers)

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    if resp.status == 200:
                        content = resp.read()
                        if len(content) > 100 and content[:2] == b"\xff\xd8":
                            with open(dest_path, "wb") as f:
                                f.write(content)
                            return signature, True, str(dest_path)
                        else:
                            last_error = "Downloaded content is not a valid JPEG"
                    else:
                        last_error = f"HTTP status {resp.status}"
            except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout, ConnectionResetError) as e:
                last_error = str(e)
                time.sleep(0.5 * (2 ** (attempt - 1)))  # Exponential backoff
            except Exception as e:
                last_error = str(e)
                break

        return signature, False, last_error

    def download_batch(
        self,
        signatures: List[str],
        category: str = "images",
        limit: Optional[int] = None
    ) -> Dict[str, str]:
        """
        Download a list of image signatures concurrently.

        Args:
            signatures: List of hex signatures.
            category: Target subfolder name ('scenes', 'products', etc.).
            limit: Maximum number of images to download.

        Returns:
            Dictionary mapping signature -> local file path for all successful downloads.
        """
        unique_sigs = list(dict.fromkeys(signatures))
        if limit:
            unique_sigs = unique_sigs[:limit]

        total = len(unique_sigs)
        logger.info(f"Starting download of {total} images into '{category}/' (workers={self.max_workers})...")

        results: Dict[str, str] = {}
        failed: List[Tuple[str, str]] = []
        completed_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_sig = {
                executor.submit(self.download_single, sig, category): sig
                for sig in unique_sigs
            }

            for future in as_completed(future_to_sig):
                sig, success, path_or_err = future.result()
                completed_count += 1

                if success:
                    results[sig] = path_or_err
                else:
                    failed.append((sig, str(path_or_err)))

                if completed_count % 50 == 0 or completed_count == total:
                    logger.info(
                        f"Progress: {completed_count}/{total} "
                        f"({(completed_count / total) * 100:.1f}%) | "
                        f"Success: {len(results)} | Failed: {len(failed)}"
                    )

        logger.info(f"Finished batch: {len(results)}/{total} downloaded successfully.")
        if failed:
            logger.warning(f"{len(failed)} images failed to download (e.g. 404 removed on Pinterest).")

        return results


class DatasetParser:
    """
    Parser to ingest and inspect raw assignment files:
    - JSON list: [{"scene": "..."}, ...]
    - JSON validation pairs: [{"product": "...", "scene": "..."}, ...]
    - JSON catalog items: [{"product": "...", ...}, ...]
    - JSON Lines (.jsonl) or CSV / TSV formats.
    """

    @staticmethod
    def load_records(filepath: Union[str, Path]) -> List[Dict]:
        """Load records from JSON, JSONL, or CSV file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        records = []
        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict):
                    # Handle dictionary wrappers, e.g. {"data": [...]} or {"scenes": [...]}
                    for key in ["data", "scenes", "products", "pairs", "records"]:
                        if key in data and isinstance(data[key], list):
                            return data[key]
                    records = [data]
        elif path.suffix.lower() in [".jsonl", ".txt"]:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        elif path.suffix.lower() in [".csv", ".tsv"]:
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                records = list(reader)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return records

    @staticmethod
    def extract_signatures(records: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        Extract scene signatures and product signatures from parsed records.

        Returns:
            Tuple of (scene_signatures, product_signatures)
        """
        scenes: Set[str] = set()
        products: Set[str] = set()

        for rec in records:
            if not isinstance(rec, dict):
                continue

            # Scene identifier keys
            for key in ["scene", "scene_id", "scene_signature", "image_signature"]:
                if key in rec and rec[key]:
                    scenes.add(str(rec[key]).strip())

            # Product identifier keys
            for key in ["product", "product_id", "product_signature"]:
                if key in rec and rec[key]:
                    products.add(str(rec[key]).strip())

        return sorted(list(scenes)), sorted(list(products))


class ShopTheLookDataset:
    """
    Unified dataset representation for evaluation and inference.
    Connects local downloaded image files with ground-truth scene-product pairs.
    """

    def __init__(self, data_dir: Union[str, Path], pairs: List[Dict]):
        self.data_dir = Path(data_dir)
        self.scenes_dir = self.data_dir / "scenes"
        self.products_dir = self.data_dir / "products"
        self.pairs = pairs

        # Filter only pairs where both or scene images are available locally
        self.valid_pairs = []
        for p in self.pairs:
            scene_sig = p.get("scene")
            prod_sig = p.get("product")
            scene_path = self.scenes_dir / f"{scene_sig}.jpg" if scene_sig else None
            prod_path = self.products_dir / f"{prod_sig}.jpg" if prod_sig else None

            entry = dict(p)
            entry["scene_path"] = str(scene_path) if scene_path and is_valid_jpeg(scene_path) else None
            entry["product_path"] = str(prod_path) if prod_path and is_valid_jpeg(prod_path) else None
            self.valid_pairs.append(entry)

    def __len__(self) -> int:
        return len(self.valid_pairs)

    def __getitem__(self, idx: int) -> Dict:
        return self.valid_pairs[idx]

    def summary(self) -> Dict[str, int]:
        total = len(self.valid_pairs)
        scenes_present = sum(1 for p in self.valid_pairs if p.get("scene_path"))
        prods_present = sum(1 for p in self.valid_pairs if p.get("product_path"))
        complete_pairs = sum(1 for p in self.valid_pairs if p.get("scene_path") and p.get("product_path"))
        return {
            "total_pairs": total,
            "scenes_cached": scenes_present,
            "products_cached": prods_present,
            "ready_for_eval": complete_pairs
        }

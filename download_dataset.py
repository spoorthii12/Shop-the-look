#!/usr/bin/env python3
"""
CLI Script to Download and Extract Shop-the-Look Dataset
======================================================
Usage:
    # 1. Download from input file (JSON / JSONL / CSV):
    python3 download_dataset.py --input data/validation.json --output-dir data/

    # 2. Download specific scene or product signatures directly:
    python3 download_dataset.py --scene cdab9160072dd1800038227960ff6467 --product 0027e30879ce3d87f82f699f148bff7e

    # 3. Download a test subset (e.g. first 20):
    python3 download_dataset.py --input data/catalog.json --limit 20
"""

import sys
import argparse
from pathlib import Path
from dataset_loader import (
    convert_to_url,
    ImageDownloader,
    DatasetParser,
    ShopTheLookDataset,
    logger
)


def main():
    parser = argparse.ArgumentParser(
        description="Download and verify Pinterest Shop-the-Look scene and product catalog images."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Path to JSON, JSONL, or CSV file containing records with 'scene' or 'product' fields."
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data",
        help="Directory to store downloaded images and manifests (default: 'data')."
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=16,
        help="Number of concurrent worker threads for downloading (default: 16)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to download (useful for quick testing / development)."
    )
    parser.add_argument(
        "--scene",
        type=str,
        help="Single scene signature to download directly."
    )
    parser.add_argument(
        "--product",
        type=str,
        help="Single product signature to download directly."
    )
    parser.add_argument(
        "--scenes-only",
        action="store_true",
        help="Only download scene images."
    )
    parser.add_argument(
        "--products-only",
        action="store_true",
        help="Only download product catalog images."
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    downloader = ImageDownloader(output_dir=output_dir, max_workers=args.workers)

    # Direct signature mode
    if args.scene or args.product:
        if args.scene:
            sig = args.scene.strip()
            logger.info(f"Downloading scene: {sig}")
            logger.info(f"URL: {convert_to_url(sig)}")
            _, success, path = downloader.download_single(sig, category="scenes")
            if success:
                logger.info(f"Successfully saved to: {path}")
            else:
                logger.error(f"Download failed: {path}")

        if args.product:
            sig = args.product.strip()
            logger.info(f"Downloading product: {sig}")
            logger.info(f"URL: {convert_to_url(sig)}")
            _, success, path = downloader.download_single(sig, category="products")
            if success:
                logger.info(f"Successfully saved to: {path}")
            else:
                logger.error(f"Download failed: {path}")
        return

    # File input mode
    if not args.input:
        parser.print_help()
        logger.error("\nError: Please provide either --input <file> or --scene / --product.")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    logger.info(f"Reading dataset from: {input_path}")
    records = DatasetParser.load_records(input_path)
    logger.info(f"Parsed {len(records)} records from {input_path.name}")

    scenes, products = DatasetParser.extract_signatures(records)
    logger.info(f"Found {len(scenes)} unique scene signatures and {len(products)} unique product signatures.")

    # Process scenes
    if not args.products_only and scenes:
        downloader.download_batch(
            signatures=scenes,
            category="scenes",
            limit=args.limit
        )

    # Process products
    if not args.scenes_only and products:
        downloader.download_batch(
            signatures=products,
            category="products",
            limit=args.limit
        )

    # Show dataset status summary
    ds = ShopTheLookDataset(data_dir=output_dir, pairs=records)
    summary = ds.summary()
    logger.info("=" * 50)
    logger.info("Dataset Extraction Summary:")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()

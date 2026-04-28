"""Command-line entry for pose auto-annotation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pose_annotator import __version__
from pose_annotator.auto_annotate import infer_labels_dir, iter_images, run_auto_annotate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pose-annotator",
        description=(
            "Auto-annotate images with Ultralytics YOLO pose models. "
            "Writes one .txt per image in YOLO pose format (see Ultralytics dataset docs)."
        ),
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    p.add_argument(
        "images",
        type=Path,
        help="Directory containing images (use with --recursive for nested folders).",
    )
    p.add_argument(
        "-o",
        "--labels-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for label .txt files; mirrors folder structure under IMAGES. "
            "Default: sibling 'labels/' if IMAGES is named 'images', else '<parent>/<name>_labels'."
        ),
    )
    p.add_argument(
        "--model",
        type=str,
        default="yolo26x-pose.pt",
        help="Ultralytics pose weights (default: yolo26x-pose.pt; downloads on first use).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device, e.g. 0, cpu, cuda:0 (default: Ultralytics auto).",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=448,
        help="Inference image size (default: 448).",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.25,
        metavar="THR",
        help="Box confidence threshold (default: 0.25).",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold (default: 0.5).",
    )
    p.add_argument(
        "--kpt-conf",
        type=float,
        default=0.25,
        metavar="THR",
        help="Keypoint confidence; below this, visibility set to 0 (default: 0.25).",
    )
    p.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Maximum detections per image (default: 300).",
    )
    p.add_argument(
        "--predict-batch",
        type=int,
        default=16,
        metavar="N",
        help="Number of images per model.predict call (default: 16). Lower if GPU OOM.",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Include images in subfolders of IMAGES.",
    )
    p.add_argument(
        "--decimals",
        type=int,
        default=6,
        help="Decimal places for normalized floats in labels (default: 6).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan images only; do not load model or write files.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (-v INFO, -vv DEBUG).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    level = logging.WARNING
    if args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose == 1:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(message)s",
    )

    labels_dir = args.labels_dir
    if args.dry_run:
        ld = labels_dir or infer_labels_dir(args.images)
        n = sum(1 for _ in iter_images(args.images.resolve(), args.recursive))
        logging.info("Dry run: would process %d images; labels root: %s", n, ld)
        return 0

    processed, written = run_auto_annotate(
        images_dir=args.images,
        labels_dir=labels_dir,
        model_name=args.model,
        device=args.device,
        imgsz=args.imgsz,
        box_conf=args.conf,
        iou=args.iou,
        kpt_conf=args.kpt_conf,
        recursive=args.recursive,
        dry_run=False,
        coord_decimals=args.decimals,
        max_det=args.max_det,
        predict_batch=args.predict_batch,
        keypoint_slot_map=None,
        output_kpt_count=None,
    )
    out = labels_dir or infer_labels_dir(args.images)
    logging.info("Done. Processed %d images, wrote %d label files under %s", processed, written, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

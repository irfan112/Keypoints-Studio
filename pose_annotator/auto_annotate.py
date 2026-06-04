"""Batch auto-annotation with a YOLO pose model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from tqdm import tqdm

from pose_annotator.formats import PoseLabelLine, visibility_from_confidence

LOGGER = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".mpo"}


def iter_images(root: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                yield p
    else:
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                yield p


def infer_labels_dir(images_dir: Path) -> Path:
    """Prefer sibling `labels/` when folder is named `images`."""
    name = images_dir.name.lower()
    if name == "images":
        return images_dir.parent / "labels"
    return images_dir.parent / f"{images_dir.name}_labels"


def _chunks(items: Sequence[Path], size: int) -> Iterable[list[Path]]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def relative_under(root: Path, path: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)


def build_label_path(
    image_path: Path,
    images_root: Path,
    labels_root: Path,
) -> Path:
    rel = relative_under(images_root, image_path)
    return labels_root / rel.with_suffix(".txt")


def label_path_next_to_image(image_path: Path) -> Path:
    """YOLO-style label file in the same folder as the image, matching stem."""
    return image_path.with_suffix(".txt")


def results_to_lines(
    result,
    box_conf_threshold: float,
    kpt_conf_threshold: float,
    coord_decimals: int,
    keypoint_slot_map: Mapping[int, int] | None = None,
    output_kpt_count: int | None = None,
) -> list[str]:
    """Convert one Results object to YOLO pose label lines (normalized)."""
    if result.boxes is None or len(result.boxes) == 0:
        return []

    im_h, im_w = result.orig_shape
    lines: list[str] = []

    boxes = result.boxes
    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()

    kpts_obj = result.keypoints
    if kpts_obj is None:
        LOGGER.warning("Result has boxes but no keypoints; skipping image.")
        return []

    kp_xy = kpts_obj.xy
    if kp_xy is None or kp_xy.numel() == 0:
        return []

    kp_xy_np = kp_xy.cpu().numpy()
    k_conf_np = (
        kpts_obj.conf.cpu().numpy()
        if getattr(kpts_obj, "conf", None) is not None
        else np.ones(kp_xy_np.shape[:2], dtype=np.float64)
    )

    for i in range(len(boxes)):
        if conf[i] < box_conf_threshold:
            continue

        n_kp = kp_xy_np.shape[1]
        xyv = np.zeros((n_kp, 3), dtype=np.float64)
        xyv[:, 0] = kp_xy_np[i, :, 0]
        xyv[:, 1] = kp_xy_np[i, :, 1]
        for j in range(n_kp):
            xyv[j, 2] = visibility_from_confidence(
                k_conf_np[i, j], kpt_conf_threshold, occluded_as_one=False
            )

        if keypoint_slot_map:
            out_k = int(output_kpt_count or max(17, max(keypoint_slot_map.keys())))
            remapped = np.zeros((out_k, 3), dtype=np.float64)
            for new_slot_1b, orig_slot_1b in keypoint_slot_map.items():
                ni = int(new_slot_1b) - 1
                oi = int(orig_slot_1b) - 1
                if 0 <= ni < out_k and 0 <= oi < xyv.shape[0]:
                    remapped[ni] = xyv[oi]
            xyv_out = remapped
        else:
            xyv_out = xyv

        pl = PoseLabelLine(
            class_id=int(cls[i]),
            xyxy=xyxy[i].astype(np.float64),
            keypoints_xyv=xyv_out,
        )
        lines.append(pl.to_normalized_line(im_w, im_h, coord_decimals=coord_decimals))

    return lines


def run_auto_annotate(
    images_dir: Path,
    labels_dir: Path | None,
    model_name: str,
    device: str | int | None,
    imgsz: int,
    box_conf: float,
    iou: float,
    kpt_conf: float,
    recursive: bool,
    dry_run: bool,
    coord_decimals: int,
    max_det: int,
    predict_batch: int,
    keypoint_slot_map: Mapping[int, int] | None = None,
    output_kpt_count: int | None = None,
    labels_same_folder_as_images: bool = False,
) -> tuple[int, int]:
    """
    Run pose prediction and write one .txt per image.
    Returns (images_processed, label_files_written).

    If ``labels_same_folder_as_images`` is True, each ``.txt`` is written next to its image;
    ``labels_dir`` is ignored for output paths (CLI/GUI backward compatibility uses the flag).
    """
    from ultralytics import YOLO
    
    images_dir = images_dir.resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    out_root = (
        images_dir
        if labels_same_folder_as_images
        else (labels_dir or infer_labels_dir(images_dir)).resolve()
    )

    image_paths = list(iter_images(images_dir, recursive))
    if not image_paths:
        LOGGER.warning("No images found under %s", images_dir)
        return 0, 0

    LOGGER.info("Loading model %s", model_name)
    model = YOLO(model_name)
    if labels_same_folder_as_images:
        LOGGER.info("Label output: alongside each image under %s", images_dir)
    else:
        LOGGER.info("Label output root: %s", out_root)

    written = 0
    processed = 0

    pb = max(1, predict_batch)

    if dry_run:
        for _ in tqdm(image_paths, desc="Scan", unit="img"):
            processed += 1
            written += 1
        return processed, written

    for chunk in tqdm(
        list(_chunks(image_paths, pb)),
        desc="Annotating",
        unit="batch",
    ):
        batch_results = model.predict(
            source=[str(p) for p in chunk],
            imgsz=imgsz,
            conf=box_conf,
            iou=iou,
            device=device,
            max_det=max_det,
            verbose=False,
        )

        if len(batch_results) != len(chunk):
            LOGGER.warning(
                "Predict batch length mismatch (%d vs %d); missing results get empty labels.",
                len(batch_results),
                len(chunk),
            )

        for i, img_path in enumerate(chunk):
            processed += 1
            label_path = (
                label_path_next_to_image(img_path)
                if labels_same_folder_as_images
                else build_label_path(img_path, images_dir, out_root)
            )
            if i >= len(batch_results):
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text("", encoding="utf-8")
                written += 1
                continue

            result = batch_results[i]
            lines = results_to_lines(
                result,
                box_conf_threshold=box_conf,
                kpt_conf_threshold=kpt_conf,
                coord_decimals=coord_decimals,
                keypoint_slot_map=keypoint_slot_map,
                output_kpt_count=output_kpt_count,
            )
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(
                ("\n".join(lines) + ("\n" if lines else "")),
                encoding="utf-8",
            )
            written += 1

    return processed, written

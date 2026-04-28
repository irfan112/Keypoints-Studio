"""Serialize detections to Ultralytics YOLO pose label lines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PoseLabelLine:
    """One row: class, xywh (normalized), then kpt triplets (x, y, visibility)."""

    class_id: int
    xyxy: np.ndarray  # shape (4,) pixel coords xyxy
    keypoints_xyv: np.ndarray  # shape (K, 3) pixel x, y, visibility 0/1/2

    def to_normalized_line(
        self,
        img_w: int,
        img_h: int,
        coord_decimals: int = 6,
    ) -> str:
        x1, y1, x2, y2 = map(float, self.xyxy.tolist())
        xc = ((x1 + x2) * 0.5) / img_w
        yc = ((y1 + y2) * 0.5) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h

        fmt = f"{{:.{coord_decimals}f}}"
        parts: list[str] = [
            str(int(self.class_id)),
            fmt.format(xc),
            fmt.format(yc),
            fmt.format(bw),
            fmt.format(bh),
        ]

        k = self.keypoints_xyv
        for i in range(k.shape[0]):
            px, py = float(k[i, 0]) / img_w, float(k[i, 1]) / img_h
            v = int(round(float(k[i, 2])))
            parts.extend([fmt.format(px), fmt.format(py), str(v)])
        return " ".join(parts)


def visibility_from_confidence(
    conf: float | np.floating,
    visible_threshold: float,
    occluded_as_one: bool = False,
) -> int:
    """
    Map keypoint confidence to COCO-style visibility stored in YOLO kpt labels.
    0 = not labeled / missing, 1 = occluded (optional), 2 = visible.
    """
    c = float(conf)
    if c < visible_threshold:
        return 0
    if occluded_as_one and c < visible_threshold + 0.25:
        return 1
    return 2

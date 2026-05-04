"""Desktop GUI for pose auto-annotation and preview."""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import (
    QColor,
    QBrush,
    QFont,
    QImage,
    QKeySequence,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSplitter,
    QSpinBox,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsPixmapItem,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
    QSizePolicy,
    QStyle,
    QMenu,
    QScrollArea,
)
from ultralytics import YOLO

from pose_annotator.auto_annotate import iter_images, run_auto_annotate
from pose_annotator.formats import PoseLabelLine


def _to_qpixmap(rgb: np.ndarray) -> QPixmap:
    """Convert HxWx3 RGB uint8 numpy array to QPixmap."""
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8, copy=False)
    h, w, _ = rgb.shape
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    # Copy to detach from numpy buffer lifetime.
    return QPixmap.fromImage(qimg.copy())


def _slot_color(slot: int) -> QColor:
    """Deterministic distinct color per keypoint slot."""
    # Golden-angle HSV palette
    hue = (int(slot) * 137) % 360
    c = QColor.fromHsv(hue, 220, 240)
    c.setAlpha(140)  # transparent fill
    return c


class ImageViewer(QGraphicsView):
    """Zoomable/pannable image viewer (mouse wheel zoom, drag to pan)."""

    _ROLE_CROSSHAIR = int(Qt.ItemDataRole.UserRole) + 31

    def __init__(self) -> None:
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pix_item)

        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setStyleSheet("QGraphicsView { background: #111; }")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        self._has_image = False
        self._click_handler = None

        # Bbox guide mode (W): crosshair + drag rectangle
        self._crosshair_mode = False
        self._line_h: Optional[QGraphicsLineItem] = None
        self._line_v: Optional[QGraphicsLineItem] = None
        self._rubber: Optional[QGraphicsRectItem] = None
        self._drag_start: Optional[QPointF] = None
        self._bbox_complete_cb = None
        self._middle_panning = False
        self._middle_last_pos = QPointF()

    def set_placeholder(self, text: str) -> None:
        self._scene.clear()
        self._pix_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pix_item)
        self._scene.addText(text).setDefaultTextColor(Qt.GlobalColor.lightGray)
        self.resetTransform()
        self._has_image = False
        self._click_handler = None

    def set_base_pixmap(self, pix: QPixmap, fit: bool = True) -> None:
        """Set the background image and clear overlays."""
        self._scene.clear()
        self._pix_item = QGraphicsPixmapItem(pix)
        self._pix_item.setZValue(0)
        self._scene.addItem(self._pix_item)
        self._scene.setSceneRect(pix.rect())
        self._has_image = True
        self.resetTransform()
        if fit:
            self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)

    def set_click_handler(self, handler) -> None:
        """If set, left-clicks will call handler(scene_x, scene_y) and be consumed."""
        self._click_handler = handler

    def add_overlay(self, item) -> None:
        """Add an overlay item above the base image."""
        item.setZValue(10)
        self._scene.addItem(item)

    def clear_overlays(self) -> None:
        """Remove all items except the base pixmap."""
        for it in list(self._scene.items()):
            if it is self._pix_item:
                continue
            if it.data(self._ROLE_CROSSHAIR):
                continue
            self._scene.removeItem(it)

    def fit_to_image(self) -> None:
        if self._has_image:
            self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)

    def set_crosshair_bbox_mode(
        self,
        enabled: bool,
        on_complete_bbox=None,
    ) -> None:
        """Enable/disable crosshair + drag-to-draw bbox (scene coordinates)."""
        self._crosshair_mode = bool(enabled)
        self._bbox_complete_cb = on_complete_bbox
        self._drag_start = None
        if self._rubber is not None:
            try:
                self._scene.removeItem(self._rubber)
            except Exception:
                pass
            self._rubber = None

        if not enabled:
            self._bbox_complete_cb = None
            self._remove_crosshair_graphics()
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self._middle_panning = False
            return

        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._ensure_crosshair_lines()
        self._update_crosshair_at_scene(self.mapToScene(self.viewport().rect().center()))

    def _remove_crosshair_graphics(self) -> None:
        for it in (self._line_h, self._line_v):
            if it is None:
                continue
            try:
                self._scene.removeItem(it)
            except Exception:
                pass
        self._line_h = None
        self._line_v = None

    def _ensure_crosshair_lines(self) -> None:
        if self._line_h is None:
            self._line_h = QGraphicsLineItem()
            self._line_h.setData(self._ROLE_CROSSHAIR, True)
            self._line_h.setZValue(99)
            self._line_h.setPen(QPen(QColor(0, 255, 200, 220), 1))
            self._scene.addItem(self._line_h)
        if self._line_v is None:
            self._line_v = QGraphicsLineItem()
            self._line_v.setData(self._ROLE_CROSSHAIR, True)
            self._line_v.setZValue(99)
            self._line_v.setPen(QPen(QColor(0, 255, 200, 220), 1))
            self._scene.addItem(self._line_v)

    def _clamp_scene_pos(self, p: QPointF) -> QPointF:
        r = self._scene.sceneRect()
        return QPointF(
            float(min(max(p.x(), r.left()), r.right())),
            float(min(max(p.y(), r.top()), r.bottom())),
        )

    def _update_crosshair_at_scene(self, p: QPointF) -> None:
        if not self._crosshair_mode or not self._has_image:
            return
        self._ensure_crosshair_lines()
        sr = self._scene.sceneRect()
        x, y = p.x(), p.y()
        self._line_h.setLine(sr.left(), y, sr.right(), y)
        self._line_v.setLine(x, sr.top(), x, sr.bottom())

    def _update_rubber(self, p1: QPointF, p2: QPointF) -> None:
        x1 = min(p1.x(), p2.x())
        y1 = min(p1.y(), p2.y())
        x2 = max(p1.x(), p2.x())
        y2 = max(p1.y(), p2.y())
        rect = QRectF(x1, y1, x2 - x1, y2 - y1)
        if self._rubber is None:
            self._rubber = QGraphicsRectItem(rect)
            self._rubber.setData(self._ROLE_CROSSHAIR, True)
            self._rubber.setZValue(98)
            self._rubber.setPen(QPen(QColor(255, 220, 60, 240), 1, Qt.PenStyle.DashLine))
            self._rubber.setBrush(QBrush(QColor(255, 220, 60, 40)))
            self._scene.addItem(self._rubber)
        else:
            self._rubber.setRect(rect)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if not self._has_image:
            return super().wheelEvent(event)
        # Typical mouse wheel delta is 120 per notch.
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 0.8
        self.scale(factor, factor)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._middle_panning and self._has_image:
            delta = QPointF(event.pos()) - self._middle_last_pos
            self._middle_last_pos = QPointF(event.pos())
            self.horizontalScrollBar().setValue(
                int(self.horizontalScrollBar().value() - delta.x())
            )
            self.verticalScrollBar().setValue(
                int(self.verticalScrollBar().value() - delta.y())
            )
            event.accept()
            return

        if self._crosshair_mode and self._has_image:
            sp = self._clamp_scene_pos(self.mapToScene(event.pos()))
            self._update_crosshair_at_scene(sp)
            if self._drag_start is not None:
                self._update_rubber(self._drag_start, sp)
            event.accept()
            return

        return super().mouseMoveEvent(event)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if self._crosshair_mode and self._has_image:
            if event.button() == Qt.MouseButton.MiddleButton:
                self._middle_panning = True
                self._middle_last_pos = QPointF(event.pos())
                event.accept()
                return
            if event.button() == Qt.MouseButton.LeftButton:
                sp = self._clamp_scene_pos(self.mapToScene(event.pos()))
                self._drag_start = QPointF(sp)
                self._update_rubber(sp, sp)
                event.accept()
                return

        if (
            self._has_image
            and self._click_handler is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            p = self.mapToScene(event.pos())
            self._click_handler(float(p.x()), float(p.y()))
            event.accept()
            return
        return super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.MiddleButton:
            self._middle_panning = False
            event.accept()
            return super().mouseReleaseEvent(event)

        if self._crosshair_mode and self._has_image and event.button() == Qt.MouseButton.LeftButton:
            if self._drag_start is not None:
                sp = self._clamp_scene_pos(self.mapToScene(event.pos()))
                x1 = min(self._drag_start.x(), sp.x())
                y1 = min(self._drag_start.y(), sp.y())
                x2 = max(self._drag_start.x(), sp.x())
                y2 = max(self._drag_start.y(), sp.y())
                self._drag_start = None
                if self._rubber is not None:
                    try:
                        self._scene.removeItem(self._rubber)
                    except Exception:
                        pass
                    self._rubber = None
                if (x2 - x1) >= 3.0 and (y2 - y1) >= 3.0 and self._bbox_complete_cb:
                    self._bbox_complete_cb(float(x1), float(y1), float(x2), float(y2))
            event.accept()
            return

        return super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        # Let parent handle deletes on selected keypoints.
        if event.key() in (Qt.Key.Key_A,):
            mw = self.window()
            if hasattr(mw, "prev_image"):
                mw.prev_image()  # type: ignore[attr-defined]
                event.accept()
                return
        if event.key() in (Qt.Key.Key_D,):
            mw = self.window()
            if hasattr(mw, "next_image"):
                mw.next_image()  # type: ignore[attr-defined]
                event.accept()
                return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            mw = self.window()
            # Prefer deleting bbox if a bbox/handle is selected, otherwise delete keypoints.
            if hasattr(mw, "delete_selected_bboxes"):
                try:
                    if mw.delete_selected_bboxes():  # type: ignore[attr-defined]
                        event.accept()
                        return
                except Exception:
                    pass
            if hasattr(mw, "delete_selected_keypoints"):
                mw.delete_selected_keypoints()  # type: ignore[attr-defined]
                event.accept()
                return
        return super().keyPressEvent(event)


class DraggableKeypoint(QGraphicsEllipseItem):
    """A draggable keypoint marker that updates a callback on move."""

    def __init__(
        self,
        center_xy: tuple[float, float],
        radius: float,
        label: str,
        on_moved,
        color: QColor,
        person_idx: int,
        out_slot: int,
    ) -> None:
        cx, cy = center_xy
        super().__init__(QRectF(cx - radius, cy - radius, radius * 2, radius * 2))
        self.setBrush(QBrush(color))
        # Slightly darker outline for visibility
        outline = QColor(color)
        outline.setAlpha(255)
        self.setPen(QPen(outline.darker(180), 1))
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(
            QGraphicsEllipseItem.GraphicsItemFlag.ItemSendsGeometryChanges, True
        )
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        self._radius = radius
        self._on_moved = on_moved
        self.person_idx = int(person_idx)
        self.out_slot = int(out_slot)  # 1-based output slot number

        self._text = QGraphicsSimpleTextItem(label)
        self._text.setParentItem(self)
        f = QFont()
        f.setBold(True)
        # Fit text inside circle: point size tied to radius.
        f.setPointSize(max(6, int(round(radius * 2.0))))
        self._text.setFont(f)
        self._text.setBrush(QBrush(QColor(255, 255, 255, 255)))  # white
        self._text.setZValue(11)
        self._center_text()

    def text_item(self) -> QGraphicsSimpleTextItem:
        return self._text

    def _center_text(self) -> None:
        # Center text in the ellipse rect (local coords).
        r = self.rect()
        tb = self._text.boundingRect()
        x = r.center().x() - tb.width() / 2.0
        y = r.center().y() - tb.height() / 2.0
        self._text.setPos(QPointF(x, y))

    def itemChange(self, change, value):  # type: ignore[override]
        if change == QGraphicsEllipseItem.GraphicsItemChange.ItemPositionHasChanged:
            # Keep text next to marker and notify.
            r = self._radius
            rect = self.rect()
            pos = self.pos()
            cx = pos.x() + rect.x() + r
            cy = pos.y() + rect.y() + r
            self._center_text()
            self._on_moved(float(cx), float(cy))
        return super().itemChange(change, value)

    def contextMenuEvent(self, event) -> None:  # type: ignore[override]
        menu = QMenu()
        act = menu.addAction("Delete keypoint")
        chosen = menu.exec(event.screenPos())
        if chosen == act:
            mw = self.scene().views()[0].window() if self.scene() and self.scene().views() else None
            if mw is not None and hasattr(mw, "delete_keypoint"):
                mw.delete_keypoint(self.person_idx, self.out_slot)  # type: ignore[attr-defined]
            else:
                # Fallback: remove only visuals
                try:
                    self.scene().removeItem(self._text)
                except Exception:
                    pass
                try:
                    self.scene().removeItem(self)
                except Exception:
                    pass
            event.accept()
            return
        return super().contextMenuEvent(event)


class _BBoxHandle(QGraphicsRectItem):
    def __init__(self, size: float, on_dragged) -> None:
        super().__init__(QRectF(-size / 2, -size / 2, size, size))
        self.setBrush(QBrush(QColor(255, 255, 255, 200)))
        self.setPen(QPen(QColor(0, 0, 0, 200), 1))
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self._on_dragged = on_dragged

    def itemChange(self, change, value):  # type: ignore[override]
        if change == QGraphicsRectItem.GraphicsItemChange.ItemPositionHasChanged:
            p: QPointF = self.pos()
            self._on_dragged(float(p.x()), float(p.y()))
        return super().itemChange(change, value)


class ResizableBBox(QGraphicsRectItem):
    """Resizable bbox with 4 corner handles."""

    def __init__(
        self,
        rect: QRectF,
        on_changed,
        on_deleted,
        person_idx: int,
        handle_size: float = 8.0,
    ) -> None:
        super().__init__(rect)
        self.setPen(QPen(QColor(255, 255, 0), 1))
        self.setBrush(Qt.BrushStyle.NoBrush)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self._on_changed = on_changed
        self._on_deleted = on_deleted
        self.person_idx = int(person_idx)
        self._min_size = 4.0

        self._tl = _BBoxHandle(handle_size, lambda x, y: self._set_corner("tl", x, y))
        self._tr = _BBoxHandle(handle_size, lambda x, y: self._set_corner("tr", x, y))
        self._bl = _BBoxHandle(handle_size, lambda x, y: self._set_corner("bl", x, y))
        self._br = _BBoxHandle(handle_size, lambda x, y: self._set_corner("br", x, y))
        for h in (self._tl, self._tr, self._bl, self._br):
            h.setParentItem(self)
            h.setZValue(12)
        self._sync_handles()

    def contextMenuEvent(self, event) -> None:  # type: ignore[override]
        menu = QMenu()
        act = menu.addAction("Delete bbox")
        chosen = menu.exec(event.screenPos())
        if chosen == act:
            self._on_deleted()
            event.accept()
            return
        return super().contextMenuEvent(event)

    def _sync_handles(self) -> None:
        r = self.rect()
        self._tl.setPos(r.left(), r.top())
        self._tr.setPos(r.right(), r.top())
        self._bl.setPos(r.left(), r.bottom())
        self._br.setPos(r.right(), r.bottom())

    def _set_corner(self, which: str, x: float, y: float) -> None:
        r = self.rect()
        left, top, right, bottom = r.left(), r.top(), r.right(), r.bottom()
        if which == "tl":
            left, top = x, y
        elif which == "tr":
            right, top = x, y
        elif which == "bl":
            left, bottom = x, y
        elif which == "br":
            right, bottom = x, y

        # Normalize
        nleft, nright = (left, right) if left <= right else (right, left)
        ntop, nbottom = (top, bottom) if top <= bottom else (bottom, top)
        if (nright - nleft) < self._min_size:
            nright = nleft + self._min_size
        if (nbottom - ntop) < self._min_size:
            nbottom = ntop + self._min_size

        self.setRect(QRectF(QPointF(nleft, ntop), QPointF(nright, nbottom)))
        self._sync_handles()
        self._on_changed(float(nleft), float(ntop), float(nright), float(nbottom))


DEFAULT_SLOT_MAP = {
    # new_slot: original_yolo_slot (1-based)
    8: 1,  # Nose
    6: 6,  # Left Shoulder
    5: 7,  # Right Shoulder
    4: 8,  # Left Elbow
    3: 9,  # Right Elbow
    2: 10,  # Left Wrist
    1: 11,  # Right Wrist
    11: 12,  # Left Hip
    10: 13,  # Right Hip
    14: 14,  # Left Knee
    13: 15,  # Right Knee
    16: 16,  # Left Ankle
    15: 17,  # Right Ankle
}

YOLO_KPT_NAMES = [
    "Nose",
    "Left Eye",
    "Right Eye",
    "Left Ear",
    "Right Ear",
    "Left Shoulder",
    "Right Shoulder",
    "Left Elbow",
    "Right Elbow",
    "Left Wrist",
    "Right Wrist",
    "Left Hip",
    "Right Hip",
    "Left Knee",
    "Right Knee",
    "Left Ankle",
    "Right Ankle",
]


@dataclass
class KeypointDef:
    slot: int  # output slot number (1-based)
    name: str  # human name
    source: str  # "yolo" or "custom"
    yolo_slot: Optional[int]  # if source=="yolo"


def _default_keypoint_defs() -> dict[int, KeypointDef]:
    # Default: your mapping as output slots (with names) + ability to add custom later.
    out: dict[int, KeypointDef] = {}
    for out_slot, yolo_slot in DEFAULT_SLOT_MAP.items():
        nm = (
            YOLO_KPT_NAMES[yolo_slot - 1]
            if 1 <= yolo_slot <= len(YOLO_KPT_NAMES)
            else f"yolo_{yolo_slot}"
        )
        out[out_slot] = KeypointDef(
            slot=out_slot, name=nm, source="yolo", yolo_slot=yolo_slot
        )
    return out


def _defs_to_text(defs: dict[int, KeypointDef]) -> str:
    # HTML so we can color only "custom" markers red.
    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    lines: list[str] = ["<b>Keypoints (output slot):</b>"]
    for s in sorted(defs.keys()):
        d = defs[s]
        if d.source == "custom":
            lines.append(
                f"{int(d.slot)}. {esc(d.name)} "
                f"<span style=\"color:#c62828; font-weight:600;\">(custom)</span>"
            )
        else:
            # YOLO-mapped: show only the keypoint name (no 'YOLO #' suffix)
            lines.append(f"{int(d.slot)}. {esc(d.name)}")
    lines.append("")
    lines.append("<span style=\"color:#666;\">Any slot not listed here is discarded (written as 0 0 0).</span>")
    return "<br/>".join(lines)


class MappingDialog(QDialog):
    """Keypoint schema editor: map YOLO keypoints to output slots + add/remove custom keypoints."""

    def __init__(self, parent: QWidget, defs: dict[int, KeypointDef]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Keypoint Mapping")
        self._result_defs: dict[int, KeypointDef] = {
            k: KeypointDef(**vars(v)) for k, v in defs.items()
        }

        root = QVBoxLayout(self)

        info = QLabel(
            "YOLO keypoints are fixed (1..17). Assign an output slot or uncheck Include to remove.\n"
            "Custom keypoints can use any output slot number (e.g. 18). Duplicated output slots: last wins."
        )
        info.setWordWrap(True)
        root.addWidget(info)

        # --- YOLO mapping table
        yolo_table = QTableWidget()
        yolo_table.setColumnCount(5)
        yolo_table.setHorizontalHeaderLabels(
            ["YOLO #", "YOLO name", "Output #", "Include", "Output name"]
        )
        yolo_table.setRowCount(len(YOLO_KPT_NAMES))
        yolo_table.verticalHeader().setVisible(False)
        yolo_table.setAlternatingRowColors(True)
        yolo_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        yolo_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        # Inverse mapping from defs
        inv_yolo: dict[int, KeypointDef] = {}
        for d in self._result_defs.values():
            if d.source == "yolo" and d.yolo_slot:
                inv_yolo[int(d.yolo_slot)] = d

        self._yolo_slot_spins: list[QSpinBox] = []
        self._yolo_include_checks: list[QCheckBox] = []
        self._yolo_name_edits: list[QLineEdit] = []

        for r, name in enumerate(YOLO_KPT_NAMES):
            yolo_slot = r + 1
            yolo_table.setItem(r, 0, QTableWidgetItem(str(yolo_slot)))
            yolo_table.setItem(r, 1, QTableWidgetItem(name))

            spin = QSpinBox()
            spin.setRange(1, 99)
            spin.setValue(
                int(
                    inv_yolo.get(yolo_slot, KeypointDef(0, "", "", None)).slot
                    or yolo_slot
                )
            )
            self._yolo_slot_spins.append(spin)
            yolo_table.setCellWidget(r, 2, spin)

            chk = QCheckBox()
            chk.setChecked(yolo_slot in inv_yolo)
            self._yolo_include_checks.append(chk)
            w = QWidget()
            l = QHBoxLayout(w)
            l.setContentsMargins(0, 0, 0, 0)
            l.addWidget(chk)
            l.addStretch(1)
            yolo_table.setCellWidget(r, 3, w)

            nm = inv_yolo[yolo_slot].name if yolo_slot in inv_yolo else name
            edit = QLineEdit(nm)
            self._yolo_name_edits.append(edit)
            yolo_table.setCellWidget(r, 4, edit)

        yolo_table.resizeColumnsToContents()
        yolo_table.setColumnWidth(1, 200)
        root.addWidget(QLabel("YOLO keypoints"))
        root.addWidget(yolo_table, stretch=1)

        # --- Custom keypoints table
        root.addWidget(QLabel("Custom keypoints"))
        custom_table = QTableWidget()
        custom_table.setColumnCount(4)
        custom_table.setHorizontalHeaderLabels(
            ["Output #", "Name", "Include", "Delete"]
        )
        custom_table.verticalHeader().setVisible(False)
        custom_table.setAlternatingRowColors(True)
        custom_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        custom_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._custom_table = custom_table
        self._custom_slot_spins: list[QSpinBox] = []
        self._custom_name_edits: list[QLineEdit] = []
        self._custom_include_checks: list[QCheckBox] = []
        self._custom_delete_checks: list[QCheckBox] = []

        custom_defs = [d for d in self._result_defs.values() if d.source == "custom"]
        custom_defs.sort(key=lambda d: d.slot)
        self._populate_custom_rows(custom_defs)

        btn_add_custom = QPushButton("Add custom keypoint")
        btn_add_custom.clicked.connect(self._add_custom_row)
        root.addWidget(btn_add_custom)
        root.addWidget(custom_table, stretch=1)

        buttons = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_apply = QPushButton("Apply")
        btn_cancel.clicked.connect(self.reject)
        btn_apply.clicked.connect(self._apply)
        buttons.addStretch(1)
        buttons.addWidget(btn_cancel)
        buttons.addWidget(btn_apply)
        root.addLayout(buttons)

    def defs(self) -> dict[int, KeypointDef]:
        return {k: KeypointDef(**vars(v)) for k, v in self._result_defs.items()}

    def _populate_custom_rows(self, defs_list: list[KeypointDef]) -> None:
        t: QTableWidget = self._custom_table
        t.setRowCount(len(defs_list))
        self._custom_slot_spins.clear()
        self._custom_name_edits.clear()
        self._custom_include_checks.clear()
        self._custom_delete_checks.clear()
        for r, d in enumerate(defs_list):
            spin = QSpinBox()
            spin.setRange(1, 99)
            spin.setValue(int(d.slot))
            self._custom_slot_spins.append(spin)
            t.setCellWidget(r, 0, spin)

            edit = QLineEdit(d.name)
            self._custom_name_edits.append(edit)
            t.setCellWidget(r, 1, edit)

            inc = QCheckBox()
            inc.setChecked(True)
            self._custom_include_checks.append(inc)
            w = QWidget()
            l = QHBoxLayout(w)
            l.setContentsMargins(0, 0, 0, 0)
            l.addWidget(inc)
            l.addStretch(1)
            t.setCellWidget(r, 2, w)

            dele = QCheckBox()
            dele.setChecked(False)
            self._custom_delete_checks.append(dele)
            w2 = QWidget()
            l2 = QHBoxLayout(w2)
            l2.setContentsMargins(0, 0, 0, 0)
            l2.addWidget(dele)
            l2.addStretch(1)
            t.setCellWidget(r, 3, w2)

        t.resizeColumnsToContents()
        t.setColumnWidth(1, 220)

    def _add_custom_row(self) -> None:
        # Append a blank custom keypoint
        current = []
        for r in range(self._custom_table.rowCount()):
            slot = int(self._custom_slot_spins[r].value())
            name = self._custom_name_edits[r].text().strip() or f"custom_{slot}"
            current.append(
                KeypointDef(slot=slot, name=name, source="custom", yolo_slot=None)
            )
        # pick next available slot
        used = {d.slot for d in current}
        next_slot = 1
        while next_slot in used:
            next_slot += 1
        current.append(
            KeypointDef(slot=next_slot, name="Trachea", source="custom", yolo_slot=None)
        )
        self._populate_custom_rows(current)

    def _apply(self) -> None:
        out_defs: dict[int, KeypointDef] = {}

        # YOLO rows
        for r, yolo_name in enumerate(YOLO_KPT_NAMES):
            yolo_slot = r + 1
            include = self._yolo_include_checks[r].isChecked()
            if not include:
                continue
            out_slot = int(self._yolo_slot_spins[r].value())
            out_name = self._yolo_name_edits[r].text().strip() or yolo_name
            out_defs[out_slot] = KeypointDef(
                slot=out_slot, name=out_name, source="yolo", yolo_slot=yolo_slot
            )

        # Custom rows
        for r in range(self._custom_table.rowCount()):
            if self._custom_delete_checks[r].isChecked():
                continue
            if not self._custom_include_checks[r].isChecked():
                continue
            out_slot = int(self._custom_slot_spins[r].value())
            out_name = self._custom_name_edits[r].text().strip() or f"custom_{out_slot}"
            out_defs[out_slot] = KeypointDef(
                slot=out_slot, name=out_name, source="custom", yolo_slot=None
            )

        if not out_defs:
            QMessageBox.warning(
                self, "Invalid mapping", "You removed all keypoints. Keep at least one."
            )
            return

        self._result_defs = out_defs
        self.accept()


@dataclass
class AppState:
    images_dir: Optional[Path] = None
    images: list[Path] = None  # type: ignore[assignment]
    index: int = 0

    def __post_init__(self) -> None:
        if self.images is None:
            self.images = []


class AnnotateWorker(QObject):
    progress = Signal(int, int)  # processed, total
    finished = Signal(int, int, str)  # processed, written, out_root
    failed = Signal(str)

    def __init__(
        self,
        images_dir: Path,
        model: str,
        device: Optional[str],
        imgsz: int,
        conf: float,
        iou: float,
        kpt_conf: float,
        max_det: int,
        predict_batch: int,
        recursive: bool,
        decimals: int,
        keypoint_slot_map: Optional[dict[int, int]] = None,
        output_kpt_count: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.images_dir = images_dir
        self.model = model
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.kpt_conf = kpt_conf
        self.max_det = max_det
        self.predict_batch = predict_batch
        self.recursive = recursive
        self.decimals = decimals
        self.keypoint_slot_map = keypoint_slot_map
        self.output_kpt_count = output_kpt_count

    def run(self) -> None:
        try:
            processed, written = run_auto_annotate(
                images_dir=self.images_dir,
                labels_dir=None,
                model_name=self.model,
                device=self.device,
                imgsz=self.imgsz,
                box_conf=self.conf,
                iou=self.iou,
                kpt_conf=self.kpt_conf,
                recursive=self.recursive,
                dry_run=False,
                coord_decimals=self.decimals,
                max_det=self.max_det,
                predict_batch=self.predict_batch,
                keypoint_slot_map=self.keypoint_slot_map,
                output_kpt_count=self.output_kpt_count,
                labels_same_folder_as_images=True,
            )
            out_root = str(self.images_dir.resolve())
            self.finished.emit(processed, written, out_root)
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Pose Annotator (YOLO)")
        self.state = AppState()
        self._model: Optional[YOLO] = None
        self._thread: Optional[QThread] = None
        self._worker: Optional[AnnotateWorker] = None
        self._kp_defs: dict[int, KeypointDef] = _default_keypoint_defs()
        self._edit_xyxy: Optional[np.ndarray] = None  # (N,4) xyxy pixels
        self._edit_kpts_xyv: Optional[np.ndarray] = None  # (N,K,3) pixels + vis
        self._edit_cls: Optional[np.ndarray] = None  # (N,) int
        # Detection delete flag: hides bbox and skips saving, but keeps keypoints visible/editable.
        self._det_deleted: Optional[np.ndarray] = None  # (N,) bool
        self._add_mode = False
        self._bbox_guide_mode = False
        self._autosave_enabled = False
        # Last bbox (xyxy normalized to source image) + class when navigating away — Ctrl+V only replaces bbox.
        self._bbox_clipboard: Optional[tuple[np.ndarray, np.ndarray]] = None
        # Last person's keypoints (normalized x,y in [0,1]) when navigating away — Ctrl+B paste.
        self._kpts_clipboard: Optional[np.ndarray] = None
        self._overlay_rects: list[QGraphicsRectItem] = []
        self._overlay_kps: list[DraggableKeypoint] = []

        root = QWidget()
        outer = QHBoxLayout(root)
        outer.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(1)
        outer.addWidget(splitter)
        self._splitter = splitter

        # Controls
        controls = QGroupBox("Project")
        controls_layout = QGridLayout(controls)

        self.images_dir_edit = QLineEdit()
        self.images_dir_edit.setReadOnly(True)

        btn_pick = QPushButton("Select Images Folder…")
        btn_pick.clicked.connect(self.pick_images_dir)

        controls_layout.addWidget(QLabel("Images folder"), 0, 0)
        controls_layout.addWidget(self.images_dir_edit, 0, 1)
        controls_layout.addWidget(btn_pick, 0, 2)

        self.current_image_project = QLabel("Images: 0/0")
        self.current_image_project.setWordWrap(True)
        self.current_image_project.setStyleSheet("QLabel { color: #1b8f2e; font-weight: 600; }")
        controls_layout.addWidget(self.current_image_project, 1, 0, 1, 3)

        # Settings
        settings = QGroupBox("Model settings")
        form = QFormLayout(settings)

        self.model_edit = QLineEdit("yolo26x-pose.pt")
        self.device_edit = QLineEdit("0")
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(128, 4096)
        self.imgsz_spin.setValue(448)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.5)

        self.kpt_conf_spin = QDoubleSpinBox()
        self.kpt_conf_spin.setRange(0.0, 1.0)
        self.kpt_conf_spin.setSingleStep(0.05)
        self.kpt_conf_spin.setValue(0.25)

        self.max_det_spin = QSpinBox()
        self.max_det_spin.setRange(1, 3000)
        self.max_det_spin.setValue(300)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(16)

        self.decimals_spin = QSpinBox()
        self.decimals_spin.setRange(0, 10)
        self.decimals_spin.setValue(6)

        self.recursive_check = QCheckBox("Recursive (include subfolders)")
        self.recursive_check.setChecked(False)

        form.addRow("Model (.pt / .onnx)", self.model_edit)
        form.addRow("Device (0 / cpu / blank)", self.device_edit)
        form.addRow("imgsz", self.imgsz_spin)
        form.addRow("box conf", self.conf_spin)
        form.addRow("IoU", self.iou_spin)
        form.addRow("kpt conf", self.kpt_conf_spin)
        form.addRow("max det", self.max_det_spin)
        form.addRow("predict batch", self.batch_spin)
        form.addRow("label decimals", self.decimals_spin)
        form.addRow("", self.recursive_check)

        mapping_box = QGroupBox("Keypoint mapping")
        mapping_layout = QVBoxLayout(mapping_box)
        self.mapping_enable = QCheckBox("Enable custom mapping (17-slot output)")
        self.mapping_enable.setChecked(True)
        btn_mapping = QPushButton("Mapping…")
        btn_mapping.clicked.connect(self.open_mapping_dialog)
        row = QHBoxLayout()
        row.addWidget(self.mapping_enable)
        row.addStretch(1)
        row.addWidget(btn_mapping)

        self.mapping_label = QLabel(_defs_to_text(self._kp_defs))
        self.mapping_label.setTextFormat(Qt.TextFormat.RichText)
        self.mapping_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.mapping_label.setStyleSheet("QLabel { color: #444; }")
        self.mapping_label.setWordWrap(True)
        mapping_layout.addLayout(row)
        mapping_layout.addWidget(self.mapping_label)

        # Preview area
        preview = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview)
        self.current_image_preview = QLabel("Current image: (none)")
        self.current_image_preview.setWordWrap(True)
        self.current_image_preview.setStyleSheet("QLabel { color: #1b8f2e; font-weight: 600; }")
        preview_layout.addWidget(self.current_image_preview)
        self.viewer = ImageViewer()
        self.viewer.setMinimumSize(640, 480)
        self.viewer.set_placeholder("Select an images folder to begin.")
        preview_layout.addWidget(self.viewer)

        # Paste bbox: only when preview has focus (so Ctrl+V still works in text fields).
        self._shortcut_paste_bbox = QShortcut(QKeySequence.StandardKey.Paste, self.viewer)
        self._shortcut_paste_bbox.setContext(Qt.ShortcutContext.WidgetShortcut)
        self._shortcut_paste_bbox.activated.connect(self.paste_clipboard_bbox)

        self._shortcut_paste_kpts = QShortcut(QKeySequence("Ctrl+B"), self.viewer)
        self._shortcut_paste_kpts.setContext(Qt.ShortcutContext.WidgetShortcut)
        self._shortcut_paste_kpts.activated.connect(self.paste_clipboard_keypoints)

        # Right-side tool panel (will be placed to the far right)
        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.predict_btn = QPushButton("Predict / Preview")
        self.save_btn = QPushButton("Save label")
        self.autosave_btn = QPushButton("Auto save: OFF")
        self.autosave_btn.setCheckable(True)
        self.annotate_btn = QPushButton("Auto-annotate folder")

        self.save_btn.clicked.connect(self.save_current_label)
        self.save_btn.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)
        )
        self.autosave_btn.toggled.connect(self._on_autosave_toggled)
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.predict_btn.clicked.connect(self.predict_current)
        self.annotate_btn.clicked.connect(self.annotate_folder)

        # Global navigation shortcuts (work anywhere in the GUI)
        self._shortcut_prev = QShortcut(QKeySequence("A"), self)
        self._shortcut_prev.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._shortcut_prev.activated.connect(self.prev_image)

        self._shortcut_next = QShortcut(QKeySequence("D"), self)
        self._shortcut_next.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._shortcut_next.activated.connect(self.next_image)

        self._shortcut_bbox_guide = QShortcut(QKeySequence("W"), self)
        self._shortcut_bbox_guide.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._shortcut_bbox_guide.activated.connect(self.toggle_bbox_guide_mode)

        self.add_mode_btn = QPushButton("Add keypoint")
        self.add_mode_btn.setCheckable(True)
        self.add_mode_btn.toggled.connect(self.toggle_add_mode)
        self.add_slot_spin = QSpinBox()
        self.add_slot_spin.setRange(1, 99)
        self.add_slot_spin.setValue(1)
        self.add_person_spin = QSpinBox()
        self.add_person_spin.setRange(1, 1)
        self.add_person_spin.setValue(1)

        tool_panel = QGroupBox("Tools")
        tool_layout = QVBoxLayout(tool_panel)
        tool_layout.setContentsMargins(8, 8, 8, 8)
        tool_layout.setSpacing(8)

        tool_layout.addWidget(self.prev_btn)
        tool_layout.addWidget(self.next_btn)

        tool_layout.addSpacing(8)
        tool_layout.addWidget(QLabel("KP#"))
        tool_layout.addWidget(self.add_slot_spin)
        tool_layout.addWidget(QLabel("Person"))
        tool_layout.addWidget(self.add_person_spin)
        tool_layout.addWidget(self.add_mode_btn)

        tool_layout.addSpacing(8)
        tool_layout.addWidget(self.predict_btn)
        tool_layout.addWidget(self.save_btn)
        tool_layout.addWidget(self.autosave_btn)
        tool_layout.addWidget(self.annotate_btn)

        # Keypoints present list (current image)
        tool_layout.addSpacing(10)
        tool_layout.addWidget(QLabel("Keypoints present"))
        self.kp_list_area = QScrollArea()
        self.kp_list_area.setWidgetResizable(True)
        self.kp_list_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.kp_list_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.kp_list_container = QWidget()
        self.kp_list_layout = QVBoxLayout(self.kp_list_container)
        self.kp_list_layout.setContentsMargins(4, 4, 4, 4)
        self.kp_list_layout.setSpacing(6)
        self.kp_list_layout.addStretch(1)
        self.kp_list_area.setWidget(self.kp_list_container)
        tool_layout.addWidget(self.kp_list_area, stretch=1)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(36)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(controls)
        left_layout.addWidget(settings)
        left_layout.addWidget(mapping_box)
        left_layout.addWidget(self.status)
        left_layout.addWidget(self.progress)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        # Split preview (left) and tools (right)
        right_split = QSplitter(Qt.Orientation.Horizontal)
        right_split.setChildrenCollapsible(False)
        right_split.setHandleWidth(1)
        right_layout.addWidget(right_split)

        preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        tool_panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        right_split.addWidget(preview)
        right_split.addWidget(tool_panel)
        right_split.setStretchFactor(0, 19)  # ~95%
        right_split.setStretchFactor(1, 1)  # ~5%
        self._right_split = right_split
        self._tool_panel = tool_panel

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)  # ~10%
        splitter.setStretchFactor(1, 9)  # ~90%
        self._left_panel = left
        self._right_panel = right

        # Lock left panel to ~10% of window width (min 320px).
        self._left_min_px = 320
        self._tool_min_px = 180
        self._apply_left_width(force=True)
        self._apply_tool_width(force=True)

        self.setCentralWidget(root)
        self._update_buttons()
        self._update_current_image_labels()
        self._update_present_keypoints_list()

    def _apply_left_width(self, force: bool = False) -> None:
        if not hasattr(self, "_splitter") or not hasattr(self, "_left_panel"):
            return
        w = max(self._left_min_px, int(self.width() * 0.10))
        self._left_panel.setMinimumWidth(w)
        self._left_panel.setMaximumWidth(w)
        if force:
            # Ensure splitter sizes match the locked width.
            total = max(1, self.width())
            self._splitter.setSizes([w, max(1, total - w)])

    def _apply_tool_width(self, force: bool = False) -> None:
        if not hasattr(self, "_right_split") or not hasattr(self, "_tool_panel"):
            return
        w = max(self._tool_min_px, int(self.width() * 0.05))
        self._tool_panel.setMinimumWidth(w)
        self._tool_panel.setMaximumWidth(w)
        if force:
            total = max(
                1,
                (
                    self._right_panel.width()
                    if hasattr(self, "_right_panel")
                    else self.width()
                ),
            )
            self._right_split.setSizes([max(1, total - w), w])

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_left_width(force=False)
        self._apply_tool_width(force=False)

    # --- Helpers
    def _update_buttons(self) -> None:
        has_images = bool(self.state.images)
        self.prev_btn.setEnabled(has_images and self.state.index > 0)
        self.next_btn.setEnabled(
            has_images and self.state.index < len(self.state.images) - 1
        )
        self.predict_btn.setEnabled(has_images)
        self.annotate_btn.setEnabled(
            self.state.images_dir is not None and len(self.state.images) > 0
        )

    def _update_current_image_labels(self) -> None:
        if self.state.images_dir is None or not self.state.images:
            self.current_image_project.setText("Images: 0/0")
            self.current_image_preview.setText("Current image: (none)")
            return

        img = self._current_image()
        rel = img.relative_to(self.state.images_dir)
        rel_str = str(rel).replace("/", "\\")
        display = f"{self.state.images_dir.name}\\{rel_str} ({self.state.index + 1}/{len(self.state.images)})"
        self.current_image_project.setText(f"Images: {self.state.index + 1}/{len(self.state.images)}")
        self.current_image_preview.setText(display)

    def _clear_layout_widgets(self, layout: QVBoxLayout) -> None:
        # Remove all widgets except final stretch.
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            w = item.widget()
            if w is not None:
                layout.removeWidget(w)
                w.deleteLater()

    def _update_present_keypoints_list(self) -> None:
        """Populate right-side list of keypoints present in current image."""
        if not hasattr(self, "kp_list_layout"):
            return
        self._clear_layout_widgets(self.kp_list_layout)

        if self._edit_kpts_xyv is None:
            # Keep stretch at bottom
            self.kp_list_layout.addStretch(1)
            return

        defs = self._active_defs()
        present_slots: list[int] = []
        for slot in sorted(defs.keys()):
            idx = slot - 1
            if idx < 0 or idx >= self._edit_kpts_xyv.shape[1]:
                continue
            # visible in any person?
            if np.any(self._edit_kpts_xyv[:, idx, 2] > 0):
                present_slots.append(slot)

        for slot in present_slots:
            d = defs[slot]
            row = QWidget()
            hl = QHBoxLayout(row)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(8)

            dot = QLabel()
            dot.setFixedSize(12, 12)
            c = _slot_color(slot)
            # Use same fill color but fully opaque for legend dot.
            c2 = QColor(c)
            c2.setAlpha(255)
            dot.setStyleSheet(
                f"background-color: rgba({c2.red()}, {c2.green()}, {c2.blue()}, 255);"
                "border-radius: 6px;"
                "border: 1px solid rgba(0,0,0,120);"
            )

            name = d.name
            text = QLabel(f"{name}  →  {slot}")
            text.setWordWrap(True)
            text.setStyleSheet("QLabel { color: #222; }")

            hl.addWidget(dot)
            hl.addWidget(text, stretch=1)
            self.kp_list_layout.addWidget(row)

        self.kp_list_layout.addStretch(1)

    def _device_value(self) -> Optional[str]:
        t = self.device_edit.text().strip()
        if not t:
            return None
        # Ultralytics accepts "0" or "cpu" or "cuda:0"
        return t

    def _effective_device(self) -> Optional[str]:
        """
        Return a device string that won't crash when CUDA isn't available.

        If user requested a CUDA device (e.g. "0" or "cuda:0") but torch reports no CUDA,
        fall back to "cpu".
        """
        req = self._device_value()
        if req is None:
            return None

        r = req.lower()
        wants_cuda = r.startswith("cuda") or r.isdigit()
        if not wants_cuda:
            return req

        try:
            import torch  # type: ignore
        except Exception:
            # If torch import fails, let Ultralytics handle it.
            return req

        if torch.cuda.is_available():
            return req

        # Fallback
        self.status.setText("CUDA not available in this environment; using CPU.")
        return "cpu"

    def _write_keypoints_schema_file(self, image_path: Path, result) -> None:
        """Write keypoints.txt beside the images folder metadata: image size + numbered keypoint names."""
        os_shape = getattr(result, "orig_shape", None)
        if os_shape is not None and len(os_shape) >= 2:
            h, w = int(os_shape[0]), int(os_shape[1])
        else:
            try:
                import cv2  # type: ignore

                bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if bgr is not None:
                    h, w = int(bgr.shape[0]), int(bgr.shape[1])
                else:
                    h, w = 0, 0
            except Exception:
                h, w = 0, 0

        defs = self._active_defs()
        lines = [
            f"image_width: {w}",
            f"image_height: {h}",
            "",
            "Keypoints:",
        ]
        for slot in sorted(defs.keys()):
            d = defs[slot]
            lines.append(f"{d.slot}. {d.name}")

        out_path = image_path.parent / "keypoints.txt"
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _active_map(self) -> Optional[dict[int, int]]:
        if not self.mapping_enable.isChecked():
            return None
        # only yolo-mapped defs
        return {slot: d.yolo_slot for slot, d in self._kp_defs.items() if d.source == "yolo" and d.yolo_slot}  # type: ignore[misc]

    def _active_defs(self) -> dict[int, KeypointDef]:
        if not self.mapping_enable.isChecked():
            # identity 1..17
            return {
                i
                + 1: KeypointDef(
                    slot=i + 1, name=YOLO_KPT_NAMES[i], source="yolo", yolo_slot=i + 1
                )
                for i in range(17)
            }
        return dict(self._kp_defs)

    def _output_slot_to_yolo_slot(self, output_slot: int) -> Optional[int]:
        """Translate displayed/output slot -> underlying YOLO slot (1..K)."""
        defs = self._active_defs()
        d = defs.get(int(output_slot))
        if d is None:
            return None
        if d.source == "yolo":
            return int(d.yolo_slot or 0) or None
        return None

    def _ensure_model(self) -> YOLO:
        model_path = self.model_edit.text().strip()
        if not model_path:
            raise ValueError("Model path is empty.")
        # Reload if model changed
        if (
            self._model is None
            or getattr(self._model, "model", None) is None
            or getattr(self._model, "ckpt_path", None) != model_path
        ):
            self._model = YOLO(model_path)
        return self._model

    def _current_image(self) -> Path:
        return self.state.images[self.state.index]

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    # --- Actions
    def pick_images_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select images folder")
        if not folder:
            return
        images_dir = Path(folder)
        self.state.images_dir = images_dir
        self.images_dir_edit.setText(str(images_dir))
        # Do not auto-annotate or auto-predict on selection; just list images.
        self.state.images = list(
            iter_images(images_dir, recursive=bool(self.recursive_check.isChecked()))
        )
        self.state.index = 0
        self.status.setText(f"Found {len(self.state.images)} images.")
        self._update_buttons()
        self._update_current_image_labels()
        self._update_present_keypoints_list()
        if (images_dir / "keypoints.txt").exists():
            self.annotate_btn.setEnabled(False)
            self.status.setText("This folder is already labeled (keypoints.txt found). Auto-annotate disabled.")
            self.viewer.set_placeholder("Click 'Predict / Preview' to view existing labels.")
        else:
            self.annotate_btn.setEnabled(True)
            self.viewer.set_placeholder("Click 'Predict / Preview' to visualize.")

    def prev_image(self) -> None:
        if self.state.index > 0:
            if not self._maybe_autosave_current_before_navigation():
                return
            self._capture_bbox_clipboard_from_current_image()
            self._capture_kpts_clipboard_from_current_image()
            self.state.index -= 1
            self._update_buttons()
            self._update_current_image_labels()
            self.predict_current()

    def next_image(self) -> None:
        if self.state.index < len(self.state.images) - 1:
            if not self._maybe_autosave_current_before_navigation():
                return
            self._capture_bbox_clipboard_from_current_image()
            self._capture_kpts_clipboard_from_current_image()
            self.state.index += 1
            self._update_buttons()
            self._update_current_image_labels()
            self.predict_current()

    def _last_bbox_person_index(self) -> Optional[int]:
        """Last row index that still has a visible bbox (not bbox-deleted)."""
        if self._edit_xyxy is None or self._edit_xyxy.shape[0] == 0:
            return None
        n = int(self._edit_xyxy.shape[0])
        del_m = (
            self._det_deleted
            if self._det_deleted is not None
            else np.zeros((n,), dtype=bool)
        )
        for i in range(n - 1, -1, -1):
            if not bool(del_m[i]):
                return i
        return None

    def _last_keypoints_source_person_index(self) -> Optional[int]:
        """Person row to copy keypoints from: prefer last visible bbox; else last row with any kp."""
        i = self._last_bbox_person_index()
        if i is not None:
            return i
        if self._edit_kpts_xyv is None or self._edit_kpts_xyv.shape[0] == 0:
            return None
        n = int(self._edit_kpts_xyv.shape[0])
        for j in range(n - 1, -1, -1):
            if np.any(self._edit_kpts_xyv[j, :, 2] > 0):
                return j
        return None

    def _read_image_wh(self, image_path: Path) -> Optional[tuple[int, int]]:
        try:
            import cv2  # type: ignore

            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            return int(bgr.shape[1]), int(bgr.shape[0])
        except Exception:
            return None

    def _capture_bbox_clipboard_from_current_image(self) -> None:
        """Remember the last visible bbox (normalized xyxy) + class — keypoints use separate clipboard (Ctrl+B)."""
        if (
            self._edit_xyxy is None
            or self._edit_cls is None
            or self._edit_xyxy.shape[0] == 0
        ):
            self._bbox_clipboard = None
            return
        idx = self._last_bbox_person_index()
        if idx is None:
            self._bbox_clipboard = None
            return
        wh = self._read_image_wh(self._current_image())
        if wh is None:
            self._bbox_clipboard = None
            return
        w, h = wh[0], wh[1]
        if w <= 0 or h <= 0:
            self._bbox_clipboard = None
            return
        xy = self._edit_xyxy[idx].astype(np.float64).copy().reshape(4)
        xy_norm = xy.copy()
        xy_norm[0] /= float(w)
        xy_norm[2] /= float(w)
        xy_norm[1] /= float(h)
        xy_norm[3] /= float(h)
        self._bbox_clipboard = (
            xy_norm.reshape(1, 4),
            self._edit_cls[idx : idx + 1].copy(),
        )

    def _capture_kpts_clipboard_from_current_image(self) -> None:
        """Remember keypoints for the source person as normalized x,y (matches bbox source when possible)."""
        if self._edit_kpts_xyv is None or not self.state.images:
            self._kpts_clipboard = None
            return
        idx = self._last_keypoints_source_person_index()
        if idx is None:
            self._kpts_clipboard = None
            return
        wh = self._read_image_wh(self._current_image())
        if wh is None:
            self._kpts_clipboard = None
            return
        w, h = wh[0], wh[1]
        if w <= 0 or h <= 0:
            self._kpts_clipboard = None
            return
        row = self._edit_kpts_xyv[idx].astype(np.float64).copy()
        row[:, 0] /= float(w)
        row[:, 1] /= float(h)
        self._kpts_clipboard = row

    def paste_clipboard_bbox(self) -> None:
        """Paste bbox only from previous image (Ctrl+V). Replaces bbox + class for selected Person; keypoints unchanged."""
        if not self.state.images:
            return
        if self._bbox_clipboard is None:
            self.status.setText(
                "No bbox to paste: draw or load a bbox on an image, then use Next/Previous."
            )
            return
        if self._edit_xyxy is None or self._edit_kpts_xyv is None or self._edit_cls is None:
            if not self._ensure_image_loaded_empty_editable():
                self.status.setText("Load this image (Predict / Preview) before pasting a bbox.")
                return

        xy_norm_arr, cls_src = self._bbox_clipboard
        wh = self._read_image_wh(self._current_image())
        if wh is None:
            self.status.setText("Could not read current image size for bbox paste.")
            return
        W, H = wh[0], wh[1]
        xy_norm = xy_norm_arr[0].astype(np.float64).copy()
        xy_px = xy_norm.copy()
        xy_px[0] *= float(W)
        xy_px[2] *= float(W)
        xy_px[1] *= float(H)
        xy_px[3] *= float(H)

        person_idx = int(self.add_person_spin.value()) - 1
        n = int(self._edit_xyxy.shape[0])

        if self._det_deleted is None:
            self._det_deleted = np.zeros((n,), dtype=bool)

        if n > 0 and 0 <= person_idx < n:
            self._edit_xyxy[person_idx] = xy_px
            self._edit_cls[person_idx] = int(cls_src[0])
            self._det_deleted[person_idx] = False
            self.status.setText(
                f"Replaced bbox for Person {person_idx + 1} (Ctrl+V). Keypoints unchanged — use Ctrl+B for keypoints."
            )
        else:
            out_k = int(self._edit_kpts_xyv.shape[1])
            self._edit_xyxy = np.vstack([self._edit_xyxy, xy_px.reshape(1, 4)])
            self._edit_kpts_xyv = np.vstack(
                [self._edit_kpts_xyv, np.zeros((1, out_k, 3), dtype=np.float64)]
            )
            self._edit_cls = np.concatenate([self._edit_cls, cls_src.astype(np.int64)])
            self._det_deleted = np.concatenate(
                [self._det_deleted, np.zeros((1,), dtype=bool)]
            )
            self.add_person_spin.setRange(1, max(1, int(self._edit_xyxy.shape[0])))
            self.status.setText(
                "Added bbox from previous image (Ctrl+V). Keypoints empty for new person — use Ctrl+B to paste keypoints."
            )

        self._render_editable_overlay(fit=False)
        self._update_present_keypoints_list()

    def paste_clipboard_keypoints(self) -> None:
        """Paste keypoints from previous image onto selected Person (Ctrl+B). Coords rescale to current image size."""
        if not self.state.images:
            return
        if self._kpts_clipboard is None:
            self.status.setText(
                "No keypoints to paste: edit or load keypoints on an image, then Next/Previous."
            )
            return
        if (
            self._edit_xyxy is None
            or self._edit_kpts_xyv is None
            or self._edit_xyxy.shape[0] == 0
        ):
            self.status.setText(
                "Need at least one person on this frame (bbox). Run Predict or paste a bbox (Ctrl+V), then try Ctrl+B."
            )
            return

        person_idx = int(self.add_person_spin.value()) - 1
        if person_idx < 0 or person_idx >= self._edit_kpts_xyv.shape[0]:
            self.status.setText("Invalid Person # for keypoint paste.")
            return

        wh = self._read_image_wh(self._current_image())
        if wh is None:
            self.status.setText("Could not read current image size for paste.")
            return
        w, h = wh[0], wh[1]

        out_k = int(self._edit_kpts_xyv.shape[1])
        src_k = int(self._kpts_clipboard.shape[0])
        k_use = min(out_k, src_k)

        restored = np.zeros((out_k, 3), dtype=np.float64)
        if k_use > 0:
            chunk = self._kpts_clipboard[:k_use].astype(np.float64).copy()
            chunk[:, 0] *= float(w)
            chunk[:, 1] *= float(h)
            restored[:k_use, :] = chunk

        self._edit_kpts_xyv[person_idx, :, :] = restored
        self._render_editable_overlay(fit=False)
        self._update_present_keypoints_list()
        self.status.setText(
            f"Pasted keypoints from previous image onto Person {person_idx + 1} (Ctrl+B)."
        )

    def _on_autosave_toggled(self, checked: bool) -> None:
        self._autosave_enabled = bool(checked)
        self.autosave_btn.setText("Auto save: ON" if checked else "Auto save: OFF")

    def _maybe_autosave_current_before_navigation(self) -> bool:
        """If Auto save is enabled, save current label before changing images.

        Returns True if navigation should proceed.
        """
        if not self._autosave_enabled:
            return True
        # Only autosave if we have a prediction/editable data for current image.
        if (
            self._edit_xyxy is None
            or self._edit_kpts_xyv is None
            or self._edit_cls is None
            or self.state.images_dir is None
            or not self.state.images
        ):
            return True

        before = self._current_image()
        try:
            ok = self._save_current_label(return_bool=True)
        except Exception:
            ok = False

        if ok:
            self.status.setText(f"Auto-saved: {before.with_suffix('.txt').name}")
            return True

        # If save failed, block navigation to avoid losing edits.
        self._show_error("Auto save failed", "Could not auto-save the current label. Fix the issue and try again.")
        return False

    def predict_current(self) -> None:
        if not self.state.images:
            return
        try:
            img_path = self._current_image()
            self._update_current_image_labels()
            labeled_folder = bool(self.state.images_dir and (self.state.images_dir / "keypoints.txt").exists())
            label_path = img_path.with_suffix(".txt")

            if labeled_folder:
                if not label_path.exists():
                    self.viewer.set_placeholder("No label file found for this image yet.")
                    self.status.setText(
                        f"Viewing labels: missing {label_path.name} ({self.state.index+1}/{len(self.state.images)})"
                    )
                    self._edit_xyxy = None
                    self._edit_kpts_xyv = None
                    self._edit_cls = None
                    self._det_deleted = None
                    self._update_present_keypoints_list()
                    return

                self.status.setText(
                    f"Viewing labels: {img_path.name} ({self.state.index+1}/{len(self.state.images)})"
                )
                self._load_editable_from_label_file(label_path, img_path)
                self._render_editable_overlay(fit=True)
                self._update_present_keypoints_list()
                self.viewer.setFocus(Qt.FocusReason.OtherFocusReason)
                return

            model = self._ensure_model()
            self.status.setText(
                f"Predicting: {img_path.name} ({self.state.index+1}/{len(self.state.images)})"
            )

            results = model.predict(
                source=str(img_path),
                imgsz=int(self.imgsz_spin.value()),
                conf=float(self.conf_spin.value()),
                iou=float(self.iou_spin.value()),
                device=self._effective_device(),
                max_det=int(self.max_det_spin.value()),
                verbose=False,
            )
            if not results:
                self.viewer.set_placeholder("No results.")
                return

            self._write_keypoints_schema_file(img_path, results[0])
            self._load_editable_from_result(results[0], img_path)
            self._render_editable_overlay(fit=True)
            self._update_present_keypoints_list()
            self.viewer.setFocus(Qt.FocusReason.OtherFocusReason)
        except Exception:
            self._show_error("Predict failed", traceback.format_exc())

    def _load_editable_from_label_file(self, label_path: Path, image_path: Path) -> None:
        """Load editable arrays from an existing YOLO-pose label file (normalized)."""
        try:
            import cv2  # type: ignore
        except Exception:
            self._show_error("Open label failed", "OpenCV is required to view existing labels.")
            self._edit_xyxy = None
            self._edit_kpts_xyv = None
            self._edit_cls = None
            self._det_deleted = None
            return

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            self._show_error("Open image failed", "Could not read image.")
            self._edit_xyxy = None
            self._edit_kpts_xyv = None
            self._edit_cls = None
            self._det_deleted = None
            return

        h, w = int(bgr.shape[0]), int(bgr.shape[1])
        rgb = bgr[..., ::-1].copy()
        self.viewer.set_base_pixmap(_to_qpixmap(rgb), fit=True)

        txt = label_path.read_text(encoding="utf-8").strip()
        if not txt:
            self._edit_xyxy = np.zeros((0, 4), dtype=np.float64)
            defs = self._active_defs()
            out_k = max(17, max(defs.keys()) if defs else 17)
            self._edit_kpts_xyv = np.zeros((0, out_k, 3), dtype=np.float64)
            self._edit_cls = np.zeros((0,), dtype=np.int64)
            self._det_deleted = np.zeros((0,), dtype=bool)
            self.add_person_spin.setRange(1, 1)
            self.add_person_spin.setValue(1)
            return

        rows = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        parsed: list[tuple[int, float, float, float, float, list[float]]] = []
        k_infer = 0
        for ln in rows:
            parts = ln.split()
            if len(parts) < 5:
                continue
            vals = [float(x) for x in parts]
            cls = int(vals[0])
            xc, yc, bw, bh = vals[1:5]
            rest = vals[5:]
            if len(rest) % 3 == 0:
                k_infer = max(k_infer, len(rest) // 3)
            parsed.append((cls, xc, yc, bw, bh, rest))

        defs = self._active_defs()
        out_k = max(17, max(defs.keys()) if defs else 17, k_infer)

        n = len(parsed)
        xyxy = np.zeros((n, 4), dtype=np.float64)
        cls_arr = np.zeros((n,), dtype=np.int64)
        xyv = np.zeros((n, out_k, 3), dtype=np.float64)

        for i, (cls, xc, yc, bw, bh, rest) in enumerate(parsed):
            cls_arr[i] = int(cls)
            x1 = (xc - bw / 2.0) * w
            y1 = (yc - bh / 2.0) * h
            x2 = (xc + bw / 2.0) * w
            y2 = (yc + bh / 2.0) * h
            xyxy[i] = np.array([x1, y1, x2, y2], dtype=np.float64)

            for j in range(0, len(rest), 3):
                k = j // 3
                if k >= out_k:
                    break
                kx, ky, kv = rest[j], rest[j + 1], rest[j + 2]
                xyv[i, k, 0] = float(kx) * w
                xyv[i, k, 1] = float(ky) * h
                xyv[i, k, 2] = float(kv)

        self._edit_xyxy = xyxy
        self._edit_kpts_xyv = xyv
        self._edit_cls = cls_arr
        self._det_deleted = np.zeros((n,), dtype=bool)
        self.add_person_spin.setRange(1, max(1, n))
        self.add_person_spin.setValue(1)

    def _load_editable_from_result(self, result, image_path: Path) -> None:
        """Store boxes/keypoints into editable numpy arrays and load base image."""
        try:
            import cv2  # type: ignore
        except Exception:
            # Fallback: plot as base image (not editable)
            bgr = result.plot()
            rgb = bgr[..., ::-1].copy()
            self.viewer.set_base_pixmap(_to_qpixmap(rgb), fit=True)
            self._edit_xyxy = None
            self._edit_kpts_xyv = None
            return

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            bgr2 = result.plot()
            rgb = bgr2[..., ::-1].copy()
            self.viewer.set_base_pixmap(_to_qpixmap(rgb), fit=True)
            self._edit_xyxy = None
            self._edit_kpts_xyv = None
            return

        rgb = bgr[..., ::-1].copy()
        self.viewer.set_base_pixmap(_to_qpixmap(rgb), fit=True)

        boxes = result.boxes
        kpts = result.keypoints
        if boxes is None or kpts is None or boxes.xyxy is None or kpts.xy is None:
            self._edit_xyxy = None
            self._edit_kpts_xyv = None
            self._edit_cls = None
            return

        xyxy = boxes.xyxy.cpu().numpy().astype(np.float64)
        cls = boxes.cls.cpu().numpy().astype(np.int64)
        kp_xy = kpts.xy.cpu().numpy().astype(np.float64)
        kp_conf = (
            kpts.conf.cpu().numpy().astype(np.float64)
            if getattr(kpts, "conf", None) is not None
            else np.ones(kp_xy.shape[:2], dtype=np.float64)
        )

        n = min(len(xyxy), len(kp_xy))
        xyxy = xyxy[:n]
        cls = cls[:n]
        kp_xy = kp_xy[:n]
        kp_conf = kp_conf[:n]

        # Build output-slot keypoints array (allows custom slots > 17)
        defs = self._active_defs()
        out_k = max(17, max(defs.keys()) if defs else 17)
        xyv = np.zeros((n, out_k, 3), dtype=np.float64)
        thr = float(self.kpt_conf_spin.value())
        # Fill YOLO-derived slots
        for out_slot, d in defs.items():
            if d.source != "yolo" or not d.yolo_slot:
                continue
            oi = int(d.yolo_slot) - 1
            ni = int(out_slot) - 1
            if 0 <= oi < kp_xy.shape[1] and 0 <= ni < out_k:
                xyv[:, ni, 0] = kp_xy[:, oi, 0]
                xyv[:, ni, 1] = kp_xy[:, oi, 1]
                xyv[:, ni, 2] = (kp_conf[:, oi] >= thr).astype(np.float64) * 2.0
        # Custom slots stay 0/0/0 until user places them

        self._edit_xyxy = xyxy
        self._edit_kpts_xyv = xyv
        self._edit_cls = cls
        self._det_deleted = np.zeros((n,), dtype=bool)
        self.add_person_spin.setRange(1, max(1, n))
        self.add_person_spin.setValue(1)

    def _render_editable_overlay(self, fit: bool) -> None:
        """Draw bbox + draggable keypoints on top of the base image."""
        if self._edit_xyxy is None or self._edit_kpts_xyv is None:
            return

        # Remove old overlays (keep base pixmap).
        self.viewer.clear_overlays()
        self._overlay_rects.clear()
        self._overlay_kps.clear()

        # Re-set base pixmap without refit if we already zoomed, unless explicitly asked.
        # (We don't have easy access to the original pixmap here; base is already set in _load_editable_from_result.)
        if fit:
            self.viewer.fit_to_image()

        defs = self._active_defs()
        thr = float(self.kpt_conf_spin.value())

        # Draw overlays
        for i in range(self._edit_xyxy.shape[0]):
            x1, y1, x2, y2 = self._edit_xyxy[i].tolist()

            def _bbox_cb(person_idx: int):
                def _cb(nx1: float, ny1: float, nx2: float, ny2: float) -> None:
                    if self._edit_xyxy is None:
                        return
                    self._edit_xyxy[person_idx, 0] = nx1
                    self._edit_xyxy[person_idx, 1] = ny1
                    self._edit_xyxy[person_idx, 2] = nx2
                    self._edit_xyxy[person_idx, 3] = ny2

                return _cb

            def _bbox_del(person_idx: int):
                def _del() -> None:
                    if self._det_deleted is None:
                        return
                    self._det_deleted[person_idx] = True
                    self.status.setText(
                        "Deleted bbox (keypoints kept). Click Save label to persist."
                    )
                    self._render_editable_overlay(fit=False)

                return _del

            # Draw bbox only if detection not deleted.
            if self._det_deleted is None or not bool(self._det_deleted[i]):
                bbox_item = ResizableBBox(
                    QRectF(x1, y1, x2 - x1, y2 - y1),
                    on_changed=_bbox_cb(i),
                    on_deleted=_bbox_del(i),
                    person_idx=i,
                )
                self.viewer.add_overlay(bbox_item)
                self._overlay_rects.append(bbox_item)

            # Show only defined output slots
            for out_slot in sorted(defs.keys()):
                ki = int(out_slot) - 1
                if ki < 0 or ki >= self._edit_kpts_xyv.shape[1]:
                    continue
                x, y, v = self._edit_kpts_xyv[i, ki].tolist()
                if v <= 0:
                    continue
                # also respect threshold if user changed it after prediction
                # (v already derived from previous threshold; still keep a hard cutoff on current threshold by hiding v==0 only)
                _ = thr

                def _mk_callback(person_idx: int, out_idx: int):
                    def _cb(nx: float, ny: float) -> None:
                        if self._edit_kpts_xyv is None:
                            return
                        self._edit_kpts_xyv[person_idx, out_idx, 0] = nx
                        self._edit_kpts_xyv[person_idx, out_idx, 1] = ny
                        # keep visible when moved
                        if self._edit_kpts_xyv[person_idx, out_idx, 2] <= 0:
                            self._edit_kpts_xyv[person_idx, out_idx, 2] = 2.0

                    return _cb

                color = QColor(0, 200, 0)
                kp = DraggableKeypoint(
                    center_xy=(x, y),
                    radius=2.5,
                    label=str(int(out_slot)),
                    on_moved=_mk_callback(i, ki),
                    color=_slot_color(int(out_slot)),
                    person_idx=i,
                    out_slot=int(out_slot),
                )
                self.viewer.add_overlay(kp)
                self.viewer.add_overlay(kp.text_item())
                self._overlay_kps.append(kp)

    def toggle_add_mode(self, enabled: bool) -> None:
        self._add_mode = bool(enabled)
        if self._add_mode:
            if getattr(self, "_bbox_guide_mode", False):
                self._bbox_guide_mode = False
                self.viewer.set_crosshair_bbox_mode(False)
            self.viewer.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewer.setCursor(Qt.CursorShape.CrossCursor)
            self.viewer.set_click_handler(self._on_add_click)
            self.status.setText("Add mode: click on image to place selected keypoint.")
        else:
            self.viewer.set_click_handler(None)
            if getattr(self, "_bbox_guide_mode", False):
                self.viewer.setDragMode(QGraphicsView.DragMode.NoDrag)
                self.viewer.setCursor(Qt.CursorShape.CrossCursor)
            else:
                self.viewer.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                self.viewer.setCursor(Qt.CursorShape.ArrowCursor)

    def toggle_bbox_guide_mode(self) -> None:
        """Toggle crosshair + drag-to-draw bbox (shortcut W)."""
        if not self.state.images:
            self.status.setText("Select an images folder first.")
            return
        if self._edit_xyxy is None or self._edit_kpts_xyv is None or self._edit_cls is None:
            if not self._ensure_image_loaded_empty_editable():
                self.status.setText("Could not load the image for manual bbox drawing.")
                return
        if self._det_deleted is None:
            self._det_deleted = np.zeros((self._edit_xyxy.shape[0],), dtype=bool)

        self._bbox_guide_mode = not self._bbox_guide_mode
        if self._bbox_guide_mode:
            if self.add_mode_btn.isChecked():
                self.add_mode_btn.setChecked(False)
                self.toggle_add_mode(False)
            self.viewer.set_click_handler(None)
            self.viewer.set_crosshair_bbox_mode(True, self._on_manual_bbox_complete)
            self.status.setText(
                "Bbox guide (W): crosshair follows the pointer. "
                "Drag with left button from the crosshair; release sets the opposite corner. "
                "Middle-drag to pan. Press W again to exit."
            )
        else:
            self.viewer.set_crosshair_bbox_mode(False)
            self.status.setText("Bbox guide off.")

    def _ensure_image_loaded_empty_editable(self) -> bool:
        """Load current image and initialize empty label arrays (manual bbox / empty label)."""
        try:
            import cv2  # type: ignore
        except Exception:
            return False
        if not self.state.images:
            return False
        img_path = self._current_image()
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            return False
        rgb = bgr[..., ::-1].copy()
        self.viewer.set_base_pixmap(_to_qpixmap(rgb), fit=True)
        defs = self._active_defs()
        out_k = max(17, max(defs.keys()) if defs else 17)
        self._edit_xyxy = np.zeros((0, 4), dtype=np.float64)
        self._edit_kpts_xyv = np.zeros((0, out_k, 3), dtype=np.float64)
        self._edit_cls = np.zeros((0,), dtype=np.int64)
        self._det_deleted = np.zeros((0,), dtype=bool)
        self.add_person_spin.setRange(1, 1)
        self.add_person_spin.setValue(1)
        return True

    def _on_manual_bbox_complete(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> None:
        if self._edit_xyxy is None or self._edit_kpts_xyv is None or self._edit_cls is None:
            return
        if self._det_deleted is None:
            self._det_deleted = np.zeros((self._edit_xyxy.shape[0],), dtype=bool)
        out_k = self._edit_kpts_xyv.shape[1]
        row = np.array([[x1, y1, x2, y2]], dtype=np.float64)
        krow = np.zeros((1, out_k, 3), dtype=np.float64)
        self._edit_xyxy = np.vstack([self._edit_xyxy, row])
        self._edit_kpts_xyv = np.vstack([self._edit_kpts_xyv, krow])
        self._edit_cls = np.concatenate(
            [self._edit_cls, np.array([0], dtype=np.int64)]
        )
        self._det_deleted = np.concatenate(
            [self._det_deleted, np.array([False], dtype=bool)]
        )
        self.add_person_spin.setRange(1, max(1, self._edit_xyxy.shape[0]))
        self._render_editable_overlay(fit=False)
        self._update_present_keypoints_list()
        self.status.setText(
            f"Added bbox ({self._edit_xyxy.shape[0]} total). Drag corners to adjust. Press W to exit guide."
        )

    def _on_add_click(self, sx: float, sy: float) -> None:
        """Place/move a keypoint for the selected person."""
        if self._edit_kpts_xyv is None:
            return
        out_slot = int(self.add_slot_spin.value())
        # Ensure slot exists in schema; if not, prompt user to add it.
        defs = self._active_defs()
        if out_slot not in defs:
            self._show_error(
                "Keypoint not defined",
                f"Output keypoint {out_slot} is not defined. Add it in Mapping… as a custom keypoint.",
            )
            return
        out_idx = out_slot - 1
        if out_idx < 0 or out_idx >= self._edit_kpts_xyv.shape[1]:
            self._show_error(
                "Invalid keypoint", f"Output keypoint {out_slot} is out of range."
            )
            return
        person_idx = int(self.add_person_spin.value()) - 1
        if person_idx < 0 or person_idx >= self._edit_kpts_xyv.shape[0]:
            return

        self._edit_kpts_xyv[person_idx, out_idx, 0] = float(sx)
        self._edit_kpts_xyv[person_idx, out_idx, 1] = float(sy)
        self._edit_kpts_xyv[person_idx, out_idx, 2] = 2.0

        # Re-render overlays. Keep current zoom (fit=False).
        # We must keep base pixmap: easiest is to re-predict? Instead, just redraw overlays on the same scene by
        # rebuilding base again from current image quickly.
        # To keep it fast and stable, we simply call Predict/Preview renderer without refit:
        self._render_editable_overlay(fit=False)
        self._update_present_keypoints_list()

    def annotate_folder(self) -> None:
        if self.state.images_dir is None or not self.state.images:
            return
        if self._thread is not None:
            self._show_error("Busy", "Auto-annotation is already running.")
            return

        images_dir = self.state.images_dir
        model = self.model_edit.text().strip()
        device = self._effective_device()

        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status.setText("Auto-annotation started…")
        self.annotate_btn.setEnabled(False)

        worker = AnnotateWorker(
            images_dir=images_dir,
            model=model,
            device=device,
            imgsz=int(self.imgsz_spin.value()),
            conf=float(self.conf_spin.value()),
            iou=float(self.iou_spin.value()),
            kpt_conf=float(self.kpt_conf_spin.value()),
            max_det=int(self.max_det_spin.value()),
            predict_batch=int(self.batch_spin.value()),
            recursive=bool(self.recursive_check.isChecked()),
            decimals=int(self.decimals_spin.value()),
            keypoint_slot_map=(self._active_map()),
            output_kpt_count=(17 if self._active_map() else None),
        )
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        def _done(processed: int, written: int, out_root: str) -> None:
            # Ensure UI updates happen on the GUI thread
            self.status.setText(
                f"Done. Processed {processed}, wrote {written} labels next to images under {out_root}"
            )
            self.progress.setVisible(False)
            self.annotate_btn.setEnabled(True)
            if self._thread is not None:
                self._thread.quit()

        def _fail(tb: str) -> None:
            self.progress.setVisible(False)
            self.annotate_btn.setEnabled(True)
            self._show_error("Auto-annotation failed", tb)
            if self._thread is not None:
                self._thread.quit()

        # Use queued connections so callbacks run on the GUI thread, not the worker thread.
        worker.finished.connect(_done, Qt.ConnectionType.QueuedConnection)
        worker.failed.connect(_fail, Qt.ConnectionType.QueuedConnection)

        def _cleanup() -> None:
            # Thread finished; safe to release references.
            if self._worker is not None:
                self._worker.deleteLater()
            if self._thread is not None:
                self._thread.deleteLater()
            self._thread = None
            self._worker = None

        thread.finished.connect(_cleanup, Qt.ConnectionType.QueuedConnection)

        self._thread = thread
        self._worker = worker
        thread.start()

    def open_mapping_dialog(self) -> None:
        dlg = MappingDialog(self, self._kp_defs)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._kp_defs = dlg.defs()
        self.mapping_label.setText(_defs_to_text(self._kp_defs))
        if self.state.images:
            self.predict_current()

    def save_current_label(self) -> None:
        self._save_current_label(return_bool=False)

    def _save_current_label(self, return_bool: bool) -> bool:
        """Write current edited boxes/keypoints to the corresponding .txt label file."""
        if self.state.images_dir is None or not self.state.images:
            return False
        if (
            self._edit_xyxy is None
            or self._edit_kpts_xyv is None
            or self._edit_cls is None
        ):
            if not return_bool:
                self._show_error("Nothing to save", "Run Predict / Preview first.")
            return False
        img_path = self._current_image()
        label_path = img_path.with_suffix(".txt")

        # Need image size for normalization
        try:
            import cv2  # type: ignore

            im = cv2.imread(str(img_path))
            if im is None:
                raise RuntimeError("Failed to read image")
            h, w = im.shape[:2]
        except Exception:
            if not return_bool:
                self._show_error("Save failed", "Could not read image to get size.")
            return False

        lines: list[str] = []
        for i in range(self._edit_xyxy.shape[0]):
            if self._det_deleted is not None and bool(self._det_deleted[i]):
                continue
            pl = PoseLabelLine(
                class_id=int(self._edit_cls[i]),
                xyxy=self._edit_xyxy[i].astype(np.float64),
                keypoints_xyv=self._edit_kpts_xyv[i].astype(np.float64),
            )
            lines.append(
                pl.to_normalized_line(
                    w, h, coord_decimals=int(self.decimals_spin.value())
                )
            )

        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        self.status.setText(f"Saved label: {label_path}")
        return True

    def delete_selected_keypoints(self) -> None:
        """Delete selected keypoints from overlay and edited data."""
        if self._edit_kpts_xyv is None:
            return
        scene = self.viewer.scene()
        if scene is None:
            return
        selected = list(scene.selectedItems())
        if not selected:
            return

        deleted_any = False
        for it in selected:
            if not isinstance(it, DraggableKeypoint):
                continue
            p = int(it.person_idx)
            slot = int(it.out_slot)
            out_idx = slot - 1
            if (
                0 <= p < self._edit_kpts_xyv.shape[0]
                and 0 <= out_idx < self._edit_kpts_xyv.shape[1]
            ):
                # Mark as removed
                self._edit_kpts_xyv[p, out_idx, :] = 0.0
                deleted_any = True

            # Remove graphics items
            try:
                scene.removeItem(it.text_item())
            except Exception:
                pass
            scene.removeItem(it)

        if deleted_any:
            self.status.setText(
                "Deleted selected keypoint(s). Click Save label to persist."
            )
            self._update_present_keypoints_list()

    def delete_keypoint(self, person_idx: int, out_slot: int) -> None:
        """Delete a single keypoint (person index + output slot), update scene + edited data."""
        if self._edit_kpts_xyv is None:
            return
        p = int(person_idx)
        slot = int(out_slot)
        out_idx = slot - 1
        if not (0 <= p < self._edit_kpts_xyv.shape[0] and 0 <= out_idx < self._edit_kpts_xyv.shape[1]):
            return

        self._edit_kpts_xyv[p, out_idx, :] = 0.0

        scene = self.viewer.scene()
        if scene is not None:
            for it in list(scene.items()):
                if isinstance(it, DraggableKeypoint) and int(it.person_idx) == p and int(it.out_slot) == slot:
                    try:
                        scene.removeItem(it.text_item())
                    except Exception:
                        pass
                    try:
                        scene.removeItem(it)
                    except Exception:
                        pass

        self.status.setText("Deleted keypoint. Click Save label to persist.")
        self._update_present_keypoints_list()

    def delete_selected_bboxes(self) -> bool:
        """Delete selected bbox(es) (keep keypoints). Returns True if any bbox was deleted."""
        if self._det_deleted is None:
            return False
        scene = self.viewer.scene()
        if scene is None:
            return False
        selected = list(scene.selectedItems())
        if not selected:
            return False

        deleted_any = False

        def _delete_bbox_item(b: ResizableBBox) -> None:
            nonlocal deleted_any
            try:
                b._on_deleted()  # uses closure to mark deleted and re-render
                deleted_any = True
            except Exception:
                pass

        for it in selected:
            if isinstance(it, ResizableBBox):
                _delete_bbox_item(it)
            elif isinstance(it, _BBoxHandle):
                parent = it.parentItem()
                if isinstance(parent, ResizableBBox):
                    _delete_bbox_item(parent)

        return deleted_any


def main(argv: list[str] | None = None) -> int:
    _ = argv  # reserved for future CLI args
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1200, 900)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

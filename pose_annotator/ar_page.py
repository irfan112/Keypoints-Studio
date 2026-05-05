"""Action recognition / detection-style bbox annotation (labelImg-like workflow)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QFileDialog,
    QGridLayout,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pose_annotator.auto_annotate import IMAGE_EXTS


def _package_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ar_classes_path() -> Path:
    return _package_root() / "data" / "ar_classes.txt"


def _to_qpixmap(rgb: np.ndarray) -> QPixmap:
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8, copy=False)
    h, w, _ = rgb.shape
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


@dataclass
class ARBox:
    cls_id: int
    rect: QRectF  # scene coords (pixels)


class ARRectItem(QGraphicsRectItem):
    """Selectable movable bbox with class label."""

    def __init__(
        self,
        rect: QRectF,
        cls_id: int,
        cls_name: str,
        on_changed,
        on_request_delete,
    ) -> None:
        super().__init__(rect)
        self.cls_id = int(cls_id)
        self._on_changed = on_changed
        self._on_request_delete = on_request_delete
        self.setPen(QPen(QColor(0, 180, 255), 2))
        self.setBrush(QBrush(QColor(0, 160, 255, 40)))
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

        self._label = QGraphicsSimpleTextItem(f"{cls_id}: {cls_name}")
        self._label.setBrush(QBrush(QColor(255, 255, 255)))
        self._label.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self._label.setParentItem(self)
        self._label.setPos(rect.left() + 2, rect.top() + 2)

    def set_class(self, cls_id: int, cls_name: str) -> None:
        self.cls_id = int(cls_id)
        self._label.setText(f"{cls_id}: {cls_name}")

    def itemChange(self, change, value):  # type: ignore[override]
        if change == QGraphicsRectItem.GraphicsItemChange.ItemPositionHasChanged:
            self._on_changed(self)
        return super().itemChange(change, value)

    def contextMenuEvent(self, event) -> None:  # type: ignore[override]
        from PySide6.QtWidgets import QMenu

        menu = QMenu()
        act_del = menu.addAction("Delete box")
        chosen = menu.exec(event.screenPos())
        if chosen == act_del:
            self._on_request_delete(self)
            event.accept()
            return
        super().contextMenuEvent(event)


class ARViewer(QGraphicsView):
    """Zoom (wheel), pan (middle drag), draw box (W + LMB drag)."""

    def __init__(self, parent_page: "ActionRecognitionPage") -> None:
        super().__init__()
        self._page = parent_page
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix = QGraphicsPixmapItem()
        self._scene.addItem(self._pix)
        self._pix.setZValue(0)

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setStyleSheet("QGraphicsView { background: #1a1a1a; }")
        self.setMouseTracking(True)

        self._draw_mode = False
        self._rubber_start: Optional[QPointF] = None
        self._rubber: Optional[QGraphicsRectItem] = None
        self._middle_panning = False
        self._middle_last = QPointF()

    def set_placeholder(self, text: str) -> None:
        self._scene.clear()
        self._pix = QGraphicsPixmapItem()
        self._scene.addItem(self._pix)
        self._scene.addText(text).setDefaultTextColor(Qt.GlobalColor.lightGray)
        self.resetTransform()

    def load_image(self, path: Path) -> tuple[int, int] | None:
        try:
            import cv2  # type: ignore

            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            rgb = bgr[..., ::-1].copy()
            h, w = rgb.shape[0], rgb.shape[1]
            pix = _to_qpixmap(rgb)
            self._scene.clear()
            self._pix = QGraphicsPixmapItem(pix)
            self._pix.setZValue(0)
            self._scene.addItem(self._pix)
            self._scene.setSceneRect(pix.rect())
            self.resetTransform()
            self.fitInView(self._pix, Qt.AspectRatioMode.KeepAspectRatio)
            return w, h
        except Exception:
            return None

    def clear_boxes(self) -> None:
        for it in list(self._scene.items()):
            if it is self._pix:
                continue
            if isinstance(it, QGraphicsRectItem) and not isinstance(it, ARRectItem):
                self._scene.removeItem(it)
            elif isinstance(it, ARRectItem):
                self._scene.removeItem(it)

    def add_rect_item(self, item: ARRectItem) -> None:
        item.setZValue(10)
        self._scene.addItem(item)

    def set_draw_mode(self, enabled: bool) -> None:
        self._draw_mode = bool(enabled)
        self._rubber_start = None
        if self._rubber is not None:
            try:
                self._scene.removeItem(self._rubber)
            except Exception:
                pass
            self._rubber = None
        self.setDragMode(
            QGraphicsView.DragMode.NoDrag if enabled else QGraphicsView.DragMode.ScrollHandDrag
        )
        self.setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor
        )

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        delta = event.angleDelta().y()
        if delta == 0:
            return super().wheelEvent(event)
        factor = 1.25 if delta > 0 else 0.8
        self.scale(factor, factor)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.MiddleButton:
            self._middle_panning = True
            self._middle_last = QPointF(event.pos())
            event.accept()
            return
        if (
            self._draw_mode
            and event.button() == Qt.MouseButton.LeftButton
            and self._pix.pixmap() is not None
            and not self._pix.pixmap().isNull()
        ):
            sp = self.mapToScene(event.pos())
            self._rubber_start = QPointF(sp)
            self._rubber = QGraphicsRectItem(QRectF(sp, sp))
            self._rubber.setPen(QPen(QColor(255, 220, 60), 1, Qt.PenStyle.DashLine))
            self._rubber.setBrush(QBrush(QColor(255, 220, 60, 50)))
            self._rubber.setZValue(5)
            self._scene.addItem(self._rubber)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._middle_panning:
            delta = QPointF(event.pos()) - self._middle_last
            self._middle_last = QPointF(event.pos())
            self.horizontalScrollBar().setValue(
                int(self.horizontalScrollBar().value() - delta.x())
            )
            self.verticalScrollBar().setValue(
                int(self.verticalScrollBar().value() - delta.y())
            )
            event.accept()
            return
        if self._draw_mode and self._rubber_start is not None and self._rubber is not None:
            cur = self.mapToScene(event.pos())
            r = QRectF(self._rubber_start, cur).normalized()
            self._rubber.setRect(r)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.MiddleButton:
            self._middle_panning = False
            event.accept()
            return super().mouseReleaseEvent(event)
        if (
            self._draw_mode
            and event.button() == Qt.MouseButton.LeftButton
            and self._rubber is not None
            and self._rubber_start is not None
        ):
            rect = self._rubber.rect().normalized()
            try:
                self._scene.removeItem(self._rubber)
            except Exception:
                pass
            self._rubber = None
            self._rubber_start = None
            if rect.width() >= 4 and rect.height() >= 4:
                self._page.on_box_drawn(rect)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class ActionRecognitionPage(QWidget):
    """labelImg-like rectangle + class annotation for images."""

    def __init__(self, on_switch_pose) -> None:
        super().__init__()
        self._on_switch_pose = on_switch_pose
        self._images_dir: Optional[Path] = None
        self._images: list[Path] = []
        self._index = 0
        self._im_wh: tuple[int, int] = (0, 0)
        self._items: list[ARRectItem] = []

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Left
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)

        proj = QGroupBox("Project")
        pg = QGridLayout(proj)
        btn_pose = QPushButton("Switch Pose")
        btn_pose.clicked.connect(lambda: self._on_switch_pose())
        pg.addWidget(btn_pose, 0, 0, 1, 3)

        btn_open = QPushButton("Open Dir…")
        btn_open.clicked.connect(self._open_dir)
        pg.addWidget(QLabel("Image folder"), 1, 0)
        self.dir_label = QLabel("(none)")
        self.dir_label.setWordWrap(True)
        pg.addWidget(self.dir_label, 1, 1)
        pg.addWidget(btn_open, 1, 2)

        self.lbl_counts = QLabel("Images: 0/0")
        self.lbl_counts.setStyleSheet("QLabel { color: #1b8f2e; font-weight: 600; }")
        pg.addWidget(self.lbl_counts, 2, 0, 1, 3)

        classes_box = QGroupBox("Classes (click to select)")
        cv = QVBoxLayout(classes_box)
        self.class_list = QListWidget()
        self.class_list.setMinimumHeight(180)
        cv.addWidget(self.class_list)

        lv.addWidget(proj)
        lv.addWidget(classes_box)
        lv.addStretch(1)

        # Right
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)

        preview = QGroupBox("Preview")
        pl = QVBoxLayout(preview)
        self.hdr = QLabel("Current: (none)")
        self.hdr.setStyleSheet("QLabel { color: #1b8f2e; font-weight: 600; }")
        self.hdr.setWordWrap(True)
        pl.addWidget(self.hdr)

        self.viewer = ARViewer(self)
        self.viewer.setMinimumSize(640, 480)
        self.viewer.set_placeholder("Open an image folder to begin.")
        pl.addWidget(self.viewer)

        tools = QGroupBox("Tools")
        tl = QVBoxLayout(tools)
        row = QHBoxLayout()
        self.btn_prev = QPushButton("Previous")
        self.btn_next = QPushButton("Next")
        self.btn_save = QPushButton("Save")
        self.btn_draw = QPushButton("Create Box (W)")
        self.btn_draw.setCheckable(True)
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next.clicked.connect(self.next_image)
        self.btn_save.clicked.connect(self.save_current)
        self.btn_draw.toggled.connect(self._toggle_draw)
        row.addWidget(self.btn_prev)
        row.addWidget(self.btn_next)
        row.addWidget(self.btn_save)
        tl.addLayout(row)
        tl.addWidget(self.btn_draw)
        hint = QLabel(
            "Hotkeys (focus preview): W toggle draw box • A/D prev/next • Del delete • Ctrl+S save • Wheel zoom • Middle-drag pan"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        tl.addWidget(hint)

        rs = QSplitter(Qt.Orientation.Horizontal)
        rs.addWidget(preview)
        rs.addWidget(tools)
        rs.setStretchFactor(0, 19)
        rs.setStretchFactor(1, 1)
        tool_panel = tools
        tool_panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        rv.addWidget(rs)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 9)

        root.addWidget(splitter)

        self._ensure_classes_file()
        self._load_class_list()

        # Shortcuts on viewer
        QShortcut(QKeySequence("W"), self.viewer, context=Qt.ShortcutContext.WidgetShortcut).activated.connect(
            lambda: self.btn_draw.toggle()
        )
        QShortcut(QKeySequence("A"), self.viewer, context=Qt.ShortcutContext.WidgetShortcut).activated.connect(
            self.prev_image
        )
        QShortcut(QKeySequence("D"), self.viewer, context=Qt.ShortcutContext.WidgetShortcut).activated.connect(
            self.next_image
        )
        QShortcut(QKeySequence.StandardKey.Save, self.viewer, context=Qt.ShortcutContext.WidgetShortcut).activated.connect(
            self.save_current
        )
        QShortcut(QKeySequence.StandardKey.Delete, self.viewer, context=Qt.ShortcutContext.WidgetShortcut).activated.connect(
            self.delete_selected
        )

    def _ensure_classes_file(self) -> None:
        p = _ar_classes_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text(
                "# One class per line (index = line order, starting at 0)\n"
                "person\n"
                "wave\n"
                "sit\n"
                "stand\n",
                encoding="utf-8",
            )

    def _load_class_list(self) -> None:
        self.class_list.clear()
        path = _ar_classes_path()
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            self.class_list.addItem(s)
        if self.class_list.count() == 0:
            self.class_list.addItem("object")

    def selected_class_id(self) -> tuple[int, str]:
        row = self.class_list.currentRow()
        if row < 0:
            row = 0
            if self.class_list.count() > 0:
                self.class_list.setCurrentRow(0)
        name = self.class_list.item(row).text() if self.class_list.count() else "object"
        return row, name

    def _open_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Open image directory")
        if not d:
            return
        folder = Path(d)
        imgs = sorted(
            [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        )
        self._images_dir = folder
        self._images = imgs
        self._index = 0
        self.dir_label.setText(str(folder))
        self.lbl_counts.setText(f"Images: {len(imgs)} total")
        self._update_nav()
        self.load_current_image()

    def _toggle_draw(self, on: bool) -> None:
        self.viewer.set_draw_mode(on)

    def _update_nav(self) -> None:
        n = len(self._images)
        self.btn_prev.setEnabled(n > 0 and self._index > 0)
        self.btn_next.setEnabled(n > 0 and self._index < n - 1)
        if n == 0:
            self.lbl_counts.setText("Images: 0/0")

    def load_current_image(self) -> None:
        self.viewer.clear_boxes()
        self._items.clear()
        if not self._images:
            self.hdr.setText("Current: (none)")
            self.viewer.set_placeholder("No images in folder.")
            return
        path = self._images[self._index]
        wh = self.viewer.load_image(path)
        if wh is None:
            QMessageBox.warning(self, "Open failed", f"Could not open:\n{path}")
            return
        self._im_wh = wh
        rel = path.name
        self.hdr.setText(f"Current: {rel} ({self._index + 1}/{len(self._images)})")
        self.lbl_counts.setText(f"Images: {self._index + 1}/{len(self._images)}")
        self._load_labels(path)
        self.viewer.setFocus(Qt.FocusReason.OtherFocusReason)

    def _load_labels(self, img_path: Path) -> None:
        txt = img_path.with_suffix(".txt")
        if not txt.exists():
            return
        W, H = self._im_wh
        if W <= 0 or H <= 0:
            return
        try:
            lines = [ln.strip() for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except Exception:
            return
        names = self._class_names_ordered()
        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                continue
            # Accept pose lines (longer than 5) — use first 5 as detection box
            cid = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])
            x1 = (xc - bw / 2.0) * W
            y1 = (yc - bh / 2.0) * H
            x2 = (xc + bw / 2.0) * W
            y2 = (yc + bh / 2.0) * H
            nm = names[cid] if 0 <= cid < len(names) else str(cid)
            self._add_item(cid, nm, QRectF(x1, y1, x2 - x1, y2 - y1))

    def _class_names_ordered(self) -> list[str]:
        out: list[str] = []
        for i in range(self.class_list.count()):
            out.append(self.class_list.item(i).text())
        return out if out else ["object"]

    def _add_item(self, cls_id: int, cls_name: str, rect: QRectF) -> ARRectItem:
        def _chg(it: ARRectItem) -> None:
            pass

        def _del(it: ARRectItem) -> None:
            try:
                self.viewer.scene().removeItem(it._label)
            except Exception:
                pass
            try:
                self.viewer.scene().removeItem(it)
            except Exception:
                pass
            if it in self._items:
                self._items.remove(it)

        item = ARRectItem(rect, cls_id, cls_name, _chg, _del)
        self.viewer.add_rect_item(item)
        self._items.append(item)
        return item

    def on_box_drawn(self, rect: QRectF) -> None:
        cid, nm = self.selected_class_id()
        self._add_item(cid, nm, rect)

    def prev_image(self) -> None:
        if self._index <= 0:
            return
        self._index -= 1
        self._update_nav()
        self.load_current_image()

    def next_image(self) -> None:
        if self._index >= len(self._images) - 1:
            return
        self._index += 1
        self._update_nav()
        self.load_current_image()

    def delete_selected(self) -> None:
        sc = self.viewer.scene()
        if sc is None:
            return
        for it in list(sc.selectedItems()):
            if isinstance(it, ARRectItem):
                try:
                    sc.removeItem(it._label)
                except Exception:
                    pass
                sc.removeItem(it)
                if it in self._items:
                    self._items.remove(it)

    def save_current(self) -> None:
        if not self._images:
            return
        path = self._images[self._index]
        W, H = self._im_wh
        if W <= 0 or H <= 0:
            return
        lines: list[str] = []
        for it in list(self._items):
            r = it.rect()
            pos = it.pos()
            x1 = r.left() + pos.x()
            y1 = r.top() + pos.y()
            x2 = r.right() + pos.x()
            y2 = r.bottom() + pos.y()
            bw = max((x2 - x1) / W, 1e-9)
            bh = max((y2 - y1) / H, 1e-9)
            xc = ((x1 + x2) / 2.0) / W
            yc = ((y1 + y2) / 2.0) / H
            lines.append(f"{int(it.cls_id)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        out = path.with_suffix(".txt")
        out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        # classes.txt next to images (labelImg-style)
        names = self._class_names_ordered()
        cls_file = self._images_dir / "classes.txt" if self._images_dir else path.parent / "classes.txt"
        try:
            cls_file.write_text("\n".join(names) + "\n", encoding="utf-8")
        except Exception:
            pass

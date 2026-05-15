"""Action recognition / detection-style bbox annotation (labelImg-like workflow).

Features
--------
  • 4-Dot Corner Resizing: Double-click or Right-click → Edit Box.
  • Premium Sidebar: Auto-Save, Navigation, and Tools.
  • Full YOLO Export: Saves .txt labels and classes.txt automatically.
  • Clipboard: Ctrl+V (Single), Ctrl+Shift+V (All) Geometry Paste.
  • Shortcuts: W (Draw), A/D (Prev/Next), Ctrl+S (Save), Del (Delete), Ctrl+F (Fit).
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

from PySide6.QtCore import QPointF, QRectF, Qt, QEvent, QTimer, QUrl, QSize
from PySide6.QtGui import (
    QBrush, QColor, QFont, QImage, QKeySequence, QPen, QPixmap, QShortcut, QCursor, QDesktopServices
)
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QDialog, QDialogButtonBox, QFrame,
    QGraphicsItem, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene,
    QGraphicsSimpleTextItem, QGraphicsView, QGroupBox, QHBoxLayout,
    QLabel, QFileDialog, QGridLayout, QListWidget, QListWidgetItem,
    QMenu, QMessageBox, QPushButton, QSplitter, QStatusBar,
    QVBoxLayout, QWidget, QGraphicsEllipseItem, QSizePolicy, QApplication,
    QStyle, QToolButton
)

from pose_annotator.auto_annotate import IMAGE_EXTS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _annotator_package_dir() -> Path:
    return Path(__file__).resolve().parent

def _ar_classes_path() -> Path:
    return _annotator_package_dir() / "data" / "ar_classes.txt"

def _to_qpixmap(rgb: np.ndarray) -> QPixmap:
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8, copy=False)
    h, w, _ = rgb.shape
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())

# ---------------------------------------------------------------------------
# Resize Handle (Dots)
# ---------------------------------------------------------------------------

class ResizeHandle(QGraphicsEllipseItem):
    _SIZE = 14 
    def __init__(self, h_frac, v_frac, viewer):
        super().__init__(QRectF(-self._SIZE/2, -self._SIZE/2, self._SIZE, self._SIZE))
        self.h_frac, self.v_frac = h_frac, v_frac
        self._viewer = viewer
        self.setPen(QPen(Qt.GlobalColor.white, 2))
        self.setBrush(QBrush(QColor(0, 255, 127)))
        self.setZValue(1000)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setAcceptHoverEvents(True)
        self._dragging = False

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._start_pos = event.scenePos()
            self._start_rect = self._viewer._resize_item.scene_rect()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._dragging:
            delta = event.scenePos() - self._start_pos
            r = self._start_rect
            x1, y1, x2, y2 = r.left(), r.top(), r.right(), r.bottom()
            if self.h_frac == 0.0: x1 += delta.x()
            elif self.h_frac == 1.0: x2 += delta.x()
            if self.v_frac == 0.0: y1 += delta.y()
            elif self.v_frac == 1.0: y2 += delta.y()
            it = self._viewer._resize_item
            it.setPos(0, 0); it.setRect(QRectF(x1, y1, max(10, x2-x1), max(10, y2-y1)))
            self._viewer._update_handle_positions()
            event.accept()

    def mouseReleaseEvent(self, event):
        self._dragging = False
        event.accept()

# ---------------------------------------------------------------------------
# Bounding Box Item
# ---------------------------------------------------------------------------

class ARRectItem(QGraphicsRectItem):
    def __init__(self, rect, cls_id, cls_name, page):
        super().__init__(rect)
        self.cls_id, self._page = cls_id, page
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self._apply_style()
        self._label = QGraphicsSimpleTextItem(f"{cls_id}: {cls_name}", self)
        self._label.setBrush(Qt.GlobalColor.white)
        self._label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self._label.setPos(rect.topLeft() + QPointF(2, 2))

    def _apply_style(self, hovered=False):
        colors = [
            QColor(255, 56, 56), QColor(255, 157, 151), QColor(255, 112, 31),
            QColor(255, 178, 29), QColor(207, 210, 49), QColor(72, 249, 10),
            QColor(146, 204, 23), QColor(61, 219, 134), QColor(26, 147, 52),
            QColor(0, 212, 187), QColor(44, 153, 168), QColor(0, 194, 255),
            QColor(52, 69, 147), QColor(100, 115, 255), QColor(0, 24, 236),
            QColor(132, 56, 255), QColor(82, 0, 133), QColor(203, 56, 255),
            QColor(255, 149, 200), QColor(255, 55, 199)
        ]
        col = colors[self.cls_id % len(colors)]
        is_selected = self.isSelected()
        
        # Highlighting logic: Darker/Thicker if selected or hovered
        pen_width = 3 if (is_selected or hovered) else 1
        pen = QPen(col, pen_width)
        if is_selected:
            pen.setStyle(Qt.PenStyle.SolidLine)
            alpha = 120
        elif hovered:
            alpha = 100
        else:
            alpha = 40 # Light for others
            
        self.setPen(pen)
        self.setBrush(QBrush(QColor(col.red(), col.green(), col.blue(), alpha)))

    def hoverEnterEvent(self, event):
        self._apply_style(hovered=True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._apply_style(hovered=False)
        super().hoverLeaveEvent(event)

    def scene_rect(self): return self.rect().translated(self.pos())

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._apply_style()
        elif change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            if self._page.viewer._resize_item is self: self._page.viewer._update_handle_positions()
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._page.viewer.show_resize_handles(self); event.accept()

    def contextMenuEvent(self, event):
        menu = QMenu(); edit_act = menu.addAction("✏  Edit Box")
        sub = menu.addMenu("Set Class"); names = self._page._class_names_ordered()
        for cid, nm in enumerate(names):
            act = sub.addAction(f"{cid}: {nm}"); act.setData(cid)
        menu.addSeparator()
        del_act = menu.addAction("🗑  Delete Box"); chosen = menu.exec(event.screenPos())
        if chosen == edit_act: self._page.viewer.show_resize_handles(self)
        elif chosen == del_act: self._page._remove_rect_item(self)
        elif chosen and chosen.parent() == sub:
            cid = chosen.data(); self.set_class(cid, names[cid])
        event.accept()

    def set_class(self, cid, name):
        self.cls_id = cid; self._label.setText(f"{cid}: {name}")
        self._apply_style()

# ---------------------------------------------------------------------------
# Graphics View (Viewer)
# ---------------------------------------------------------------------------

class ARViewer(QGraphicsView):
    def __init__(self, page):
        super().__init__()
        self._page, self._scene = page, QGraphicsScene(self)
        self.setScene(self._scene); self._pix = QGraphicsPixmapItem()
        self._scene.addItem(self._pix)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setStyleSheet("background: #000; border: none;")
        self._resize_item, self._handles, self._draw_mode, self._rubber = None, [], False, None
        self.setMouseTracking(True); self._mouse_pos = None

    def load_image(self, path):
        pix = QPixmap(str(path))
        if pix.isNull():
            try:
                import cv2
                img = cv2.imread(str(path))
                if img is not None: pix = _to_qpixmap(img[..., ::-1])
            except: pass
        if not pix.isNull():
            self._scene.clear(); self._pix = QGraphicsPixmapItem(pix); self._pix.setZValue(0)
            self._scene.addItem(self._pix); self._scene.setSceneRect(pix.rect())
            self.resetTransform(); self.fitInView(self._pix, Qt.AspectRatioMode.KeepAspectRatio)
            return pix.width(), pix.height()
        return None

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            f = 1.25 if event.angleDelta().y() > 0 else 0.8; self.scale(f, f)
            event.accept()
        else:
            super().wheelEvent(event)

    def show_resize_handles(self, item):
        self.hide_resize_handles(); self._resize_item = item
        for h, v in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]:
            handle = ResizeHandle(h, v, self); self._scene.addItem(handle); self._handles.append(handle)
        self._update_handle_positions()

    def hide_resize_handles(self):
        for h in self._handles:
            try: self._scene.removeItem(h)
            except: pass
        self._handles, self._resize_item = [], None

    def _update_handle_positions(self):
        if self._resize_item:
            r = self._resize_item.scene_rect()
            for h in self._handles: h.setPos(r.left() + h.h_frac * r.width(), r.top() + h.v_frac * r.height())

    def set_draw_mode(self, on):
        self._draw_mode = on; self.setDragMode(QGraphicsView.DragMode.NoDrag if on else QGraphicsView.DragMode.ScrollHandDrag)
        self.setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)
        if not on: self._mouse_pos = None; self.scene().invalidate(self.sceneRect(), QGraphicsScene.SceneLayer.ForegroundLayer)

    def leaveEvent(self, event):
        self._mouse_pos = None; self.scene().invalidate(self.sceneRect(), QGraphicsScene.SceneLayer.ForegroundLayer)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        
        # In Draw Mode, we only select if clicking in the "center" of a box
        if self._draw_mode and event.button() == Qt.MouseButton.LeftButton:
            target_item = None
            for item in self.items(event.pos()):
                if isinstance(item, ARRectItem):
                    target_item = item; break
                if isinstance(item, ResizeHandle): # Always allow handles
                    super().mousePressEvent(event); return

            is_center_click = False
            if target_item:
                r = target_item.scene_rect()
                # Center zone: 40% of the box
                center_q = QRectF(r.x() + r.width()*0.3, r.y() + r.height()*0.3, r.width()*0.4, r.height()*0.4)
                if center_q.contains(scene_pos):
                    is_center_click = True
            
            if not is_center_click:
                self._start_draw = scene_pos
                self._rubber = QGraphicsRectItem(QRectF(scene_pos, scene_pos))
                self._rubber.setPen(QPen(Qt.GlobalColor.yellow, 1, Qt.PenStyle.DashLine))
                self._scene.addItem(self._rubber)
                event.accept(); return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self._mouse_pos = self.mapToScene(event.pos())
        if self._draw_mode:
            self.scene().invalidate(self.sceneRect(), QGraphicsScene.SceneLayer.ForegroundLayer)
        if self._draw_mode and self._rubber:
            self._rubber.setRect(QRectF(self._start_draw, self._mouse_pos).normalized())
        else: super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._draw_mode and self._rubber:
            rect = self._rubber.rect().normalized(); self._scene.removeItem(self._rubber); self._rubber = None
            if rect.width() > 5 and rect.height() > 5: self._page.on_box_drawn(rect)
        else: super().mouseReleaseEvent(event)

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        if self._draw_mode and self._mouse_pos:
            # Draw crosshair spanning the visible area
            painter.setPen(QPen(QColor(255, 255, 255, 150), 0)) # White-ish semi-transparent
            x, y = self._mouse_pos.x(), self._mouse_pos.y()
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            
            # Draw a black 1px line for contrast
            painter.setPen(QPen(Qt.GlobalColor.black, 0))
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

# ---------------------------------------------------------------------------
# Class Editor Dialog
# ---------------------------------------------------------------------------

class ClassEditorDialog(QDialog):
    def __init__(self, names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Classes")
        self.setMinimumSize(400, 500)
        layout = QVBoxLayout(self); layout.addWidget(QLabel("<b>Double-click</b> to rename. <b>Drag</b> to reorder."))
        self.lw = QListWidget(); self.lw.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        for nm in names:
            it = QListWidgetItem(nm); it.setFlags(it.flags() | Qt.ItemFlag.ItemIsEditable); self.lw.addItem(it)
        layout.addWidget(self.lw); btns = QHBoxLayout()
        add, rem = QPushButton("Add New"), QPushButton("Delete")
        for b in [add, rem]: btns.addWidget(b)
        layout.addLayout(btns); add.clicked.connect(self._add); rem.clicked.connect(self._rem)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject); layout.addWidget(bb)
    def _add(self):
        it = QListWidgetItem("new_class"); it.setFlags(it.flags() | Qt.ItemFlag.ItemIsEditable); self.lw.addItem(it); self.lw.editItem(it)
    def _rem(self):
        if self.lw.currentRow() >= 0: self.lw.takeItem(self.lw.currentRow())
    def get_names(self):
        return [self.lw.item(i).text().strip() for i in range(self.lw.count()) if self.lw.item(i).text().strip()]

# ---------------------------------------------------------------------------
# Main Page
# ---------------------------------------------------------------------------

class ActionRecognitionPage(QWidget):
    def __init__(self, on_switch_pose):
        super().__init__()
        self._on_switch_pose, self._images, self._items, self._index, self._im_wh = on_switch_pose, [], [], 0, (0, 0)
        self._all_bboxes_clipboard = []
        self._selected_bbox_clipboard = None
        self._root_dir = None
        self._skip_delete_confirm = False
        self.setStyleSheet("""
            QWidget { background-color: #1e1e1e; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
            QGroupBox { border: 1px solid #333; border-radius: 8px; margin-top: 15px; font-weight: bold; padding: 15px; }
            QPushButton { background-color: #3a3a3a; border: 1px solid #444; border-radius: 5px; padding: 8px 16px; font-weight: 500; }
            QPushButton:hover { background-color: #4a4a4a; border-color: #666; }
            QPushButton[checkable="true"]:checked { background-color: #0078d4; border-color: #0086f0; color: white; }
            QToolButton { background-color: transparent; border: 1px solid transparent; border-radius: 5px; padding: 5px; font-weight: 500; }
            QToolButton:hover { background-color: #3a3a3a; border-color: #555; }
            QToolButton[checkable="true"]:checked { background-color: #0078d4; border-color: #0086f0; color: white; }
            QCheckBox { spacing: 8px; }
            QListWidget { background-color: #252526; border: 1px solid #333; border-radius: 6px; }
        """)
        
        main_layout = QHBoxLayout(self); splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # ---------------------------------------------------------
        # LEFT SIDEBAR: Buttons & Actions
        # ---------------------------------------------------------
        left = QWidget(); lv = QVBoxLayout(left)
        style = QApplication.style()

        def create_tool_btn(text, icon_sp, checkable=False):
            btn = QToolButton()
            btn.setText(text)
            btn.setIcon(style.standardIcon(icon_sp))
            btn.setIconSize(QSize(28, 28))
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            if checkable: btn.setCheckable(True)
            return btn
        
        # Workspace Group
        ws_grp = QGroupBox("Workspace"); ws_v = QGridLayout(ws_grp)
        btn_pose = create_tool_btn("Pose Mode", QStyle.StandardPixmap.SP_BrowserReload)
        btn_pose.clicked.connect(lambda: self._on_switch_pose())
        btn_dir = create_tool_btn("Open Dir", QStyle.StandardPixmap.SP_DirIcon)
        btn_dir.clicked.connect(self._open_dir)
        btn_root_dir = create_tool_btn("Open Root", QStyle.StandardPixmap.SP_DirOpenIcon)
        btn_root_dir.clicked.connect(self._open_root_dir)
        ws_v.addWidget(btn_pose, 0, 0, 1, 2); ws_v.addWidget(btn_dir, 1, 0); ws_v.addWidget(btn_root_dir, 1, 1)
        lv.addWidget(ws_grp)
        
        # Tools Group
        tool_grp = QGroupBox("Tools"); tv = QHBoxLayout(tool_grp)
        self.btn_draw = create_tool_btn("Create Box", QStyle.StandardPixmap.SP_TitleBarNormalButton, checkable=True)
        save_btn = create_tool_btn("Save", QStyle.StandardPixmap.SP_DialogSaveButton)
        save_btn.clicked.connect(self.save_current)
        tv.addWidget(self.btn_draw); tv.addWidget(save_btn)
        lv.addWidget(tool_grp)

        # Properties Group
        prop_grp = QGroupBox("Properties"); pv = QVBoxLayout(prop_grp)
        self.chk_autosave = QCheckBox("Auto Saving"); self.chk_autosave.setChecked(True); pv.addWidget(self.chk_autosave)
        lv.addWidget(prop_grp)

        # Navigation Group
        nav_grp = QGroupBox("Navigation"); nv = QHBoxLayout(nav_grp)
        self.btn_prev = create_tool_btn("Previous", QStyle.StandardPixmap.SP_ArrowLeft)
        self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_next = create_tool_btn("Next", QStyle.StandardPixmap.SP_ArrowRight)
        self.btn_next.clicked.connect(lambda: self._step(1))
        nv.addWidget(self.btn_prev); nv.addWidget(self.btn_next)
        lv.addWidget(nav_grp)
        
        lv.addStretch()
        
        # ---------------------------------------------------------
        # CENTER: Viewer
        # ---------------------------------------------------------
        center = QWidget(); cv_layout = QVBoxLayout(center); self.hdr = QLabel("Current: (none)")
        self.hdr.setStyleSheet("font-weight: bold; color: #4CAF50;"); cv_layout.addWidget(self.hdr)
        self.viewer = ARViewer(self); cv_layout.addWidget(self.viewer)
        self._status_bar = QStatusBar(); cv_layout.addWidget(self._status_bar)
        
        # ---------------------------------------------------------
        # RIGHT SIDEBAR: Information & Lists
        # ---------------------------------------------------------
        right = QWidget(); rv = QVBoxLayout(right)

        # Status Group
        info_grp = QGroupBox("Status"); iv = QVBoxLayout(info_grp)
        self.lbl_counts = QLabel("Images: 0/0")
        iv.addWidget(self.lbl_counts)
        rv.addWidget(info_grp)

        # Classes Group
        cls_grp = QGroupBox("Classes"); cv = QVBoxLayout(cls_grp)
        btn_edit_cls = QToolButton()
        btn_edit_cls.setText("Edit Classes")
        btn_edit_cls.setIcon(QApplication.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView))
        btn_edit_cls.setIconSize(QSize(28, 28))
        btn_edit_cls.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        btn_edit_cls.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        btn_edit_cls.clicked.connect(self._open_class_editor)
        cv.addWidget(btn_edit_cls)
        self.class_list = QListWidget(); cv.addWidget(self.class_list); rv.addWidget(cls_grp)

        # Files & Labels Group
        lbl_grp = QGroupBox("Files & Labels"); lv_txt = QVBoxLayout(lbl_grp)
        self.lbl_folder = QLabel("📁 Folder: (none)")
        self.lbl_folder.setStyleSheet("color: #FFD700; font-weight: bold;")
        self.lbl_folder.setCursor(Qt.CursorShape.PointingHandCursor)
        self.lbl_folder.mousePressEvent = self._on_folder_lbl_clicked
        lv_txt.addWidget(self.lbl_folder)
        
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self._on_file_selected_via_click)
        self.file_list.itemDoubleClicked.connect(self._on_file_double_clicked)
        lv_txt.addWidget(self.file_list)
        rv.addWidget(lbl_grp)

        # Shortcuts Group
        sc_grp = QGroupBox("Shortcuts"); sg = QGridLayout(sc_grp)
        self.lbl_sc_w = QLabel("W : Draw Box")
        self.lbl_sc_ad = QLabel("A / D : Prev/Next")
        self.lbl_sc_del = QLabel("Delete : Remove")
        self.lbl_sc_ctrl_e = QLabel("Ctrl+E : Edit Class")
        self.lbl_sc_ctrl_s = QLabel("Ctrl+S : Save")
        self.lbl_sc_ctrl_f = QLabel("Ctrl+F : Fit View")
        self.lbl_sc_ctrl_plus = QLabel("Ctrl + : Zoom In")
        self.lbl_sc_ctrl_minus = QLabel("Ctrl - : Zoom Out")
        self.lbl_sc_ctrl_d = QLabel("Ctrl+D : Dup Box")
        self.lbl_sc_ctrl_shift_d = QLabel("Ctrl+Sh+D : Del Img")
        self.lbl_sc_ctrl_v = QLabel("Ctrl+V : Paste 1")
        self.lbl_sc_ctrl_shift_v = QLabel("Ctrl+Sh+V : Paste All")
        labels = [self.lbl_sc_w, self.lbl_sc_ad, self.lbl_sc_del, self.lbl_sc_ctrl_e, self.lbl_sc_ctrl_s, 
                  self.lbl_sc_ctrl_f, self.lbl_sc_ctrl_plus, self.lbl_sc_ctrl_minus, 
                  self.lbl_sc_ctrl_d, self.lbl_sc_ctrl_shift_d, self.lbl_sc_ctrl_v, self.lbl_sc_ctrl_shift_v]
        for i, lbl in enumerate(labels):
            sg.addWidget(lbl, i // 2, i % 2)
        rv.addWidget(sc_grp)
        
        # ---------------------------------------------------------
        # Wrap Up
        # ---------------------------------------------------------
        splitter.addWidget(left); splitter.addWidget(center); splitter.addWidget(right)
        splitter.setStretchFactor(1, 4); main_layout.addWidget(splitter)
        
        self.btn_draw.toggled.connect(self.viewer.set_draw_mode)
        self._load_classes()
        QShortcut(QKeySequence("W"), self, self._toggle_draw)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_current)
        QShortcut(QKeySequence("A"), self, lambda: self._step(-1))
        QShortcut(QKeySequence("D"), self, lambda: self._step(1))
        QShortcut(QKeySequence("Ctrl+="), self, self._zoom_in)
        QShortcut(QKeySequence("Ctrl++"), self, self._zoom_in)
        QShortcut(QKeySequence("Ctrl+-"), self, self._zoom_out)
        QShortcut(QKeySequence("Ctrl+F"), self, self._fit_view)
        QShortcut(QKeySequence("Ctrl+E"), self, self._edit_selected_class)
        QShortcut(QKeySequence.StandardKey.Delete, self, self.delete_selected)
        QShortcut(QKeySequence("Ctrl+D"), self, self.duplicate_selected)
        QShortcut(QKeySequence("Ctrl+Shift+D"), self, self.delete_current_image)
        QShortcut(QKeySequence("Ctrl+V"), self, self._paste_single_bbox)
        QShortcut(QKeySequence("Ctrl+Shift+V"), self, self._paste_all_bboxes)
        
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            modifiers = event.modifiers(); key = event.key()
            
            ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier) or key == Qt.Key.Key_Control
            shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier) or key == Qt.Key.Key_Shift
            
            for lbl in [self.lbl_sc_w, self.lbl_sc_ad, self.lbl_sc_del, self.lbl_sc_ctrl_e, self.lbl_sc_ctrl_s, 
                        self.lbl_sc_ctrl_f, self.lbl_sc_ctrl_plus, self.lbl_sc_ctrl_minus,
                        self.lbl_sc_ctrl_d, self.lbl_sc_ctrl_shift_d, self.lbl_sc_ctrl_v, self.lbl_sc_ctrl_shift_v]:
                if "yellow" not in lbl.styleSheet():
                    lbl.setStyleSheet("")

            if ctrl and shift:
                for lbl in [self.lbl_sc_ctrl_shift_d, self.lbl_sc_ctrl_shift_v]:
                    if "yellow" not in lbl.styleSheet(): lbl.setStyleSheet("color: #00BFFF;")
            elif ctrl:
                for lbl in [self.lbl_sc_ctrl_e, self.lbl_sc_ctrl_s, self.lbl_sc_ctrl_f, 
                            self.lbl_sc_ctrl_plus, self.lbl_sc_ctrl_minus,
                            self.lbl_sc_ctrl_d, self.lbl_sc_ctrl_shift_d, self.lbl_sc_ctrl_v, self.lbl_sc_ctrl_shift_v]:
                    if "yellow" not in lbl.styleSheet(): lbl.setStyleSheet("color: #00BFFF;")
            elif shift:
                for lbl in [self.lbl_sc_ctrl_shift_d, self.lbl_sc_ctrl_shift_v]:
                    if "yellow" not in lbl.styleSheet(): lbl.setStyleSheet("color: #00BFFF;")
        return super().eventFilter(obj, event)

    def _flash_sc(self, lbl):
        lbl.setStyleSheet("color: yellow; font-weight: bold;")
        QTimer.singleShot(300, lambda: lbl.setStyleSheet("color: #00BFFF;" if "Ctrl" in lbl.text() else ""))

    def _on_folder_lbl_clicked(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._images and 0 <= self._index < len(self._images):
                current_parent = self._images[self._index].parent
                if current_parent.exists():
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(current_parent)))

    def _on_file_double_clicked(self, item):
        text_path = item.text()
        if not self._images or self._index >= len(self._images): return
        current_parent = self._images[self._index].parent
        path_to_open = None
        for img in self._images:
            if img.parent == current_parent:
                if text_path == img.name:
                    path_to_open = img; break
                elif text_path == img.with_suffix(".txt").name:
                    path_to_open = img.with_suffix(".txt"); break
        if path_to_open and path_to_open.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path_to_open)))

    def _toggle_draw(self):
        self._flash_sc(self.lbl_sc_w)
        self.btn_draw.toggle()

    def _zoom_in(self):
        self._flash_sc(self.lbl_sc_ctrl_plus)
        self.viewer.scale(1.25, 1.25)

    def _zoom_out(self):
        self._flash_sc(self.lbl_sc_ctrl_minus)
        self.viewer.scale(0.8, 0.8)

    def _fit_view(self):
        self._flash_sc(self.lbl_sc_ctrl_f)
        self.viewer.fitInView(self.viewer._pix, Qt.AspectRatioMode.KeepAspectRatio)

    def _class_names_ordered(self): return [self.class_list.item(i).text() for i in range(self.class_list.count())]

    def _open_class_editor(self):
        old_names = self._class_names_ordered(); dlg = ClassEditorDialog(old_names, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_names = dlg.get_names(); _ar_classes_path().write_text("\n".join(new_names), encoding="utf-8")
            self.class_list.clear(); self.class_list.addItems(new_names)

    def _load_classes(self):
        p = _ar_classes_path(); p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists(): p.write_text("person\nobject\n", encoding="utf-8")
        self.class_list.clear(); self.class_list.addItems(p.read_text(encoding="utf-8").splitlines())

    def _open_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Open Folder")
        if d:
            self._root_dir = Path(d)
            self._images = sorted([p for p in self._root_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
            self._finish_load_dir()

    def _open_root_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Open Root Directory")
        if d:
            self._root_dir = Path(d)
            self._images = sorted([p for p in self._root_dir.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
            self._finish_load_dir()

    def _finish_load_dir(self):
        self.lbl_counts.setText(f"Images: 0/{len(self._images)}")
        if self._images: 
            self._index = 0
            self._load_image_by_index(0, force_list_update=True)
        else: 
            self._update_file_list()

    def _update_file_list(self):
        self.file_list.blockSignals(True)
        self.file_list.clear()
        if not self._images:
            self.lbl_folder.setText("📁 Folder: (none)")
            self.file_list.blockSignals(False); return
            
        current_parent = self._images[self._index].parent
        if self._root_dir and current_parent != self._root_dir:
            try:
                rel = current_parent.relative_to(self._root_dir).as_posix()
                self.lbl_folder.setText(f"📁 Folder: {rel}")
            except:
                self.lbl_folder.setText(f"📁 Folder: {current_parent.name}")
        else:
            self.lbl_folder.setText(f"📁 Folder: {current_parent.name}")

        for img in self._images:
            if img.parent == current_parent:
                self.file_list.addItem(img.name)
                txt = img.with_suffix(".txt")
                if txt.exists(): self.file_list.addItem(txt.name)
        
        self._highlight_current_in_list()
        self.file_list.blockSignals(False)

    def _highlight_current_in_list(self):
        self.file_list.blockSignals(True)
        if 0 <= self._index < len(self._images):
            img = self._images[self._index]
            img_name = img.name
            txt_name = img.with_suffix(".txt").name
            
            first_item = None
            for i in range(self.file_list.count()):
                item = self.file_list.item(i)
                item.setBackground(Qt.GlobalColor.transparent)
                item.setForeground(QColor("#e0e0e0"))
                if item.text() in (img_name, txt_name):
                    item.setBackground(QColor("#094771"))
                    item.setForeground(QColor("white"))
                    if not first_item: first_item = item
            if first_item:
                self.file_list.scrollToItem(first_item)
        self.file_list.blockSignals(False)

    def _on_file_selected_via_click(self, item):
        text_path = item.text()
        current_parent = self._images[self._index].parent
        idx = -1
        for i, img in enumerate(self._images):
            if img.parent == current_parent:
                if text_path in (img.name, img.with_suffix(".txt").name):
                    idx = i; break
        
        if idx != -1 and idx != self._index:
            if self.chk_autosave.isChecked() and self._images and self._index < len(self._images): 
                self.save_current()
            self._load_image_by_index(idx)

    def _load_image_by_index(self, idx, force_list_update=False):
        if not (0 <= idx < len(self._images)): return
        
        folder_changed = force_list_update or self.file_list.count() == 0
        if self._images and self._index < len(self._images):
            if self._images[self._index].parent != self._images[idx].parent:
                folder_changed = True
                
        self._capture_bbox_clipboard()
        self._index = idx
        self.viewer.hide_resize_handles()
        self._items = []
        path = self._images[idx]
        wh = self.viewer.load_image(path)
        if wh:
            self._im_wh = wh
            self.hdr.setText(f"Current: {path.name} ({idx+1}/{len(self._images)})")
            self.lbl_counts.setText(f"Images: {idx+1}/{len(self._images)}")
            self._load_labels(path)
            
            if folder_changed:
                self._update_file_list()
            else:
                self._highlight_current_in_list()

    def _load_labels(self, img_path: Path):
        txt = img_path.with_suffix(".txt"); names, W, H = self._class_names_ordered(), self._im_wh[0], self._im_wh[1]
        if not txt.exists() or W <= 0: return
        for ln in txt.read_text(encoding="utf-8").splitlines():
            p = ln.strip().split(); 
            if len(p) >= 5:
                cid = int(float(p[0])); xc, yc, bw, bh = map(float, p[1:5])
                self._add_item(cid, names[cid] if cid < len(names) else str(cid), QRectF((xc-bw/2)*W, (yc-bh/2)*H, bw*W, bh*H))

    def _add_item(self, cid, name, rect):
        it = ARRectItem(rect, cid, name, self)
        self.viewer._scene.addItem(it); it.setZValue(10); self._items.append(it); return it

    def _remove_rect_item(self, it):
        if it in self._items: self._items.remove(it)
        self.viewer.hide_resize_handles(); self.viewer._scene.removeItem(it)

    def _edit_selected_class(self):
        self._flash_sc(self.lbl_sc_ctrl_e)
        selected = [it for it in self.viewer.scene().selectedItems() if isinstance(it, ARRectItem)]
        if not selected: return
        menu = QMenu(self)
        names = self._class_names_ordered()
        for cid, nm in enumerate(names):
            act = menu.addAction(f"{cid}: {nm}"); act.setData(cid)
        chosen = menu.exec(QCursor.pos())
        if chosen:
            cid = chosen.data()
            for it in selected:
                it.set_class(cid, names[cid])

    def delete_selected(self):
        self._flash_sc(self.lbl_sc_del)
        for it in self.viewer.scene().selectedItems():
            if isinstance(it, ARRectItem): self._remove_rect_item(it)

    def duplicate_selected(self):
        self._flash_sc(self.lbl_sc_ctrl_d)
        names, W, H = self._class_names_ordered(), self._im_wh[0], self._im_wh[1]
        if W <= 0: return
        for it in self.viewer.scene().selectedItems():
            if isinstance(it, ARRectItem):
                r = it.scene_rect(); w_box, h_box = r.width(), r.height()
                new_rect = QRectF(W/2 - w_box/2, H/2 - h_box/2, w_box, h_box)
                nm = names[it.cls_id] if it.cls_id < len(names) else "object"
                new_it = self._add_item(it.cls_id, nm, new_rect); new_it.setSelected(True); it.setSelected(False)

    def delete_current_image(self):
        self._flash_sc(self.lbl_sc_ctrl_shift_d)
        if not self._images or self._index >= len(self._images): return
        path = self._images[self._index]; txt = path.with_suffix(".txt")
        
        if not self._skip_delete_confirm:
            msg = QMessageBox(self)
            msg.setWindowTitle("Delete Image")
            msg.setText(f"Permanently delete {path.name} and labels?")
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            cb = QCheckBox("Don't ask me again")
            msg.setCheckBox(cb)
            ans = msg.exec()
            if ans != QMessageBox.StandardButton.Yes: return
            if cb.isChecked(): self._skip_delete_confirm = True

        try:
            if path.exists(): path.unlink()
            if txt.exists(): txt.unlink()
            self._images.pop(self._index)
            if self._index >= len(self._images): self._index = len(self._images)-1
            if self._images:
                self._load_image_by_index(self._index)
            else:
                self.viewer._scene.clear()
                self.hdr.setText("Current: (none)")
                self._update_file_list()
        except Exception as e: QMessageBox.warning(self, "Error", str(e))

    def on_box_drawn(self, rect):
        r = self.class_list.currentRow(); cid = r if r >= 0 else 0
        names = self._class_names_ordered(); self._add_item(cid, names[cid] if cid < len(names) else "object", rect)

    def _capture_bbox_clipboard(self):
        self._all_bboxes_clipboard = []
        self._selected_bbox_clipboard = None
        W, H = self._im_wh
        if W > 0:
            names = self._class_names_ordered()
            for it in self._items:
                entry = {
                    'cid': it.cls_id, 'name': names[it.cls_id] if it.cls_id < len(names) else "object",
                    'geom': (it.scene_rect().center().x()/W, it.scene_rect().center().y()/H, it.scene_rect().width()/W, it.scene_rect().height()/H)
                }
                self._all_bboxes_clipboard.append(entry)
                if it.isSelected(): self._selected_bbox_clipboard = entry
            # Fallback: if none selected, last one is 'selected'
            if not self._selected_bbox_clipboard and self._items:
                it = self._items[-1]
                self._selected_bbox_clipboard = {
                    'cid': it.cls_id, 'name': names[it.cls_id] if it.cls_id < len(names) else "object",
                    'geom': (it.scene_rect().center().x()/W, it.scene_rect().center().y()/H, it.scene_rect().width()/W, it.scene_rect().height()/H)
                }

    def _paste_single_bbox(self):
        self._flash_sc(self.lbl_sc_ctrl_v)
        if self._selected_bbox_clipboard and self._im_wh[0] > 0:
            W, H = self._im_wh; entry = self._selected_bbox_clipboard; xc, yc, bw, bh = entry['geom']
            self._add_item(entry['cid'], entry['name'], QRectF((xc-bw/2)*W, (yc-bh/2)*H, bw*W, bh*H))

    def _paste_all_bboxes(self):
        self._flash_sc(self.lbl_sc_ctrl_shift_v)
        if self._all_bboxes_clipboard and self._im_wh[0] > 0:
            W, H = self._im_wh
            for entry in self._all_bboxes_clipboard:
                xc, yc, bw, bh = entry['geom']
                self._add_item(entry['cid'], entry['name'], QRectF((xc-bw/2)*W, (yc-bh/2)*H, bw*W, bh*H))

    def save_current(self):
        self._flash_sc(self.lbl_sc_ctrl_s)
        if not self._images or self._index >= len(self._images) or self._im_wh[0] <= 0: return
        W, H = self._im_wh; img_path = self._images[self._index]
        lines = [f"{it.cls_id} {it.scene_rect().center().x()/W:.6f} {it.scene_rect().center().y()/H:.6f} {it.scene_rect().width()/W:.6f} {it.scene_rect().height()/H:.6f}" for it in self._items]
        img_path.with_suffix(".txt").write_text("\n".join(lines), encoding="utf-8")
        try: (img_path.parent / "classes.txt").write_text("\n".join(self._class_names_ordered()), encoding="utf-8")
        except: pass
        self._status_bar.showMessage(f"Saved {img_path.name}", 1000)
        self._update_file_list()

    def _step(self, delta):
        self._flash_sc(self.lbl_sc_ad)
        if self._images:
            idx = self._index + delta
            if 0 <= idx < len(self._images):
                current_parent = self._images[self._index].parent
                next_parent = self._images[idx].parent
                
                if current_parent != next_parent:
                    rel_name = next_parent.name
                    if self._root_dir:
                        try: rel_name = next_parent.relative_to(self._root_dir).as_posix()
                        except: pass
                    
                    ans = QMessageBox.question(self, "Next Folder", 
                                               f"Moving to folder:\n{rel_name}\n\nContinue?", 
                                               QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                    if ans == QMessageBox.StandardButton.No:
                        return
                        
                if self.chk_autosave.isChecked(): self.save_current()
                self._load_image_by_index(idx)

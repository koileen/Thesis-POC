#!/usr/bin/env python3
"""Native PySide6 visual tester for saved coffee-defect checkpoints."""

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import torch
from PySide6.QtCore import Qt, QThread, Signal, QRectF, QPointF, QSize
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QToolButton,
    QMenu,
    QWidgetAction,
    QFileDialog,
    QFrame,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from test_saved_checkpoint import (
    CLASSES,
    collect_images,
    get_transforms,
    load_model,
    load_run_config,
    predict,
)


ROOT = Path(__file__).resolve().parent
CHECKPOINT_ROOT = ROOT / "linuxcheckpoints"
FILTERS = ("variety", "model", "optimizer")
IMAGE_FILTER = "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)"
BAR_COLORS = {
    "black": QColor("#267452"),
    "broken": QColor("#d29a2e"),
    "moldy": QColor("#b44d3c"),
}


def read_metadata(run_dir):
    try:
        return json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def discover_runs():
    """Discover existing checkpoints on application startup."""
    roots = [CHECKPOINT_ROOT, ROOT / "linuxruns_v2", ROOT / "runs"]
    seen = set()
    runs = []
    for folder in roots:
        if not folder.is_dir():
            continue
        # Include immediate run directories, also support a folder selected directly.
        candidates = [folder] + sorted(p for p in folder.iterdir() if p.is_dir())
        for run_dir in candidates:
            checkpoint = run_dir / "best_model.pth"
            if not checkpoint.is_file():
                continue
            identity = str(checkpoint.resolve())
            if identity in seen:
                continue
            metadata = read_metadata(run_dir)
            if all(metadata.get(key) for key in FILTERS):
                seen.add(identity)
                runs.append({
                    "path": run_dir,
                    "metadata": metadata,
                    **{key: str(metadata[key]).strip().lower() for key in FILTERS},
                })
    return sorted(runs, key=lambda run: (run["variety"], run["model"], run["optimizer"]))


def actual_class(image_path):
    for parent in image_path.parents:
        if parent.name.lower() in CLASSES:
            return parent.name.lower()
    return None


def tested_variety(image_path):
    parts = [part.lower() for part in image_path.parts]
    for variety in ("arabica", "robusta"):
        if variety in parts:
            return variety
    return "Unknown"


class ArrowComboBox(QComboBox):
    """Draw a clear arrow that remains visible with Qt's Mac stylesheet."""
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#174e3e"))
        x = self.width() - 24
        y = self.height() / 2
        painter.drawPolygon(QPolygonF([
            QPointF(x - 7, y - 3), QPointF(x + 7, y - 3), QPointF(x, y + 5)
        ]))
        painter.end()


class MultiCheckSelector(QToolButton):
    """Large checkbox menu; check any number of models, optimizers or varieties.

    An empty checked set means All. The popup stays open while toggling checks.
    """
    changed = Signal()

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.title = title
        self.checks = {}
        self.setMinimumHeight(52)
        self.setMinimumWidth(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.popup = QMenu(self)
        self.popup.setStyleSheet("QMenu { background: #fffefa; border: 1px solid #9daa9e; padding: 8px; }")
        self.setMenu(self.popup)
        self._update_label()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#174e3e"))
        x, y = self.width()-24, self.height()/2
        painter.drawPolygon(QPolygonF([QPointF(x-7,y-3), QPointF(x+7,y-3), QPointF(x,y+5)]))
        painter.end()

    def set_values(self, values):
        previous = self.selected_values()
        self.popup.clear()
        self.checks = {}
        if not values:
            action = self.popup.addAction("No checkpoints found")
            action.setEnabled(False)
        for value in values:
            holder = QWidget(self.popup)
            holder_layout = QHBoxLayout(holder)
            holder_layout.setContentsMargins(12, 4, 12, 4)
            check = QCheckBox(value, holder)
            check.setChecked(value in previous)
            check.setMinimumHeight(38)
            # macOS may hide native QCheckBox indicators inside QWidgetAction menus.
            # Draw an independent, always-visible selection mark instead.
            mark = QLabel("✓" if check.isChecked() else "")
            mark.setFixedSize(28, 30)
            mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
            mark.setStyleSheet(
                "QLabel { color: #07583f; background: #f0f6f1; "
                "border: 1px solid #9daa9e; border-radius: 5px; "
                "font-size: 18pt; font-weight: 800; }"
            )
            # Suppress only the native indicator (the QLabel above replaces it).
            check.setStyleSheet(
                "QCheckBox { background: #fffefa; font-size: 14pt; color: #202b28; }"
                "QCheckBox::indicator { width: 0px; height: 0px; }"
            )
            check.toggled.connect(lambda checked, label=mark: label.setText("✓" if checked else ""))
            check.toggled.connect(self._on_change)
            holder_layout.addWidget(mark)
            holder_layout.addWidget(check)
            widget_action = QWidgetAction(self.popup)
            widget_action.setDefaultWidget(holder)
            self.popup.addAction(widget_action)
            self.checks[value] = check
        if values:
            self.popup.addSeparator()
            clear = self.popup.addAction("Clear selection (All)")
            clear.triggered.connect(self.clear_selection)
        self.popup.setMinimumWidth(295)
        self._update_label()

    def selected_values(self):
        return {v for v, check in self.checks.items() if check.isChecked()}

    def clear_selection(self):
        for check in self.checks.values():
            check.blockSignals(True)
            check.setChecked(False)
            check.blockSignals(False)
            # Keep our custom indicators in sync when signals are blocked.
            holder = check.parentWidget()
            if holder is not None:
                mark = holder.layout().itemAt(0).widget()
                if isinstance(mark, QLabel):
                    mark.setText("")
        self._on_change()

    def _on_change(self, _=None):
        self._update_label()
        self.changed.emit()

    def _update_label(self):
        choices = sorted(self.selected_values())
        summary = ", ".join(choices) if choices else "All"
        self.setText(f"{self.title}: {summary}")
        self.setToolTip(f"{self.title}: {summary} (check multiple; none = All)")


class FitImagePreview(QWidget):
    """Paint the entire source image on every resize, without influencing layout."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.source = QPixmap()
        self.setMinimumSize(380, 380)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_image(self, pixmap):
        self.source = pixmap
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), QColor("#e9ece8"))
        if self.source.isNull():
            painter.setPen(QColor("#50635c"))
            painter.setFont(QFont("Sans Serif", 15))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Select an image to preview")
        else:
            available = self.rect().adjusted(12, 12, -12, -12)
            target_size = self.source.size()
            target_size.scale(available.size(), Qt.AspectRatioMode.KeepAspectRatio)
            x = available.x() + (available.width() - target_size.width()) / 2
            y = available.y() + (available.height() - target_size.height()) / 2
            painter.drawPixmap(QRectF(x, y, target_size.width(), target_size.height()),
                               self.source, QRectF(self.source.rect()))
        painter.end()


def classification_metrics(rows):
    """Calculate image-level metrics from true and predicted class labels."""
    valid = [row for row in rows if row.get("actual") in CLASSES and row.get("prediction") in CLASSES]
    total = len(valid)
    if not total:
        return None
    correct = sum(row["actual"] == row["prediction"] for row in valid)
    per_class = {}
    for cls in CLASSES:
        tp = sum(row["actual"] == cls and row["prediction"] == cls for row in valid)
        fp = sum(row["actual"] != cls and row["prediction"] == cls for row in valid)
        fn = sum(row["actual"] == cls and row["prediction"] != cls for row in valid)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1,
                          "support": sum(row["actual"] == cls for row in valid)}
    return {
        "accuracy": correct / total,
        "precision": sum(x["precision"] for x in per_class.values()) / len(CLASSES),
        "recall": sum(x["recall"] for x in per_class.values()) / len(CLASSES),
        "f1": sum(x["f1"] for x in per_class.values()) / len(CLASSES),
        "correct": correct,
        "total": total,
        "per_class": per_class,
    }


class ConfidenceChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.values = [0.0] * len(CLASSES)
        self.setMinimumHeight(195)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_values(self, values):
        self.values = list(values)
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        left = 102
        right = self.width() - 87
        bar_width = max(80, right - left)
        painter.setFont(QFont("Sans Serif", 13))
        for index, (label, value) in enumerate(zip(CLASSES, self.values)):
            y = 17 + index * 43
            painter.setPen(QColor("#35413c"))
            painter.drawText(4, y + 17, label.title())
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#e5e9e2"))
            painter.drawRoundedRect(left, y, bar_width, 23, 5, 5)
            painter.setBrush(BAR_COLORS.get(label, QColor("#267452")))
            painter.drawRoundedRect(left, y, int(bar_width * max(0.0, min(value, 1.0))), 23, 5, 5)
            painter.setPen(QColor("#202b28"))
            painter.drawText(self.width() - 77, y + 17, f"{value:.1%}")
        painter.end()


class PredictionWorker(QThread):
    completed = Signal(object, int, int, object)

    def __init__(self, selected_runs, image_root, recursive, device_name, parent=None):
        super().__init__(parent)
        self.selected_runs = selected_runs
        self.image_root = image_root
        self.recursive = recursive
        self.device_name = device_name

    def run(self):
        rows = []
        failures = []
        try:
            images = collect_images(self.image_root, self.recursive)
            device = torch.device(self.device_name)
            for run in self.selected_runs:
                metadata = run["metadata"]
                try:
                    config = SimpleNamespace(
                        run_dir=str(run["path"]), checkpoint=None, model=None, dropout=None
                    )
                    checkpoint, model_name, dropout = load_run_config(config)
                    _, transform = get_transforms(model_name, train=False)
                    model = load_model(checkpoint, model_name, dropout, device)
                    for image_path in images:
                        predicted, confidence, probabilities = predict(
                            model, transform, image_path, device
                        )
                        rows.append({
                            "image": str(image_path),
                            "name": str(image_path.relative_to(self.image_root))
                            if self.image_root.is_dir() else image_path.name,
                            "actual": actual_class(image_path),
                            "tested_variety": tested_variety(image_path),
                            "checkpoint": f"{run['variety']} / {model_name} / {run['optimizer']}",
                            "trained_variety": run["variety"],
                            "model": model_name,
                            "optimizer": run["optimizer"],
                            "checkpoint_path": str(checkpoint),
                            "checkpoint_name": run["path"].name,
                            "prediction": predicted,
                            "confidence": confidence,
                            "probabilities": probabilities,
                            "black_probability": probabilities[0],
                            "broken_probability": probabilities[1],
                            "moldy_probability": probabilities[2],
                            "checkpoint_filename_label": self._short_label(run),
                        })
                    del model
                except Exception as error:
                    failures.append(f"{run['path'].name}: {error}")
            self.completed.emit(rows, len(self.selected_runs), len(images), failures)
        except Exception as error:
            self.completed.emit([], len(self.selected_runs), 0, [str(error)])

    @staticmethod
    def _short_label(run):
        return f"{run['variety']}_{run['model']}_{run['optimizer']}"


class CheckpointTesterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Coffee Bean Checkpoint Tester")
        self.resize(1520, 980)
        self.setMinimumSize(950, 650)
        self.runs = discover_runs()
        self.rows = []
        self.summary_groups = {}
        self.preview_source = QPixmap()
        self.worker = None
        self._build_ui()
        self._populate_filters()
        self._refresh_matches()

    def _build_ui(self):
        self.setStyleSheet("""
            QWidget { color: #202b28; background: #f4f3ed; font: 13pt 'Sans Serif'; }
            QGroupBox { border: 1px solid #d9ddd5; border-radius: 5px; margin-top: 10px; padding: 10px; background: #fffefa; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QLineEdit, QComboBox, QToolButton { border: 1px solid #9daa9e; border-radius: 7px; padding: 11px 12px; background: #fffefa; min-height: 30px; }
            QComboBox { padding-right: 46px; }
            QComboBox::drop-down { border: none; width: 38px; }
            QComboBox::down-arrow { image: none; }
            /* Hide QToolButton's built-in menu arrow; MultiCheckSelector paints its own green arrow. */
            QToolButton::menu-indicator { image: none; width: 0px; height: 0px; }
            QComboBox QAbstractItemView { font-size: 14pt; min-width: 250px; background: #ffffff; color: #202b28; selection-background-color: #17664f; selection-color: white; outline: 0; }
            QPushButton { border: 1px solid #17664f; border-radius: 4px; padding: 12px 15px; color: white; background: #17664f; font-weight: 600; }
            QPushButton:hover { background: #10563f; }
            QPushButton:disabled { background: #8aa59a; }
            QListWidget { border: 1px solid #d9ddd5; border-radius: 4px; background: #fffefa; padding: 4px; }
            QListWidget::item { padding: 12px; border-bottom: 1px solid #eceee9; }
            QListWidget::item:selected { color: #fff; background: #17664f; }
        """)
        # A scrollable page prevents Qt from crushing the image/chart when
        # the window height is smaller than the complete dashboard.
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        central = QWidget()
        central.setMinimumWidth(920)
        scroll.setWidget(central)
        self.setCentralWidget(scroll)
        self.page_scroll = scroll
        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(12)

        title = QLabel("Coffee bean checkpoint tester")
        title.setStyleSheet("font: 700 27pt 'Serif'; color: #174e3e;")
        layout.addWidget(title)
        subtitle = QLabel("Filter saved models, test bean images, and compare class confidence.")
        subtitle.setStyleSheet("color: #66736e;")
        layout.addWidget(subtitle)

        setup = QGroupBox("Test setup")
        setup_layout = QVBoxLayout(setup)
        filter_row = QHBoxLayout()
        self.filters = {}
        for key, title_text in (("variety", "Trained variety"), ("model", "Model"), ("optimizer", "Optimizer")):
            selector = MultiCheckSelector(title_text)
            selector.changed.connect(self._refresh_matches)
            self.filters[key] = selector
            filter_row.addWidget(selector, 1)
        refresh_button = QPushButton("Clear selections")
        refresh_button.clicked.connect(self._clear_filter_selections)
        filter_row.addWidget(refresh_button, 0)
        setup_layout.addLayout(filter_row)

        self.match_count = QLabel()
        self.match_count.setStyleSheet("color: #66736e;")
        setup_layout.addWidget(self.match_count)

        input_row = QHBoxLayout()
        self.path_label = QLineEdit()
        self.path_label.setReadOnly(True)
        self.path_label.setPlaceholderText("Choose an image or a class-labeled folder…")
        self.path_label.setMinimumWidth(280)
        self.path_label.setStyleSheet("padding: 8px; border: 1px solid #d9ddd5; background: #fffefa;")
        input_row.addWidget(self.path_label, 1)
        choose_image_button = QPushButton("Choose image…")
        choose_image_button.clicked.connect(self._choose_image)
        input_row.addWidget(choose_image_button)
        choose_folder_button = QPushButton("Choose folder…")
        choose_folder_button.clicked.connect(self._choose_folder)
        input_row.addWidget(choose_folder_button)
        setup_layout.addLayout(input_row)

        action_row = QHBoxLayout()
        self.recursive = QCheckBox("Search folders recursively")
        self.recursive.setChecked(True)
        action_row.addWidget(self.recursive)
        action_row.addSpacing(12)
        action_row.addWidget(QLabel("Device"))
        self.device = ArrowComboBox()
        self.device.addItems(["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
        self.device.setMinimumWidth(115)
        self.device.setCurrentText("cuda" if torch.cuda.is_available() else "cpu")
        action_row.addWidget(self.device)
        action_row.addStretch(1)
        self.test_button = QPushButton("Test matching checkpoints")
        self.test_button.clicked.connect(self._start_prediction)
        action_row.addWidget(self.test_button)
        clear_button = QPushButton("Clear results")
        clear_button.clicked.connect(self._clear_results)
        action_row.addWidget(clear_button)
        setup_layout.addLayout(action_row)
        layout.addWidget(setup)

        summary_box = QGroupBox("Overall test results — calculated from labeled images")
        summary_layout = QVBoxLayout(summary_box)
        summary_top = QHBoxLayout()
        summary_top.addWidget(QLabel("Summarize:"))
        self.summary_choice = ArrowComboBox()
        self.summary_choice.setMinimumHeight(46)
        self.summary_choice.setMinimumWidth(440)
        self.summary_choice.currentIndexChanged.connect(self._on_summary_choice_changed)
        summary_top.addWidget(self.summary_choice, 1)
        self.summary_info = QLabel("Run a test to calculate performance metrics.")
        self.summary_info.setWordWrap(True)
        summary_top.addWidget(self.summary_info, 2)
        summary_layout.addLayout(summary_top)
        metric_row = QHBoxLayout()
        self.metric_values = {}
        for label, key in (("Accuracy", "accuracy"), ("Macro Precision", "precision"),
                           ("Macro Recall", "recall"), ("Macro F1-score", "f1")):
            metric = QFrame()
            metric.setStyleSheet("QFrame { background: #e7f0eb; border: 1px solid #c3d9cc; border-radius: 8px; }")
            mlay = QVBoxLayout(metric)
            text = QLabel(label)
            text.setStyleSheet("font-size: 12pt; font-weight: 600; color: #36584a; border: 0;")
            value = QLabel("—")
            value.setStyleSheet("font-size: 25pt; font-weight: 800; color: #14563f; border: 0;")
            mlay.addWidget(text)
            mlay.addWidget(value)
            metric_row.addWidget(metric, 1)
            self.metric_values[key] = value
        summary_layout.addLayout(metric_row)
        layout.addWidget(summary_box, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setMinimumHeight(670)
        splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # Results-view filter: unlike Test setup, this only changes which
        # already-computed predictions are visible. It does not run any model.
        results_panel = QWidget()
        results_layout = QVBoxLayout(results_panel)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(8)
        results_title = QLabel("Individual results")
        results_title.setStyleSheet("font-size: 15pt; font-weight: 700; color: #174e3e;")
        results_layout.addWidget(results_title)
        self.result_filter = ArrowComboBox()
        self.result_filter.setMinimumHeight(52)
        self.result_filter.setMinimumWidth(355)
        self.result_filter.setToolTip("Choose which completed checkpoint to display; no rerun required.")
        self.result_filter.currentIndexChanged.connect(self._filter_result_list)
        results_layout.addWidget(self.result_filter)
        self.result_count = QLabel("Run a test to view individual predictions.")
        self.result_count.setStyleSheet("color: #66736e; font-size: 11pt;")
        results_layout.addWidget(self.result_count)
        self.result_list = QListWidget()
        self.result_list.setMinimumWidth(355)
        self.result_list.setWordWrap(True)
        self.result_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.result_list.currentRowChanged.connect(self._show_result)
        results_layout.addWidget(self.result_list, 1)
        splitter.addWidget(results_panel)

        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(12, 4, 4, 4)
        preview_card = QFrame()
        preview_card.setStyleSheet("QFrame { background: #ebece5; border: 1px solid #d9ddd5; border-radius: 4px; }")
        preview_layout = QVBoxLayout(preview_card)
        self.preview = FitImagePreview()
        preview_card.setMinimumHeight(410)
        preview_layout.addWidget(self.preview)
        detail_layout.addWidget(preview_card, 5)

        self.prediction = QLabel("Choose an image and test matching checkpoints")
        self.prediction.setStyleSheet("font: 700 19pt 'Serif'; color: #17664f;")
        detail_layout.addWidget(self.prediction)
        self.actual = QLabel("Actual label: unavailable")
        self.actual.setStyleSheet("color: #35413c; font-size: 13pt; font-weight: 700;")
        self.actual.setWordWrap(True)
        detail_layout.addWidget(self.actual)
        self.checkpoint = QLabel("")
        self.checkpoint.setStyleSheet("color: #202b28; font-size: 13pt; font-weight: 700;")
        self.checkpoint.setWordWrap(True)
        detail_layout.addWidget(self.checkpoint)
        chart_box = QGroupBox("Class confidence")
        chart_layout = QVBoxLayout(chart_box)
        self.chart = ConfidenceChart()
        chart_box.setMinimumHeight(235)
        chart_layout.addWidget(self.chart)
        detail_layout.addWidget(chart_box, 0)
        splitter.addWidget(detail)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 0)

        self.status = QLabel(f"{len(self.runs)} saved checkpoints discovered.")
        self.status.setStyleSheet("color: #66736e; padding: 3px;")
        layout.addWidget(self.status)

    def _populate_filters(self):
        for key, selector in self.filters.items():
            values = sorted({run[key] for run in self.runs})
            selector.set_values(values)

    def _matching_runs(self):
        selected = {key: selector.selected_values() for key, selector in self.filters.items()}
        return [
            run for run in self.runs
            if all(not selected[key] or run[key] in selected[key] for key in FILTERS)
        ]

    def _refresh_matches(self):
        if not hasattr(self, "match_count"):
            return
        count = len(self._matching_runs())
        self.match_count.setText(
            f"{count} checkpoint(s) will run. Check multiple options; no checks means All."
        )

    def _clear_filter_selections(self):
        """Clear model filters and selected input, without erasing test results."""
        # Reset all multi-select filters to All.
        for selector in self.filters.values():
            selector.clear_selection()

        # Forget the previously chosen image or folder.
        self.input_path = None
        self.path_label.clear()
        self.path_label.setToolTip("")

        # Keep the current preview visible; Clear results removes it.
        self._refresh_matches()
        self.status.setText("Test setup cleared: filters reset to All and image/folder removed.")

    def _choose_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose bean image", str(ROOT), IMAGE_FILTER)
        if path:
            self._set_input_path(Path(path))

    def _choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Choose image folder", str(ROOT))
        if path:
            self._set_input_path(Path(path))

    def _set_input_path(self, path):
        self.input_path = path
        self.path_label.setText(str(path))
        self.path_label.setToolTip(str(path))
        preview_path = path
        if path.is_dir():
            try:
                preview_path = collect_images(path, self.recursive.isChecked())[0]
            except (OSError, ValueError, IndexError):
                self.preview_source = QPixmap()
                self.preview.set_image(self.preview_source)
                return
        self.preview_source = QPixmap(str(preview_path))
        self._update_preview()

    def _update_preview(self):
        self.preview.set_image(self.preview_source)

    def _start_prediction(self):
        matching = self._matching_runs()
        image_root = getattr(self, "input_path", None)
        if not matching or image_root is None:
            QMessageBox.warning(self, "Missing input", "Choose an image or folder and select matching filters.")
            return
        if self.device.currentText() == "cuda" and not torch.cuda.is_available():
            QMessageBox.warning(self, "CUDA unavailable", "CUDA is not available. Choose CPU.")
            return
        if len(matching) > 12:
            answer = QMessageBox.question(
                self,
                "Run many checkpoints?",
                f"The selected filters match {len(matching)} checkpoints. This may take a while. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.test_button.setEnabled(False)
        self.status.setText(f"Testing {len(matching)} matching checkpoint(s)…")
        self.worker = PredictionWorker(
            matching,
            image_root,
            self.recursive.isChecked(),
            self.device.currentText(),
            self,
        )
        self.worker.completed.connect(self._show_results)
        self.worker.start()

    def _show_results(self, rows, checkpoint_count, image_count, failures):
        # Keep all predictions in memory; the results dropdown only controls
        # which ones are displayed in the list.
        self.rows = rows
        self._populate_summary()
        self._populate_result_filters()
        succeeded = checkpoint_count - len(failures)
        self.status.setText(
            f"Completed {len(rows)} prediction(s) from {succeeded}/{checkpoint_count} checkpoint(s); "
            f"{image_count} image(s)."
        )
        self.test_button.setEnabled(True)
        if failures:
            QMessageBox.warning(self, "Checkpoint errors", "\n".join(failures))

    def _populate_result_filters(self):
        """Offer every completed checkpoint and target variety as a view."""
        self.result_filter.blockSignals(True)
        self.result_filter.clear()
        if self.rows:
            self.result_filter.addItem(f"All results ({len(self.rows)} predictions)", None)
            for key, group_rows in sorted(self.summary_groups.items()):
                run_label = group_rows[0]["checkpoint"]
                tested = key[1]
                label = f"{run_label} → {tested} ({len(group_rows)})"
                self.result_filter.addItem(label, key)
                self.result_filter.setItemData(
                    self.result_filter.count() - 1, label,
                    Qt.ItemDataRole.ToolTipRole
                )
            # Show one checkpoint by default, rather than merging predictions
            # across all selected models and optimizers.
            self.result_filter.setCurrentIndex(1)
        self.result_filter.blockSignals(False)
        self._filter_result_list()

    def _filter_result_list(self, _index=None):
        """Filter visible rows without changing the stored predictions."""
        selected_key = self.result_filter.currentData()
        if selected_key is None:
            visible = list(enumerate(self.rows))
        else:
            visible = [
                (original_index, row)
                for original_index, row in enumerate(self.rows)
                if (row["checkpoint_name"], row["tested_variety"]) == selected_key
            ]

        self.result_list.blockSignals(True)
        self.result_list.clear()
        for original_index, row in visible:
            actual = row["actual"] or "actual unknown"
            text = (
                f"{Path(row['image']).name}\n"
                f"{row['prediction'].title()} · {row['confidence']:.1%} · {actual}\n"
                f"{row['checkpoint']}"
            )
            item = QListWidgetItem(text)
            item.setSizeHint(QSize(310, 99))
            # Preserve the original index even though this list is filtered.
            item.setData(Qt.ItemDataRole.UserRole, original_index)
            self.result_list.addItem(item)
        self.result_list.blockSignals(False)

        self.result_count.setText(
            f"Showing {len(visible)} of {len(self.rows)} predictions."
            if self.rows else "Run a test to view individual predictions."
        )

        if selected_key is not None:
            # Keep the performance summary aligned with the checkpoint being viewed.
            summary_index = self.summary_choice.findData(selected_key)
            if summary_index >= 0 and summary_index != self.summary_choice.currentIndex():
                self.summary_choice.blockSignals(True)
                self.summary_choice.setCurrentIndex(summary_index)
                self.summary_choice.blockSignals(False)
                self._update_summary()

        if visible:
            self.result_list.setCurrentRow(0)
        else:
            self.prediction.setText("No individual results to display")
            self.actual.setText("Actual label: unavailable")
            self.checkpoint.clear()
            self.chart.set_values([0.0] * len(CLASSES))
            self.preview_source = QPixmap()
            self._update_preview()

    def _on_summary_choice_changed(self, _index=None):
        """Selecting a summary also shows that checkpoint's individual rows."""
        self._update_summary()
        key = self.summary_choice.currentData()
        if key is None:
            return
        view_index = self.result_filter.findData(key)
        if view_index >= 0 and view_index != self.result_filter.currentIndex():
            self.result_filter.setCurrentIndex(view_index)

    def _populate_summary(self):
        self.summary_choice.blockSignals(True)
        self.summary_choice.clear()
        groups = {}
        for row in self.rows:
            key = (row["checkpoint_name"], row["tested_variety"])
            groups.setdefault(key, []).append(row)
        self.summary_groups = groups
        for (checkpoint_name, variety), rows in sorted(groups.items()):
            ck = rows[0]["checkpoint"]
            self.summary_choice.addItem(f"{ck}  → tested: {variety} ({len(rows)} predictions)",
                                        (checkpoint_name, variety))
        self.summary_choice.blockSignals(False)
        self._update_summary()

    def _update_summary(self, _index=None):
        key = self.summary_choice.currentData()
        group_rows = self.summary_groups.get(key, []) if key is not None and hasattr(self, "summary_groups") else []
        metrics = classification_metrics(group_rows)
        if metrics is None:
            for value in self.metric_values.values():
                value.setText("—")
            if group_rows:
                self.summary_info.setText("No ground-truth labels detected. Use a test folder whose parent class folders are black, broken, and moldy.")
            else:
                self.summary_info.setText("Run a test to calculate performance metrics.")
            return
        for key, label in self.metric_values.items():
            label.setText(f"{metrics[key]:.2%}")
        self.summary_info.setText(
            f"{metrics['correct']} / {metrics['total']} labeled predictions correct. "
            f"{len(group_rows) - metrics['total']} unlabeled predictions excluded. "
            "Precision, recall, and F1 use unweighted macro averages across all 3 defect classes."
        )

    def _show_result(self, index):
        if index < 0 or index >= len(self.rows):
            return
        item = self.result_list.item(index)
        if item is None:
            return
        original_index = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(original_index, int) or not (0 <= original_index < len(self.rows)):
            return
        row = self.rows[original_index]
        self.preview_source = QPixmap(row["image"])
        self._update_preview()
        summary_key = (row["checkpoint_name"], row["tested_variety"])
        idx = self.summary_choice.findData(summary_key)
        if idx >= 0 and idx != self.summary_choice.currentIndex():
            # When showing All, selecting an image changes the metric summary,
            # but must not forcibly switch away from the All view.
            self.summary_choice.blockSignals(True)
            self.summary_choice.setCurrentIndex(idx)
            self.summary_choice.blockSignals(False)
            self._update_summary()
        self.prediction.setText(f"Predicted: {row['prediction'].title()} · {row['confidence']:.1%}")
        self.actual.setText(f"Actual label: {row['actual'] or 'unavailable'} · Tested variety: {row['tested_variety']}")
        self.checkpoint.setText(
            f"Trained variety: {row['trained_variety']}    Model: {row['model']}    Optimizer: {row['optimizer']}"
        )
        self.chart.set_values(row["probabilities"])

    def _clear_results(self):
        self.rows.clear()
        self.result_list.clear()
        self.result_filter.blockSignals(True)
        self.result_filter.clear()
        self.result_filter.blockSignals(False)
        self.result_count.setText("Run a test to view individual predictions.")
        self.prediction.setText("Choose an image and test matching checkpoints")
        self.actual.setText("Actual label: unavailable")
        self.checkpoint.clear()
        self.chart.set_values([0.0] * len(CLASSES))
        self.summary_groups = {}
        self.summary_choice.clear()
        self._update_summary()

        # Clear the image preview with the prediction results.
        # Do not call _set_input_path here: that would restore the preview.
        self.preview_source = QPixmap()
        self._update_preview()


def main():
    app = QApplication(sys.argv)
    window = CheckpointTesterWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
"""
app.py
======

Simple Sound Generator — desktop GUI for Stable Audio (Open + Stable Audio 3
architecture pipelines).

Responsibilities
----------------
* model dropdown marked ● downloaded / ◐ partial / ○ not downloaded, live
  description panel, and weight downloads that can be cancelled and resumed
* generation controls (prompt, negative prompt, duration, steps, CFG, seed,
  batch track count) with a cancellable worker thread
* a generation queue: prompts entered while a batch is still running wait their
  turn and start automatically, listed above the status line with model, length,
  batch size, steps, CFG, seed and output formats; waiting jobs can be removed,
  cleared or dragged into a different order without touching the running one
* results table with prompt/model/duration/size/timestamp metadata, newest track
  on the top row (a running batch pushes older tracks down)
* embedded player: play/pause, ±1 s / ±5 s seek, fine scrubbing slider and an
  interactive waveform with a playback head
* native drag-and-drop out of the results table (``text/uri-list``), so a row
  can be dropped straight into DaVinci Resolve's Media Pool or timeline
* a status line above the results that collapses to one row when idle and shows
  the progress bar only while a download or batch is running
* collapsible "Model details" and "Advanced sampling" sections; their open
  state is stored in ``config/settings.ini`` and restored on the next launch
* Hugging Face token handling: the field is outlined red while empty, the value
  is written to ``config/.hf_token`` and exported as ``HF_TOKEN`` so transformers
  authenticates too, and gated-repo failures name the missing token instead of
  surfacing a raw 401

Every long operation runs on a ``QThread`` and takes an ``threading.Event``
based cancel: downloads keep their partial files, a cancelled load writes
nothing, and a cancelled batch keeps the tracks that were already saved. Closing
the window cancels first and only destroys threads that have stopped (a stray
thread is detached instead of aborting the process).

Run through ``run.bat`` / ``run.sh`` (they build ``.venv`` and install
``code/requirements.txt``); running ``python code/app.py`` directly also works when the
environment is already prepared. The launcher starts the GUI through
``pythonw.exe``, so on Windows no console window stays behind the app.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

# Wall-clock marks for the one-line startup breakdown logged at every start:
# taken as early as possible, so a slow first window can be traced to a stage
# (Qt import, project imports, window build) instead of guessed at.
_STARTUP_T0 = time.perf_counter()

# --- portable environment first, before any ML import ----------------------
import paths

paths.bootstrap_env()
_STARTUP_BOOTSTRAP = time.perf_counter()

from PySide6.QtCore import (  # noqa: E402
    QMimeData,
    QSettings,
    QSize,
    Qt,
    QThread,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import (  # noqa: E402
    QColor,
    QDesktopServices,
    QDrag,
    QFont,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
_STARTUP_QT = time.perf_counter()

from model_manager import DownloadCancelled, ModelManager, ModelSpec  # noqa: E402

_STARTUP_DEPS = time.perf_counter()

# Threads that refused to stop in time are detached from the window and kept
# alive here: destroying a running QThread aborts the process, so we let the
# interpreter exit instead of crashing the user's session.
_DETACHED_THREADS: list = []

try:  # QtMultimedia ships with the PySide6 wheel on all desktop platforms
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

    MEDIA_AVAILABLE = True
except Exception:  # noqa: BLE001 - player degrades gracefully
    QAudioOutput = None  # type: ignore[assignment]
    QMediaPlayer = None  # type: ignore[assignment]
    MEDIA_AVAILABLE = False

APP_TITLE = "Simple Sound Generator · Stable Audio Desktop"
#: Shown in the window title, the startup log and used by ``_dev/build_delivery.py``
#: to name the delivery zip. Bump it before building a new distribution.
APP_VERSION = "1.7.3"
#: Title bar text: the version is part of it so a user can tell which build is
#: running (and quote it when reporting a problem) without opening a log file.
WINDOW_TITLE = f"{APP_TITLE} · v{APP_VERSION}"
# The model the window preselects at startup. It is deliberately an *id* and not
# a list position: the order in config/models.json is a presentation choice, so moving
# entries around must never silently change which checkpoint loads by default.
DEFAULT_MODEL_ID = "stable-audio-3-small-sfx"
WAVEFORM_BUCKETS = 1200

QSS = """
* { font-family: 'Segoe UI', 'Inter', 'Helvetica Neue', Arial, sans-serif; font-size: 13px; }
QMainWindow, QWidget { background: #14161b; color: #e6e9ef; }
QGroupBox { border: 1px solid #262b35; border-radius: 10px; margin-top: 14px;
            padding: 14px 10px 10px 10px; font-weight: 600; color: #9fb0c9; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextBrowser, QTableWidget {
    background: #1b1f27; border: 1px solid #2b313d; border-radius: 7px; padding: 5px 7px;
    selection-background-color: #2f6df6; color: #e6e9ef; }
QTextBrowser { padding: 8px; }
QComboBox::drop-down { border: none; width: 18px; }
QComboBox QAbstractItemView { background: #1b1f27; border: 1px solid #2b313d; selection-background-color: #2f4f9e; }
QPushButton { background: #232936; border: 1px solid #313a4a; border-radius: 7px; padding: 7px 12px; }
QPushButton:hover { background: #2b3342; }
QPushButton:disabled { color: #6b7484; background: #1c2029; }
QPushButton#primary { background: #2f6df6; border-color: #2f6df6; color: white; font-weight: 600; }
QPushButton#primary:hover { background: #3d7bff; }
QPushButton#primary:disabled { background: #26314a; color: #7d8aa3; border-color: #26314a; }
QPushButton#danger { background: #a83a3a; border-color: #a83a3a; color: white; }
QPushButton#danger:hover { background: #bd4545; }
QPushButton#danger:disabled { background: #3a2a2c; color: #8a7376; border-color: #3a2a2c; }
QHeaderView::section { background: #1b1f27; border: none; border-bottom: 1px solid #2b313d;
                       padding: 6px; color: #9fb0c9; font-weight: 600; }
QTableWidget { gridline-color: #232833; }
QTableWidget::item:selected { background: #2a3550; }
QSlider::groove:horizontal { height: 4px; background: #2b313d; border-radius: 2px; }
QSlider::handle:horizontal { background: #2f6df6; width: 12px; height: 12px; margin: -5px 0; border-radius: 6px; }
QSlider::sub-page:horizontal { background: #2f6df6; border-radius: 2px; }
QProgressBar { background: #1b1f27; border: 1px solid #2b313d; border-radius: 6px;
               height: 14px; text-align: center; color: #cfd7e5; }
QProgressBar::chunk { background: #2f6df6; border-radius: 5px; }
QScrollArea { border: none; }
QStatusBar { color: #8b96a8; }
QSplitter::handle { background: #1b1f27; }
"""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def fmt_time(ms: int) -> str:
    """Format a duration as m:ss for the transport labels."""
    ms = max(0, int(ms))
    minutes, rem = divmod(ms, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def fmt_eta(seconds: float) -> str:
    """Rough duration for an estimate: seconds, then minutes, then hours."""
    total = max(0, int(round(float(seconds))))
    if total < 60:
        return f"{total} s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes} min {secs:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


_MARK_COLORS = {"ready": "#4ec37a", "partial": "#e0a63c", "missing": "#8b96a8"}


def model_mark(state: str) -> str:
    """● downloaded / ◐ partially downloaded / ○ not downloaded."""
    return "●" if state == "ready" else ("◐" if state == "partial" else "○")


def model_item_text(spec: ModelSpec, status) -> str:
    """Dropdown entry that spells out the local storage situation."""
    if status.ready:
        suffix = f"downloaded \u00b7 {paths.human_size(status.size_bytes)}"
    elif status.state == "partial":
        suffix = f"partial \u00b7 {paths.human_size(status.size_bytes)}"
    else:
        suffix = "not downloaded"
    return f"{model_mark(status.state)}  {spec.name} \u2014 {suffix}"


def peaks_from_wav(path: str, buckets: int = WAVEFORM_BUCKETS) -> List[float]:
    """Read a WAV and compute a normalized peak envelope (best effort)."""
    try:
        import numpy as np
        import soundfile as sf

        data, _rate = sf.read(path, dtype="float32", always_2d=True)
        if data.size == 0:
            return []
        mono = data.mean(axis=1)
        buckets = max(1, int(buckets))
        if mono.size < buckets:
            peak = float(np.max(np.abs(mono))) or 1.0
            return [float(v / peak) for v in np.abs(mono)]
        step = mono.size // buckets
        trimmed = mono[: step * buckets].reshape(buckets, step)
        values = np.max(np.abs(trimmed), axis=1)
        peak = float(np.max(values)) or 1.0
        return [float(min(1.0, v / peak)) for v in values]
    except Exception:  # noqa: BLE001 - a missing waveform must not break the UI
        return []


# ---------------------------------------------------------------------------
# waveform widget
# ---------------------------------------------------------------------------
class WaveformWidget(QWidget):
    """Interactive waveform: click or drag to scrub."""

    seekRequested = Signal(int)  # milliseconds

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Start with a flat line until a track supplies peaks."""
        super().__init__(parent)
        self.setMinimumHeight(86)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click or drag to scrub · arrow keys on the slider move 10 ms")
        self._peaks: List[float] = []
        self._duration_ms = 0
        self._position_ms = 0
        self._scrubbing = False

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt API)
        return QSize(560, 96)

    def set_audio(self, peaks: List[float], duration_ms: int) -> None:
        """Store a new peak envelope and repaint the waveform."""
        self._peaks = list(peaks or [])
        self._duration_ms = max(0, int(duration_ms))
        self._position_ms = 0
        self.update()

    def set_position(self, ms: int, force: bool = False) -> None:
        """Move the playback head without re-reading the audio."""
        if self._scrubbing and not force:
            return
        self._position_ms = max(0, int(ms))
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt API)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        area = self.rect().adjusted(0, 0, -1, -1)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#0f1218"))
        painter.drawRoundedRect(area, 8, 8)

        centre_y = area.center().y() + 1
        painter.setPen(QPen(QColor("#1e2531"), 1))
        painter.drawLine(area.left() + 6, centre_y, area.right() - 6, centre_y)

        if not self._peaks:
            painter.setPen(QColor("#55617a"))
            painter.setFont(QFont(self.font().family(), 9))
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, "No audio loaded")
            return

        pad = 8
        usable = max(1, area.width() - 2 * pad)
        count = max(1, min(len(self._peaks), max(1, usable // 3)))
        step = len(self._peaks) / count
        bar_w = max(1.0, usable / count - 1.0)
        half = (area.height() - 16) / 2.0
        played_x = area.left() + pad + (
            (self._position_ms / self._duration_ms) * usable if self._duration_ms else 0
        )

        played_color = QColor("#4d8bff")
        idle_color = QColor("#33415a")
        for index in range(count):
            value = self._peaks[min(len(self._peaks) - 1, int(index * step))]
            height = max(1.0, float(value) * half)
            x = area.left() + pad + index * (usable / count)
            painter.setBrush(played_color if x <= played_x else idle_color)
            painter.drawRoundedRect(
                int(x), int(centre_y - height), int(bar_w), int(height * 2), 1, 1
            )

        painter.setPen(QPen(QColor("#8ab4ff"), 2))
        painter.drawLine(int(played_x), area.top() + 4, int(played_x), area.bottom() - 4)

    def _seek_to_x(self, x: int) -> None:
        """Map a click x-position to a 0..1 fraction and emit it."""
        if self._duration_ms <= 0:
            return
        pad = 8
        usable = max(1, self.width() - 2 * pad)
        fraction = min(1.0, max(0.0, (x - pad) / usable))
        self._position_ms = int(fraction * self._duration_ms)
        self.update()
        self.seekRequested.emit(self._position_ms)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self._scrubbing = True
        self._seek_to_x(int(event.position().x()))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if self._scrubbing:
            self._seek_to_x(int(event.position().x()))

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self._scrubbing = False
        self._seek_to_x(int(event.position().x()))


# ---------------------------------------------------------------------------
# track formats
# ---------------------------------------------------------------------------
# Best first: preview and "Format" both follow this, so playback never goes
# through a lossy file while a lossless one sits next to it.
PREVIEW_FORMAT_ORDER = ("wav", "flac", "mp3")


def track_files(record: Dict[str, Any]) -> Dict[str, str]:
    """Format → absolute path for one track, tolerating pre-1.2.0 sidecars."""
    folder = Path(record.get("path") or ".").parent
    files: Dict[str, str] = {}
    for fmt, name in (record.get("files") or {}).items():
        candidate = folder / str(name)
        if candidate.exists():
            files[str(fmt).lower()] = str(candidate)
    if not files:  # older sidecar: one file, extension is the format
        path = record.get("path")
        if path and Path(path).exists():
            files[Path(path).suffix.lstrip(".").lower()] = str(path)
    return files


def best_preview_path(record: Dict[str, Any]) -> Optional[str]:
    """Highest-quality existing file of a track (WAV → FLAC → MP3)."""
    files = track_files(record)
    for fmt in PREVIEW_FORMAT_ORDER:
        if fmt in files:
            return files[fmt]
    return next(iter(files.values()), None)


def format_label(record: Dict[str, Any]) -> str:
    """Compact ``WAV + FLAC + MP3`` list for the results table."""
    files = track_files(record)
    ordered = [fmt for fmt in PREVIEW_FORMAT_ORDER if fmt in files]
    ordered += sorted(fmt for fmt in files if fmt not in PREVIEW_FORMAT_ORDER)
    return " + ".join(fmt.upper() for fmt in ordered) or "—"


def format_tooltip(record: Dict[str, Any]) -> str:
    """Per-format size, best first, marking the one preview will play."""
    files = track_files(record)
    best = best_preview_path(record)
    rows = []
    for fmt in [f for f in PREVIEW_FORMAT_ORDER if f in files] + sorted(
        f for f in files if f not in PREVIEW_FORMAT_ORDER
    ):
        path = Path(files[fmt])
        try:
            size = paths.human_size(path.stat().st_size)
        except OSError:
            size = "missing"
        label = f"{fmt.upper()} — {size}"
        if str(path) == best:
            label += " (preview)"
        rows.append(label)
    return "\n".join(rows) or "No file on disk"


# ---------------------------------------------------------------------------
# results table with native drag-out
# ---------------------------------------------------------------------------
class ResultsTable(QTableWidget):
    """Result list whose rows can be dragged out as real file handles."""

    HEADERS = [
        "#",
        "Prompt",
        "File",
        "Model",
        "Length",
        "Format",
        "Size",
        "Created",
    ]

    #: Starting widths. Every column stays interactive, so a long prompt or file
    #: name can be widened instead of being cut off for good.
    DEFAULT_WIDTHS = (42, 300, 210, 170, 72, 78, 78, 118)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Build the empty track table with drag-out and resizable columns."""
        super().__init__(0, len(self.HEADERS), parent)
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(False)
        self.setShowGrid(False)
        self.setWordWrap(False)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)
        self.setToolTip(
            "Double-click to play · drag into DaVinci Resolve (Media Pool / timeline) "
            "or any DCC app"
        )
        header = self.horizontalHeader()
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(38)
        for column, width in enumerate(self.DEFAULT_WIDTHS):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
            header.resizeSection(column, width)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.records: List[Dict[str, Any]] = []

    # -- data --------------------------------------------------------------
    def add_result(self, meta: Dict[str, Any]) -> int:
        """Add a track at the top of the list: the newest generation comes first.

        Rows therefore read newest-to-oldest downwards, and the ``#`` column keeps
        counting in generation order (so it counts *down* the table).
        """
        row = 0
        self.insertRow(row)
        self._write_row(row, meta, str(len(self.records) + 1))
        self.records.insert(row, meta)
        return row

    def update_result(self, row: int, meta: Dict[str, Any]) -> bool:
        """Re-draw a row in place, keeping its position and ``#`` number.

        Used by **Rescan outputs** when the files behind a row changed: a file
        that came back, or an extra format that appeared next to it.
        """
        if not 0 <= row < len(self.records):
            return False
        first = self.item(row, 0)
        number = first.text() if first is not None else str(row + 1)
        self._write_row(row, meta, number)
        self.records[row] = meta
        return True

    def _write_row(self, row: int, meta: Dict[str, Any], number: str) -> None:
        """Fill an already inserted row from ``meta``."""
        prompt = (meta.get("prompt") or "").replace("\n", " ")
        # Text is stored in full: the view elides it at the column width, and
        # widening the column reveals the rest instead of a permanently cut string.
        filename = str(
            meta.get("filename") or Path(str(meta.get("path") or "")).name or ""
        )
        cells = [
            number,
            prompt,
            filename,
            meta.get("model_name") or meta.get("model_id") or "",
            f"{float(meta.get('duration_s') or 0):.1f} s",
            format_label(meta),
            paths.human_size(meta.get("size_bytes") or 0),
            str(meta.get("created") or ""),
        ]
        for column, text in enumerate(cells):
            item = QTableWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, meta)
            if column == 1:
                item.setToolTip(prompt)
            elif column == 2:
                item.setToolTip(str(meta.get("path") or filename))
            elif column == 4:
                item.setToolTip(f"Seed {meta.get('seed')} · {meta.get('sample_rate')} Hz")
            elif column == 5:
                item.setToolTip(format_tooltip(meta))
            if column in (0, 4, 5, 6, 7):
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setItem(row, column, item)

    def selected_record(self) -> Optional[Dict[str, Any]]:
        """Return the record behind the current row, or None."""
        row = self.currentRow()
        if 0 <= row < len(self.records):
            return self.records[row]
        return None

    def clear_results(self) -> None:
        """Drop every row and record, e.g. before reloading ./outputs."""
        self.setRowCount(0)
        self.records.clear()

    # -- drag out ----------------------------------------------------------
    def startDrag(self, supported_actions) -> None:  # noqa: N802 (Qt API)
        """Export the selected track as a file handle for an NLE drop."""
        record = self.selected_record()
        path = (record or {}).get("path")
        if not path or not Path(path).exists():
            return
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(path)])  # text/uri-list -> Resolve, Premiere, Explorer
        mime.setText(path)  # text/plain fallback
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)


# ---------------------------------------------------------------------------
# worker threads
# ---------------------------------------------------------------------------
class DeviceProbe(QThread):
    """Detects cuda/mps/cpu without freezing the window on torch import."""

    finished_probe = Signal(dict)

    def __init__(self, preference: str = "auto", parent: Optional[QWidget] = None) -> None:
        """Remember which compute preference the probe should honour."""
        super().__init__(parent)
        self.preference = preference

    def run(self) -> None:  # noqa: D102
        """Detect the usable device off the GUI thread and report it."""
        try:
            import audio_engine

            info = audio_engine.detect_device(self.preference)
            self.finished_probe.emit(info.as_dict())
        except Exception as exc:  # noqa: BLE001
            self.finished_probe.emit(
                {
                    "device": "unavailable",
                    "label": "PyTorch unavailable",
                    "dtype": "-",
                    "vram_gb": None,
                    "summary": f"PyTorch unavailable — {exc}",
                }
            )


class DownloadWorker(QThread):
    """Background weight download for the explicit 'Download weights' button."""

    log = Signal(str)
    progress = Signal(int, str)
    failed = Signal(str)
    cancelled = Signal(str)
    succeeded = Signal(str)

    def __init__(self, manager: ModelManager, spec: ModelSpec, token: str, parent=None) -> None:
        """Capture the model and token one download needs."""
        super().__init__(parent)
        self.manager = manager
        self.spec = spec
        self.token = token
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Abort the transfer; the bytes already received stay on disk."""
        self._cancel.set()

    def run(self) -> None:  # noqa: D102
        """Fetch the weights, reporting progress and honouring cancel."""
        try:
            self.manager.download(
                self.spec,
                log=self.log.emit,
                progress=lambda text: self.progress.emit(-1, text),
                token=self.token or None,
                cancel_event=self._cancel,
            )
            self.progress.emit(100, "Weights ready")
            self.succeeded.emit(self.spec.id)
        except DownloadCancelled:
            self.cancelled.emit(self.spec.id)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class GenerateWorker(QThread):
    """Loads the model (downloading first when needed) and runs the batch loop."""

    log = Signal(str)
    progress = Signal(int, str)
    track = Signal(dict)
    failed = Signal(str)
    cancelled = Signal(str)
    succeeded = Signal(int)
    #: seconds spent generating, and how many tracks that produced. Measuring
    #: here excludes model load/download, which is what an estimate wants.
    timing = Signal(float, int)

    def __init__(
        self,
        manager: ModelManager,
        spec: ModelSpec,
        params: Dict[str, Any],
        device_preference: str,
        token: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        """Capture the model, parameters and cancel token for one batch."""
        super().__init__(parent)
        self.manager = manager
        self.spec = spec
        self.params = params
        self.device_preference = device_preference
        self.token = token
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Ask the batch to stop at its next safe point."""
        self._cancel.set()

    def run(self) -> None:  # noqa: D102
        """Load if needed, generate the tracks and report each one."""
        try:
            import audio_engine
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"audio_engine could not be imported: {type(exc).__name__}: {exc}")
            return

        try:
            status = self.manager.status(self.spec)
            if not status.ready:
                self.progress.emit(-1, "Downloading model weights…")
                self.log.emit(
                    f"{self.spec.name}: weights not present locally ({status.detail}). "
                    f"Downloading {self.spec.repo_id} → {self.manager.local_dir(self.spec)}"
                )
                self.manager.download(
                    self.spec,
                    log=self.log.emit,
                    progress=lambda text: self.progress.emit(-1, text),
                    token=self.token or None,
                    cancel_event=self._cancel,
                )
                if self._cancel.is_set():
                    self.cancelled.emit("Cancelled before loading.")
                    return

            engine = audio_engine.AudioEngine(
                device_preference=self.device_preference, log=self.log.emit
            )
            summary = engine.load(
                self.spec,
                self.manager.local_dir(self.spec),
                progress=lambda pct, text: self.progress.emit(pct, text),
                cancel_event=self._cancel,
            )
            self.log.emit(summary)

            generation_params = audio_engine.GenerationParams(**self.params)
            generation_started = time.perf_counter()
            results = engine.generate(
                self.spec,
                generation_params,
                paths.OUTPUTS_DIR,
                on_track=lambda result: self.track.emit(result.as_meta(include_peaks=True)),
                on_progress=lambda pct, text: self.progress.emit(pct, text),
                cancel_event=self._cancel,
            )
            if results:  # a cancelled batch still measures what it produced
                self.timing.emit(time.perf_counter() - generation_started, len(results))
            engine.unload()
            if self._cancel.is_set():
                self.cancelled.emit(
                    f"Cancelled — {len(results)} finished track(s) are kept in ./outputs."
                )
                return
            self.succeeded.emit(len(results))
        except audio_engine.GenerationCancelled:
            # Cancel is a normal outcome, not a failure: no traceback, no dialog.
            self.cancelled.emit("Cancelled before any track was written.")
        except DownloadCancelled:
            self.cancelled.emit(
                "Download cancelled - the weights stay marked as not downloaded "
                "and resume from the same point next time."
            )
        except Exception as exc:  # noqa: BLE001 - reported in the GUI
            self.log.emit(traceback.format_exc())
            self.failed.emit(str(exc))


# ---------------------------------------------------------------------------
# main window
# ---------------------------------------------------------------------------
TOKEN_HINT = (
    "The token must be created on the Hugging Face website:\n"
    "huggingface.co → Settings → Access Tokens → New token → type Read.\n\n"
    "Accept the licence of each model page you want to use while signed in, "
    "otherwise the download still fails with 401 (gated repo).\n\n"
    "The value is saved to config/.hf_token inside this folder and never logged. "
    "Step-by-step instructions: README → “Getting a Hugging Face token”."
)

#: Waiting-job columns. Sampling detail and the batch size are shown per row,
#: because a queue is only useful if a long unattended run can be checked at a
#: glance instead of being remembered.
QUEUE_HEADERS = [
    "#",
    "Prompt",
    "Model",
    "Length",
    "Batch",
    "Steps",
    "CFG",
    "Seed",
    "Format",
    "Added",
]


class QueueTable(QTableWidget):
    """Queue list whose rows can be dragged into a different running order."""

    #: source rows, the row that was dropped on, and whether the drop landed in
    #: its lower half (insert after rather than before).
    rowDropped = Signal(list, int, bool)

    #: Starting widths, matching the track table: every divider drags, so a long
    #: prompt can be widened while the narrow sampling columns stay compact.
    DEFAULT_WIDTHS = (38, 240, 170, 72, 62, 58, 52, 60, 78, 142)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Build the waiting-job table with drag reordering and resizable columns."""
        super().__init__(0, len(QUEUE_HEADERS), parent)
        self.setHorizontalHeaderLabels(QUEUE_HEADERS)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setShowGrid(False)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        header = self.horizontalHeader()
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(38)
        for column, width in enumerate(self.DEFAULT_WIDTHS):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
            header.resizeSection(column, width)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._drag_rows: List[int] = []

    def startDrag(self, supported_actions) -> None:  # noqa: N802 (Qt API)
        """Record the dragged rows, then let Qt begin the drag."""
        self._drag_rows = sorted({index.row() for index in self.selectedIndexes()})
        super().startDrag(supported_actions)

    def dropEvent(self, event) -> None:  # noqa: N802 (Qt API)
        """Report where the rows should go instead of moving Qt's own items.

        The queue lives in ``MainWindow.queue``; letting the view shuffle its
        cells would leave that list — and what gets saved — out of step with
        what is on screen, so the move is applied to the data and the table is
        simply redrawn from it.
        """
        rows = [row for row in self._drag_rows if 0 <= row < self.rowCount()]
        self._drag_rows = []
        if not rows:
            event.ignore()
            return
        point = event.position().toPoint()
        index = self.indexAt(point)
        if index.isValid():
            target = index.row()
            after = point.y() > self.rowHeight(target) * 0.5
        else:
            target = max(0, self.rowCount() - 1)
            after = True
        event.setDropAction(Qt.DropAction.MoveAction)
        event.accept()
        self.rowDropped.emit(rows, target, after)


class MainWindow(QMainWindow):
    """Desktop window: model choice, sampling, queue, tracks and playback."""
    def __init__(self) -> None:
        """Build the UI, player and settings, then restore the last session."""
        super().__init__()
        self._closing = False
        self._close_timer: Optional[QTimer] = None
        # Rendered history for the warnings-only filter; the file log keeps the
        # full record regardless of what the panel is currently showing.
        self._log_entries = []  # (level, rendered line)
        self._log_warnings_only = False
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1320, 860)
        self.setMinimumSize(1060, 680)

        self.manager = ModelManager(log=self._log_line)
        self.settings = QSettings(str(paths.SETTINGS_FILE), QSettings.Format.IniFormat)
        self.device_preference = "auto"
        self.device_summary = "Detecting hardware…"

        self.worker: Optional[GenerateWorker] = None
        # Only *waiting* work lives in the queue; the running job is _active_job.
        self.queue: List[Dict[str, Any]] = []
        self._active_job: Optional[Dict[str, Any]] = None
        self._queue_paused = False
        #: Measured seconds per track from the last finished batch, for the ETA.
        self._per_track_s: Optional[float] = None
        self.download_worker: Optional[DownloadWorker] = None
        self.probe: Optional[DeviceProbe] = None

        self._loaded_path: Optional[str] = None
        self._suppress_selection_load = False
        self._scrubbing = False
        self._current_specs: List[ModelSpec] = []

        self._build_ui()
        self._init_player()
        self._populate_models()
        self._restore_settings()
        self._restore_queue()
        self._load_existing_outputs()
        self._start_probe()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        """Assemble the sidebar and results panel inside the splitter."""
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_results_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([460, 860])
        self.setCentralWidget(splitter)

        self.status_label = QLabel("Ready")
        self.statusBar().addPermanentWidget(self.status_label)
        self.device_label = QLabel(self.device_summary)
        self.statusBar().addPermanentWidget(self.device_label)

    def _build_sidebar(self) -> QWidget:
        """Build the scrollable model, prompt, sampling and action panel."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(4)

        # --- model selection & description --------------------------------
        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout(model_group)
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_layout.addWidget(self.model_combo)

        status_row = QHBoxLayout()
        self.model_status_label = QLabel("")
        self.model_status_label.setWordWrap(True)
        status_row.addWidget(self.model_status_label, 1)
        self.download_button = QPushButton("Download weights")
        self.download_button.clicked.connect(self._download_clicked)
        status_row.addWidget(self.download_button)
        model_layout.addLayout(status_row)

        folder_row = QHBoxLayout()
        self.open_model_folder = QPushButton("Open model folder")
        self.open_model_folder.clicked.connect(self._open_model_folder)
        folder_row.addWidget(self.open_model_folder)
        self.refresh_button = QPushButton("Rescan models")
        self.refresh_button.clicked.connect(self._refresh_registry)
        folder_row.addWidget(self.refresh_button)
        model_layout.addLayout(folder_row)

        # The description block is tall (name, use case, sampling table, storage),
        # so it starts collapsed behind a header row and only takes space when\
        # the user opens it.
        self.details_toggle = QToolButton()
        self.details_toggle.setCheckable(True)
        self.details_toggle.setChecked(False)
        self.details_toggle.setText("Model details")
        self.details_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.details_toggle.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.details_toggle.setAutoRaise(True)
        self.details_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.details_toggle.setToolTip(
            "Show architecture, use case, recommended sampling and local storage"
        )
        self.details_toggle.setStyleSheet(
            "QToolButton { color: #cbd5e1; font-weight: 600; padding: 2px 0; }"
        )
        self.details_toggle.toggled.connect(self._toggle_details)
        model_layout.addWidget(self.details_toggle)

        self.description = QTextBrowser()
        self.description.setMinimumHeight(240)
        self.description.setOpenExternalLinks(True)
        self.description.setVisible(False)
        model_layout.addWidget(self.description)

        token_row = QHBoxLayout()
        token_label = QLabel("HF token")
        token_label.setToolTip(TOKEN_HINT)
        token_row.addWidget(token_label)
        self.token_edit = QLineEdit(paths.read_hf_token())
        self.token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_edit.setPlaceholderText("hf_… (needed for gated repos)")
        self.token_edit.setToolTip(TOKEN_HINT)
        self.token_edit.textChanged.connect(self._update_token_style)
        token_row.addWidget(self.token_edit, 1)
        save_token = QPushButton("Save")
        save_token.clicked.connect(self._save_token)
        token_row.addWidget(save_token)
        model_layout.addLayout(token_row)
        self._update_token_style()
        layout.addWidget(model_group)

        # --- prompt -------------------------------------------------------
        prompt_group = QGroupBox("Prompt")
        prompt_layout = QVBoxLayout(prompt_group)
        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText(
            "e.g. warm analog synth pad, slow attack, cinematic, no drums"
        )
        self.prompt_edit.setFixedHeight(84)
        prompt_layout.addWidget(self.prompt_edit)
        prompt_layout.addWidget(QLabel("Negative prompt"))
        self.negative_edit = QLineEdit()
        self.negative_edit.setPlaceholderText("low quality, distorted, clipping")
        prompt_layout.addWidget(self.negative_edit)
        layout.addWidget(prompt_group)

        # --- sampling -----------------------------------------------------
        sampling_group = QGroupBox("Sampling")
        sampling_layout = QVBoxLayout(sampling_group)
        sampling_layout.setSpacing(6)

        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1.0, 47.0)
        self.duration_spin.setDecimals(1)
        self.duration_spin.setSingleStep(1.0)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.setValue(10.0)

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 300)
        self.steps_spin.setValue(100)

        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setRange(0.0, 20.0)
        self.guidance_spin.setDecimals(2)
        self.guidance_spin.setSingleStep(0.5)
        self.guidance_spin.setValue(7.0)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2 ** 31 - 1)
        self.seed_spin.setValue(42)

        self.random_seed_check = QCheckBox("Random seed each track")
        self.random_seed_check.setChecked(True)

        self.tracks_spin = QSpinBox()
        self.tracks_spin.setRange(1, 999)
        self.tracks_spin.setValue(2)
        self.tracks_spin.setToolTip(
            "Number of distinct tracks generated in this request — they are made "
            "one after another, so a long batch takes proportionally longer"
        )

        self.device_combo = QComboBox()
        for value, text in (("auto", "Auto (CUDA → MPS → CPU)"), ("cuda", "CUDA"), ("mps", "Apple MPS"), ("cpu", "CPU")):
            self.device_combo.addItem(text, value)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)

        # Always visible: how long and how many. Everything a power user tunes
        # lives behind the collapsed "Advanced sampling" header.
        basic = QGridLayout()
        basic.setHorizontalSpacing(10)
        basic.setVerticalSpacing(8)
        for row, (label, widget) in enumerate(
            (("Duration", self.duration_spin), ("Batch tracks", self.tracks_spin))
        ):
            basic.addWidget(QLabel(label), row, 0)
            basic.addWidget(widget, row, 1)

        # Output format sits with the basics: it decides what lands in ./outputs,
        # so it does not belong behind "Advanced sampling".
        self.format_checks: Dict[str, QCheckBox] = {}
        format_row = QHBoxLayout()
        format_row.setSpacing(12)
        for value, label, tip in (
            (
                "mp3",
                "MP3",
                "320 kbps CBR at LAME's best quality — a 3 min clip is ~7 MB",
            ),
            ("wav", "WAV", "24-bit PCM — lossless, ~31 MB for 3 minutes"),
            ("flac", "FLAC", "24-bit lossless, about half the size of WAV"),
        ):
            box = QCheckBox(label)
            box.setToolTip(tip)
            box.setChecked(value == "mp3")  # MP3 is the default output format
            box.toggled.connect(self._on_format_toggled)
            self.format_checks[value] = box
            format_row.addWidget(box)
        format_row.addStretch(1)
        basic.addWidget(QLabel("Format"), 2, 0)
        basic.addLayout(format_row, 2, 1)
        sampling_layout.addLayout(basic)

        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setChecked(False)
        self.advanced_toggle.setText("Advanced sampling")
        self.advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.advanced_toggle.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.advanced_toggle.setAutoRaise(True)
        self.advanced_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.advanced_toggle.setToolTip(
            "Inference steps, guidance / CFG, seed and compute device"
        )
        self.advanced_toggle.setStyleSheet(
            "QToolButton { color: #cbd5e1; font-weight: 600; padding: 2px 0; }"
        )
        self.advanced_toggle.toggled.connect(self._toggle_advanced)

        # One click back to what the selected checkpoint recommends. It sits next
        # to the header so it is reachable without opening the advanced panel.
        self.defaults_button = QPushButton("Defaults")
        self.defaults_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.defaults_button.setToolTip(
            "Reset duration, steps and guidance to the selected model's defaults"
        )
        self.defaults_button.clicked.connect(self._reset_sampling_defaults)
        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(8)
        toggle_row.addWidget(self.advanced_toggle)
        toggle_row.addStretch(1)
        toggle_row.addWidget(self.defaults_button)
        sampling_layout.addLayout(toggle_row)

        self.advanced_panel = QWidget()
        advanced = QGridLayout(self.advanced_panel)
        advanced.setContentsMargins(0, 0, 0, 0)
        advanced.setHorizontalSpacing(10)
        advanced.setVerticalSpacing(8)
        for row, (label, widget) in enumerate(
            (
                ("Inference steps", self.steps_spin),
                ("Guidance / CFG", self.guidance_spin),
            )
        ):
            advanced.addWidget(QLabel(label), row, 0)
            advanced.addWidget(widget, row, 1)

        # Seed and "Random seed each track" answer one question, so they share
        # a row: type a seed, or tick the box to ignore it.
        seed_row = QHBoxLayout()
        seed_row.setSpacing(8)
        seed_row.addWidget(self.seed_spin, 1)
        seed_row.addWidget(self.random_seed_check)
        advanced.addWidget(QLabel("Seed"), 2, 0)
        advanced.addLayout(seed_row, 2, 1)
        advanced.addWidget(QLabel("Compute"), 3, 0)
        advanced.addWidget(self.device_combo, 3, 1)
        advanced.setColumnStretch(1, 1)
        self.advanced_panel.setVisible(False)
        sampling_layout.addWidget(self.advanced_panel)
        layout.addWidget(sampling_group)

        # --- actions ------------------------------------------------------
        action_row = QHBoxLayout()
        self.generate_button = QPushButton("Generate")
        self.generate_button.setObjectName("primary")
        self.generate_button.setMinimumHeight(36)
        self.generate_button.clicked.connect(self._start_generation)
        self.generate_button.setToolTip(
            "Add these settings to the queue. Jobs run one after another, so you can "
            "keep entering prompts while a batch is still generating."
        )
        action_row.addWidget(self.generate_button, 2)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("danger")
        self.cancel_button.setMinimumHeight(36)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_generation)
        action_row.addWidget(self.cancel_button, 1)
        layout.addLayout(action_row)

        # Emergency actions are destructive to preferences, so they stay behind
        # a collapsed header. Opening this section is deliberate; the panel is
        # never restored open, even when the rest of the layout is remembered.
        self.emergency_toggle = QToolButton()
        self.emergency_toggle.setCheckable(True)
        self.emergency_toggle.setChecked(False)
        self.emergency_toggle.setText("Emergency")
        self.emergency_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.emergency_toggle.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.emergency_toggle.setAutoRaise(True)
        self.emergency_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.emergency_toggle.setToolTip(
            "Reveal maintenance actions, including Factory reset. "
            "This section starts closed so the reset cannot be clicked by accident."
        )
        self.emergency_toggle.setStyleSheet(
            "QToolButton { color: #fca5a5; font-weight: 600; padding: 2px 0; }"
        )
        self.emergency_toggle.toggled.connect(self._toggle_emergency)
        layout.addWidget(self.emergency_toggle)

        self.emergency_panel = QWidget()
        emergency_layout = QVBoxLayout(self.emergency_panel)
        emergency_layout.setContentsMargins(0, 2, 0, 0)
        emergency_layout.setSpacing(6)
        emergency_note = QLabel(
            "Factory reset restores every preference. Tracks, downloaded models "
            "and the HF token are kept."
        )
        emergency_note.setWordWrap(True)
        emergency_note.setStyleSheet("color: #fca5a5; font-size: 11px;")
        emergency_layout.addWidget(emergency_note)

        reset_row = QHBoxLayout()
        reset_row.addStretch(1)
        self.factory_reset_button = QPushButton("Factory reset")
        self.factory_reset_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.factory_reset_button.setToolTip(
            "Factory reset — put every preference back to its default:\n"
            "• sampling: 10 s, the model's steps and CFG, seed 42, random seed on, "
            "2 tracks, MP3 only\n"
            "• prompt and negative prompt cleared\n"
            "• queue cleared, column widths reset\n"
            "• volume 85 %, device Auto, model details, advanced sampling and the "
            "activity log collapsed, and the Emergency section closed\n\n"
            "Tracks in ./outputs, downloaded models and the HF token are kept. "
            "A batch that is already generating keeps running."
        )
        self.factory_reset_button.clicked.connect(self._factory_reset)
        reset_row.addWidget(self.factory_reset_button)
        emergency_layout.addLayout(reset_row)
        self.emergency_panel.setVisible(False)
        layout.addWidget(self.emergency_panel)

        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setMinimumWidth(420)
        scroll.setMaximumWidth(560)
        return scroll

    def _build_results_panel(self) -> QWidget:
        """Build the queue, status, track table, player and log."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 0, 0, 0)

        # The queue sits above the status and the finished tracks: that is where
        # new prompts land while a batch runs, and where they are checked before
        # they start.
        queue_group = QGroupBox("Queue")
        queue_layout = QVBoxLayout(queue_group)
        queue_layout.setSpacing(4)
        self.queue_table = QueueTable()
        self.queue_table.setMaximumHeight(118)
        self.queue_table.setToolTip(
            "Jobs waiting their turn. The running job is named above the list; "
            "drag a row to move it earlier or later in the queue."
        )
        self.queue_table.itemSelectionChanged.connect(self._on_queue_selection)
        self.queue_table.rowDropped.connect(self._move_queued_rows)
        queue_layout.addWidget(self.queue_table, 1)

        queue_buttons = QHBoxLayout()
        self.queue_hint = QLabel("Nothing queued — press Generate to add a job")
        self.queue_hint.setStyleSheet("color: #8b96a8;")
        self.queue_hint.setWordWrap(True)
        queue_buttons.addWidget(self.queue_hint, 1)
        self.queue_eta = QLabel("")
        self.queue_eta.setStyleSheet("color: #8b96a8;")
        self.queue_eta.setVisible(False)
        queue_buttons.addWidget(self.queue_eta)
        self.remove_queued_button = QPushButton("Remove selected")
        self.remove_queued_button.setEnabled(False)
        self.remove_queued_button.setToolTip(
            "Drop the highlighted waiting job(s) — the job that is running is untouched"
        )
        self.remove_queued_button.clicked.connect(self._remove_queued)
        queue_buttons.addWidget(self.remove_queued_button)
        self.clear_queue_button = QPushButton("Clear queue")
        self.clear_queue_button.setEnabled(False)
        self.clear_queue_button.setToolTip(
            "Drop every waiting job; a job that is already generating keeps going"
        )
        self.clear_queue_button.clicked.connect(self._clear_queue)
        queue_buttons.addWidget(self.clear_queue_button)
        self.resume_queue_button = QPushButton("Resume queue")
        self.resume_queue_button.setVisible(False)
        self.resume_queue_button.setToolTip(
            "Start the waiting jobs again after a cancel or a failure"
        )
        self.resume_queue_button.clicked.connect(self._resume_queue)
        queue_buttons.addWidget(self.resume_queue_button)
        queue_layout.addLayout(queue_buttons)
        layout.addWidget(queue_group)

        # Live status sits above the results so the user sees progress without
        # looking back at the bottom of the (often scrolled) left column. Only
        # the text line is shown while idle; the bar appears while work runs.
        self.status_box = QWidget()
        status_layout = QVBoxLayout(self.status_box)
        status_layout.setContentsMargins(0, 0, 0, 4)
        status_layout.setSpacing(3)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(10)
        self.progress.hide()
        status_layout.addWidget(self.progress)
        self.progress_text = QLabel("Idle — press Generate to start")
        self.progress_text.setStyleSheet("color: #8b96a8;")
        self.progress_text.setWordWrap(True)
        status_layout.addWidget(self.progress_text)
        layout.addWidget(self.status_box)

        results_group = QGroupBox("Generated tracks")
        results_layout = QVBoxLayout(results_group)
        self.table = ResultsTable()
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.itemDoubleClicked.connect(self._on_table_double_click)
        results_layout.addWidget(self.table, 1)
        hint_row = QHBoxLayout()
        hint = QLabel(
            "Double-click a row to audition it. Drag a row into DaVinci Resolve "
            "(Media pool / timeline) or any NLE — the file is handed over as a native "
            "24-bit WAV."
        )
        hint.setStyleSheet("color: #8b96a8;")
        hint.setWordWrap(True)
        hint_row.addWidget(hint, 1)
        self.open_outputs_button = QPushButton("Open output folder")
        self.open_outputs_button.setToolTip(
            "Open the folder that holds the selected track (./outputs when nothing is selected)"
        )
        self.open_outputs_button.clicked.connect(self._open_output_folder)
        hint_row.addWidget(self.open_outputs_button)
        self.rescan_outputs_button = QPushButton("Rescan outputs")
        self.rescan_outputs_button.setToolTip(
            "Re-read ./outputs: pick up tracks written by another session, and files "
            "that came back after being deleted — no restart needed"
        )
        self.rescan_outputs_button.clicked.connect(self._rescan_outputs)
        hint_row.addWidget(self.rescan_outputs_button)
        results_layout.addLayout(hint_row)
        layout.addWidget(results_group, 3)

        player_group = QGroupBox("Preview")
        player_layout = QVBoxLayout(player_group)
        self.now_playing = QLabel("No track selected")
        self.now_playing.setStyleSheet("color: #9fb0c9;")
        player_layout.addWidget(self.now_playing)

        self.waveform = WaveformWidget()
        self.waveform.seekRequested.connect(self._on_waveform_seek)
        player_layout.addWidget(self.waveform)

        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setSingleStep(10)  # 10 ms arrow-key scrubbing
        self.seek_slider.setPageStep(1000)
        self.seek_slider.sliderPressed.connect(self._on_slider_pressed)
        self.seek_slider.sliderReleased.connect(self._on_slider_released)
        self.seek_slider.sliderMoved.connect(self._on_slider_moved)
        player_layout.addWidget(self.seek_slider)

        time_row = QHBoxLayout()
        self.time_label = QLabel("00:00.000 / 00:00.000")
        self.time_label.setStyleSheet("color: #8b96a8;")
        time_row.addWidget(self.time_label)
        time_row.addStretch(1)
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(85)
        self.volume_slider.setFixedWidth(110)
        self.volume_slider.valueChanged.connect(self._on_volume_changed)
        time_row.addWidget(QLabel("Volume"))
        time_row.addWidget(self.volume_slider)
        player_layout.addLayout(time_row)

        controls = QHBoxLayout()
        self.play_button = QPushButton("▶ Play")
        self.play_button.setMinimumWidth(92)
        self.play_button.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_button)

        self.stop_button = QPushButton("■ Stop")
        self.stop_button.clicked.connect(self._stop_playback)
        controls.addWidget(self.stop_button)

        for label, delta, tip in (
            ("-5 s", -5000, "Rewind 5 seconds"),
            ("-1 s", -1000, "Rewind 1 second"),
            ("+1 s", 1000, "Forward 1 second"),
            ("+5 s", 5000, "Fast-forward 5 seconds"),
        ):
            button = QPushButton(label)
            button.setToolTip(tip)
            button.clicked.connect(lambda _checked=False, d=delta: self._seek_relative(d))
            controls.addWidget(button)
        controls.addStretch(1)
        player_layout.addLayout(controls)
        layout.addWidget(player_group, 2)

        # Collapsed by default: when the log is not being read, the tracks get the
        # space instead. Its open state is a preference and is restored on launch.
        self.log_toggle = QToolButton()
        self.log_toggle.setCheckable(True)
        self.log_toggle.setChecked(False)
        self.log_toggle.setText("Activity log")
        self.log_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.log_toggle.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.log_toggle.setAutoRaise(True)
        self.log_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.log_toggle.setToolTip(
            "Show or hide the activity log — the collapsed state is remembered"
        )
        self.log_toggle.setStyleSheet(
            "QToolButton { color: #cbd5e1; font-weight: 600; padding: 2px 0; }"
        )
        self.log_toggle.toggled.connect(self._toggle_log)
        log_head = QHBoxLayout()
        self.log_filter = QCheckBox("Warnings only")
        self.log_filter.setToolTip(
            "Hide INFO lines and keep warnings and errors — the choice is remembered"
        )
        self.log_filter.setCursor(Qt.CursorShape.PointingHandCursor)
        self.log_filter.setStyleSheet("QCheckBox { color: #cbd5e1; font-size: 11px; }")
        self.log_filter.toggled.connect(self._set_log_filter)
        log_head.addWidget(self.log_toggle)
        log_head.addWidget(self.log_filter)
        log_head.addStretch(1)
        layout.addLayout(log_head)

        self.log_panel = QWidget()
        log_layout = QVBoxLayout(self.log_panel)
        log_layout.setContentsMargins(0, 0, 0, 0)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setFixedHeight(110)
        self.log_view.setStyleSheet("font-family: 'Cascadia Mono', Consolas, monospace; font-size: 11px;")
        log_layout.addWidget(self.log_view)
        self.log_panel.setVisible(False)
        layout.addWidget(self.log_panel)
        return panel

    # --------------------------------------------------------------- player
    def _init_player(self) -> None:
        """Create the player and connect its signals to the transport UI."""
        self.player = None
        self.audio_output = None
        if not MEDIA_AVAILABLE:
            self.play_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.now_playing.setText(
                "QtMultimedia is unavailable — playback and waveform scrubbing are disabled "
                "(files are still generated and draggable)."
            )
            return
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(self.volume_slider.value() / 100.0)
        self.player.positionChanged.connect(self._on_player_position)
        self.player.durationChanged.connect(self._on_player_duration)
        self.player.playbackStateChanged.connect(self._on_playback_state)
        self.player.mediaStatusChanged.connect(self._on_media_status)
        self.player.errorOccurred.connect(self._on_player_error)

    def _set_player_hint(self, text: str, *, warning: bool = False) -> None:
        """Write the one-line status shown in the Preview box.

        Amber is reserved for problems, so a glance at the player says whether a
        track loaded or whether something needs the user's attention.
        """
        self.now_playing.setText(text)
        self.now_playing.setStyleSheet("color: #f0b46b;" if warning else "color: #9fb0c9;")

    def _show_preview_error(
        self, record: Dict[str, Any], *, reason: str = "", missing: bool = False
    ) -> None:
        """Explain in the Preview box why nothing can be played.

        Rows outlive their files (a track can be deleted in Explorer while the
        sidecar stays) and Qt refuses files it cannot decode. Both used to leave
        the previous track's name sitting in the player, which reads as "playing,
        just silent"; say what happened there instead.
        """
        name = Path(str(record.get("path") or "")).name or "this track"
        if missing:
            hint = f"Cannot preview {name} — the file is missing (moved or deleted)."
            self._log_line(
                f"Preview skipped: {name} is no longer where the track expects it",
                logging.WARNING,
            )
        else:
            detail = (reason or "").strip().rstrip(".")
            hint = f"Cannot preview {name} — {detail or 'Qt could not open the file'}."
            hint += " It may be damaged or in a format the player cannot decode."
            self._log_line(
                f"Preview failed for {name}: {detail or 'unknown error'}", logging.WARNING
            )
        self._loaded_path = None
        if self.player is not None:
            self.player.stop()
            if self.player.source().isValid():
                self.player.setSource(QUrl())
        self.waveform.set_audio([], 0)
        self._scrubbing = True
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setValue(0)
        self._scrubbing = False
        self.time_label.setText(f"{fmt_time(0)} / {fmt_time(0)}")
        self.play_button.setText("▶ Play")
        if MEDIA_AVAILABLE:
            self.play_button.setEnabled(False)
            self.stop_button.setEnabled(False)
        self._set_player_hint(hint, warning=True)

    # -------------------------------------------------------------- models
    def _populate_models(self) -> None:
        """Fill the dropdown and id-select the startup model.

        Selection is by id (kept selection first, then ``DEFAULT_MODEL_ID``), never
        by row, so reordering ``config/models.json`` or a rescan cannot change which model
        the window starts on. Only if both ids are absent does it fall back to the
        first entry.
        """
        current = self.model_combo.currentData()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self._current_specs = self.manager.specs(refresh=True)
        for spec in self._current_specs:
            status = self.manager.status(spec)
            self.model_combo.addItem(model_item_text(spec, status), spec.id)
        self.model_combo.blockSignals(False)
        self._refresh_model_marks()
        index = self.model_combo.findData(current) if current else -1
        if index < 0:
            # Previously selected model vanished from the registry, or first run.
            index = self.model_combo.findData(DEFAULT_MODEL_ID)
        if index < 0:
            index = 0 if self.model_combo.count() else -1
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        self._on_model_changed()

    def _refresh_model_marks(self) -> None:
        """Re-stamp every dropdown entry with its current download state.

        Runs after a download finishes, is cancelled or fails, so the ●/◐/○
        marker and the "downloaded / not downloaded" suffix never go stale.
        """
        specs = {spec.id: spec for spec in self._current_specs}
        self.model_combo.blockSignals(True)
        try:
            for index in range(self.model_combo.count()):
                spec = specs.get(self.model_combo.itemData(index))
                if spec is None:
                    continue
                status = self.manager.status(spec)
                self.model_combo.setItemText(index, model_item_text(spec, status))
                self.model_combo.setItemData(
                    index,
                    f"{spec.name}\n{status.detail}\n{self.manager.local_dir(spec)}",
                    Qt.ItemDataRole.ToolTipRole,
                )
                self.model_combo.setItemData(
                    index,
                    QColor(_MARK_COLORS.get(status.state, "#8b96a8")),
                    Qt.ItemDataRole.ForegroundRole,
                )
        finally:
            self.model_combo.blockSignals(False)

    def current_spec(self) -> Optional[ModelSpec]:
        """Return the ModelSpec behind the dropdown selection, if any."""
        model_id = self.model_combo.currentData()
        for spec in self._current_specs:
            if spec.id == model_id:
                return spec
        return None

    def _on_model_changed(self, *_args) -> None:
        """Refresh the description and apply the model's sampling defaults."""
        spec = self.current_spec()
        if spec is None:
            self.description.setHtml("<p>No models are registered.</p>")
            return
        status = self.manager.status(spec)
        self._update_description(spec, status)
        self._refresh_model_status()

        self._apply_sampling_defaults(spec, reset_duration=False)

    def _apply_sampling_defaults(self, spec: ModelSpec, *, reset_duration: bool) -> None:
        """Put steps, guidance and the duration limit back to this model's values.

        ``reset_duration=False`` is the model-switch path: it keeps whatever length
        the user typed and only clamps it when the new checkpoint cannot reach it.
        """
        # Read the length before setRange(): narrowing the range silently clamps the
        # value, which would hide that the user's choice no longer fits this model.
        wanted = float(self.duration_spin.value())
        self.duration_spin.setRange(1.0, max(1.0, float(spec.max_duration)))
        self.steps_spin.setRange(1, 300)
        self.steps_spin.setValue(int(spec.default_steps))
        self.guidance_spin.setValue(float(spec.default_guidance))
        self.guidance_spin.setEnabled(bool(spec.supports_guidance))
        self.negative_edit.setEnabled(bool(spec.supports_negative_prompt))
        if not spec.supports_negative_prompt:
            self.negative_edit.setPlaceholderText(
                "Not used by this checkpoint (guidance is baked into the weights)"
            )
        if reset_duration or wanted > spec.max_duration:
            self.duration_spin.setValue(min(10.0, spec.max_duration))

    def _reset_sampling_defaults(self) -> None:
        """Revert duration, steps and guidance to the selected model's defaults."""
        spec = self.current_spec()
        if spec is None:
            return
        self._apply_sampling_defaults(spec, reset_duration=True)
        guidance = (
            f"CFG {spec.default_guidance:g}"
            if spec.supports_guidance
            else "CFG baked in"
        )
        self._log_line(
            f"Sampling reset to {spec.name} defaults · "
            f"{self.duration_spin.value():.1f}s · "
            f"{int(spec.default_steps)} steps · {guidance}"
        )

    def _reset_widths(self) -> None:
        """Put both tables back to their built-in starting widths."""
        for table in (self.table, self.queue_table):
            header = table.horizontalHeader()
            for column, width in enumerate(table.DEFAULT_WIDTHS):
                header.resizeSection(column, width)

    def _factory_reset(self) -> None:
        """Restore every preference to what a fresh install starts with.

        Only preferences are reset. Tracks in ./outputs, the weights in config/models
        and the HF token are user content/credentials and are left alone, and a
        batch that is already generating is never cancelled by this.
        """
        if self._closing:
            return
        answer = QMessageBox.question(
            self,
            "Factory reset",
            "Restore all settings to their defaults?\n\n"
            "Sampling, prompt, formats, queue, column widths, volume, device and "
            "panel state are reset. Tracks in ./outputs, downloaded models and "
            "the HF token are kept.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        # Clearing first means anything the handlers below re-save starts from a
        # clean file instead of leaving stale keys behind.
        self.settings.clear()
        self.prompt_edit.clear()
        self.negative_edit.clear()
        self.tracks_spin.setValue(2)
        self.seed_spin.setValue(42)
        self.random_seed_check.setChecked(True)
        for fmt, box in self.format_checks.items():
            box.blockSignals(True)
            box.setChecked(fmt == "mp3")
            box.blockSignals(False)
        spec = self.current_spec()
        if spec is not None:
            self._apply_sampling_defaults(spec, reset_duration=True)
        else:
            self.duration_spin.setValue(10.0)
        self.queue.clear()
        self._queue_paused = False
        self._per_track_s = None
        self._refresh_queue()
        self.volume_slider.setValue(85)
        index = self.device_combo.findData("auto")
        if index >= 0:
            self.device_combo.setCurrentIndex(index)
        self.details_toggle.setChecked(False)
        self.advanced_toggle.setChecked(False)
        self.log_toggle.setChecked(False)
        self.emergency_toggle.setChecked(False)
        self.log_filter.setChecked(False)
        self._reset_widths()
        self.settings.sync()
        self._log_line(
            "Factory reset: preferences back to defaults · queue cleared · "
            "column widths reset"
        )

    def _refresh_model_status(self) -> None:
        """Re-stamp the status line and the Download button."""
        spec = self.current_spec()
        if spec is None:
            return
        status = self.manager.status(spec)
        colors = {"ready": "#4ec37a", "partial": "#e0a63c", "missing": "#8b96a8"}
        self.model_status_label.setText(
            f"<span style='color:{colors.get(status.state, '#8b96a8')}'>"
            f"{'●' if status.ready else '○'} {status.detail}</span>"
        )
        self.download_button.setEnabled(
            not status.ready and bool(spec.repo_id) and not self._busy()
        )
        self.download_button.setText("Re-download" if status.ready else "Download weights")

    def _update_description(self, spec: ModelSpec, status) -> None:
        """Render the selected model's metadata into the details panel."""
        status_color = {"ready": "#4ec37a", "partial": "#e0a63c"}.get(status.state, "#8b96a8")
        highlights = "".join(f"<li>{item}</li>" for item in (spec.highlights or []))
        rows = [
            ("Inference steps", f"{spec.default_steps}"),
            ("Guidance / CFG", f"{spec.default_guidance:g}" if spec.supports_guidance else "baked in"),
            ("Max duration", f"{spec.max_duration:g} s"),
            ("Sample rate", f"{spec.sample_rate} Hz"),
        ]
        if spec.min_vram_gb:
            rows.append(("Suggested VRAM", f"{spec.min_vram_gb:g} GB"))
        rows.append(("Download", self.manager.expected_size(spec)))
        table = "".join(
            f"<tr><td style='color:#8b96a8;padding-right:12px'>{key}</td>"
            f"<td>{value}</td></tr>"
            for key, value in rows
        )
        storage = (
            f"<p style='margin-top:10px'><b>Local storage</b><br>"
            f"<span style='color:{status_color}'>{status.detail}</span><br>"
            f"<span style='color:#6f7b90;font-size:11px'>{self.manager.local_dir(spec)}</span></p>"
        )
        notes = (
            f"<p style='color:#8b96a8;font-size:11px'>{spec.notes}</p>" if spec.notes else ""
        )
        gated = (
            "<p style='color:#e0a63c'><b>Gated repository</b> — accept the licence on "
            f"<a href='https://huggingface.co/{spec.repo_id}' style='color:#6ea2ff'>"
            f"huggingface.co/{spec.repo_id}</a> and provide an HF token.</p>"
            if spec.gated and spec.repo_id
            else ""
        )
        self.description.setHtml(
            f"<h2 style='margin:0 0 4px 0;font-size:17px'>{spec.name}</h2>"
            f"<p style='color:#9fb0c9;margin:0 0 10px 0'>{spec.architecture}</p>"
            f"<p style='margin:0 0 8px 0'><b>Use case</b><br>{spec.use_case}</p>"
            f"<p style='margin:0 0 8px 0'>{spec.summary}</p>"
            f"<ul style='margin:0 0 10px 18px;padding:0'>{highlights}</ul>"
            f"<p style='margin:0 0 4px 0'><b>Recommended sampling</b></p>"
            f"<table cellspacing='0' cellpadding='2'>{table}</table>"
            f"{gated}{storage}{notes}"
        )

    def _toggle_details(self, expanded: bool) -> None:
        """Show or hide the model description and remember the choice."""
        self.description.setVisible(expanded)
        self.details_toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )

    def _save_token(self) -> None:
        """Persist the HF token to config/.hf_token, or delete it when empty."""
        paths.write_hf_token(self.token_edit.text())
        self._update_token_style()
        self._log_line("HF token saved to config/.hf_token (kept inside the app folder).")

    def _toggle_advanced(self, checked: bool) -> None:
        """Reveal the advanced sampling controls (collapsed by default)."""
        self.advanced_panel.setVisible(bool(checked))
        self.advanced_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

    def _toggle_emergency(self, checked: bool) -> None:
        """Reveal the Emergency actions only after an explicit click."""
        self.emergency_panel.setVisible(bool(checked))
        self.emergency_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

    def _update_token_style(self, *_args) -> None:
        """Outline the token field red while it is empty — gated repos need one."""
        if self.token_edit.text().strip():
            self.token_edit.setStyleSheet("")
            state = "A token is entered (stored in config/.hf_token) — needed for gated repos."
        else:
            self.token_edit.setStyleSheet(
                "QLineEdit { border: 1px solid #e05a5a; border-radius: 3px;"
                " background: rgba(224, 90, 90, 0.14); }"
            )
            state = (
                "No token entered — gated checkpoints (most Stable Audio 3 models) "
                "cannot be downloaded without one."
            )
        self.token_edit.setToolTip(f"{state}\n\n{TOKEN_HINT}")

    def _open_model_folder(self) -> None:
        """Open the selected model's folder in the file manager."""
        spec = self.current_spec()
        if spec is None:
            return
        directory = self.manager.local_dir(spec)
        directory.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def _open_output_folder(self) -> None:
        """Open the folder holding the selected track (falls back to ./outputs)."""
        record = self.table.selected_record()
        path = (record or {}).get("path")
        target = Path(path).parent if path and Path(path).exists() else paths.OUTPUTS_DIR
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _toggle_log(self, checked: bool) -> None:
        """Fold the log away to its header; the choice survives a restart."""
        self.log_panel.setVisible(bool(checked))
        self.log_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self.settings.setValue("log_open", "true" if checked else "false")

    def _set_log_filter(self, checked: bool) -> None:
        """Show only warning and error lines while the box stays ticked.

        The panel keeps its own copy of the rendered history, so switching the
        filter redraws every line received so far instead of losing the INFO
        entries that were hidden while the filter was on.
        """
        self._log_warnings_only = bool(checked)
        view = getattr(self, "log_view", None)
        if view is None:
            return
        view.clear()
        for level, line in getattr(self, "_log_entries", []):
            if not self._log_warnings_only or level >= logging.WARNING:
                view.appendPlainText(line)

    def _refresh_registry(self) -> None:
        """Re-read config/models.json and config/models, then repopulate the dropdown."""
        self._populate_models()
        self._log_line("Rescanned config/models and reloaded config/models.json.")

    def _download_clicked(self) -> None:
        """Start a cancellable weight download for the selected model."""
        spec = self.current_spec()
        if spec is None or self.download_worker is not None:
            return
        if not spec.repo_id:
            QMessageBox.information(
                self,
                "No repository",
                f"'{spec.name}' is a local folder. Copy the weights into\n"
                f"{self.manager.local_dir(spec)} instead.",
            )
            return
        self.download_worker = DownloadWorker(
            self.manager, spec, self.token_edit.text().strip(), self
        )
        self.download_worker.log.connect(self._log_line)
        self.download_worker.progress.connect(self._on_progress)
        self.download_worker.failed.connect(self._on_download_failed)
        self.download_worker.cancelled.connect(self._on_download_cancelled)
        self.download_worker.succeeded.connect(self._on_download_finished)
        self.download_worker.finished.connect(self._on_download_thread_done)
        self._set_busy(True)
        self.progress.setRange(0, 0)
        self.progress_text.setText(f"Downloading {spec.name}…")
        self.download_button.setText("Downloading…")
        self._log_line(
            f"Downloading {spec.name} — Cancel stops it and keeps the partial files, "
            "so the next attempt continues where it stopped."
        )
        self.download_worker.start()

    def _on_download_finished(self, model_id: str) -> None:
        """Mark the model ready and refresh its dropdown marks."""
        self._refresh_model_marks()
        self._refresh_model_status()
        spec = self.current_spec()
        if spec is not None:
            self._update_description(spec, self.manager.status(spec))
        self._log_line(f"Weights for {model_id} are ready — marked as downloaded.")

    def _on_download_cancelled(self, model_id: str) -> None:
        """Explain that the partial weights are kept for a retry."""
        self._refresh_model_marks()
        self._refresh_model_status()
        self.progress_text.setText("Download cancelled · resumes next time")
        self._log_line(
            f"Download of {model_id} cancelled — the partial files stay in config/models, "
            "so pressing Download again (or Generate) continues from there."
        )

    def _on_download_failed(self, message: str) -> None:
        """Report a download failure, unless the app is closing."""
        self._refresh_model_marks()
        self._log_line(f"Download failed: {message}", logging.ERROR)
        if self._closing:
            return  # never block a shutdown behind a modal dialog
        QMessageBox.critical(self, "Download failed", message)

    def _on_download_thread_done(self) -> None:
        """Release the download worker once its thread stopped."""
        self.download_worker = None
        self._set_busy(False)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress_text.setText("Idle — press Generate to start")
        self._pump_queue()  # jobs queued behind a download start now

    # ------------------------------------------------------------- formats
    def _selected_formats(self) -> tuple:
        """Ticked output formats, in the canonical write order.

        ``format_checks`` is populated in the engine's ``OUTPUT_FORMATS`` order
        (mp3, wav, flac), so its insertion order is already the write order.
        """
        return tuple(
            fmt for fmt, box in self.format_checks.items() if box.isChecked()
        )

    def _on_format_toggled(self, checked: bool) -> None:
        """A batch needs at least one format, so refuse to untick the last one."""
        if checked or self._selected_formats():
            return
        box = self.sender()
        if isinstance(box, QCheckBox):
            box.blockSignals(True)
            box.setChecked(True)
            box.blockSignals(False)
        self._log_line("At least one output format has to stay ticked.")

    # ----------------------------------------------------------- generation
    def _start_generation(self) -> None:
        """Add the current settings to the queue and start it when the app is free.

        Pressing Generate while a batch runs no longer does nothing: the job is
        appended and starts as soon as the running one finishes, so a session can
        be planned prompt by prompt instead of waiting for each batch to end.
        """
        spec = self.current_spec()
        if spec is None:
            QMessageBox.warning(self, "No model", "Select a model first.")
            return
        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            QMessageBox.information(
                self, "Prompt required", "Describe the audio you want to generate."
            )
            self.prompt_edit.setFocus()
            return

        formats = self._selected_formats()
        if not formats:
            QMessageBox.information(
                self,
                "Output format",
                "Tick at least one output format (MP3, WAV or FLAC).",
            )
            return

        token = self.token_edit.text().strip()
        if token:
            paths.write_hf_token(token)

        job = {
            "prompt": prompt,
            "negative_prompt": self.negative_edit.text(),
            "model_id": spec.id,
            "model_name": spec.name,
            "duration": float(self.duration_spin.value()),
            "steps": int(self.steps_spin.value()),
            "guidance": float(self.guidance_spin.value()),
            "seed": -1 if self.random_seed_check.isChecked() else int(self.seed_spin.value()),
            "num_tracks": int(self.tracks_spin.value()),
            "formats": formats,
            "device": self.device_preference,
            "added": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.queue.append(job)
        self._queue_paused = False
        self._refresh_queue()
        seed_text = "random seed" if job["seed"] < 0 else f"seed {job['seed']}"
        self._log_line(
            f"Queued: {prompt[:60]!r} [{job['model_name'] or job['model_id']}] · "
            f"{job['num_tracks']} track(s) · {job['duration']:.1f}s · "
            f"{job['steps']} steps · CFG {job['guidance']:g} · {seed_text} · "
            f"{'+'.join(f.upper() for f in formats)} · {len(self.queue)} waiting"
        )
        self._pump_queue()

    def _spec_by_id(self, model_id: Optional[str]) -> Optional[ModelSpec]:
        """Look up a model id, used when starting a queued job."""
        for spec in self._current_specs:
            if spec.id == model_id:
                return spec
        return None

    def _pump_queue(self) -> None:
        """Start the next waiting job when nothing else is running."""
        if self._queue_paused or self._busy() or not self.queue:
            self._refresh_queue()
            return
        self._start_job(self.queue.pop(0))

    def _start_job(self, job: Dict[str, Any]) -> None:
        """Run one queued job on a worker thread."""
        spec = self._spec_by_id(job.get("model_id"))
        if spec is None:
            # config/models.json can change between queueing and starting.
            self._log_line(
                f"Skipped a queued job — model {job.get('model_name') or job.get('model_id')!r} "
                "is no longer listed in config/models."
            )
            self._pump_queue()
            return
        params = {
            "prompt": job["prompt"],
            "negative_prompt": job.get("negative_prompt") or "",
            "duration": float(job["duration"]),
            "steps": int(job["steps"]),
            "guidance": float(job["guidance"]),
            "seed": int(job["seed"]),
            "num_tracks": int(job["num_tracks"]),
            "formats": tuple(job["formats"]),
        }
        self._active_job = job
        self.worker = GenerateWorker(
            self.manager,
            spec,
            params,
            job.get("device") or self.device_preference,
            self.token_edit.text().strip(),
            self,
        )
        self.worker.log.connect(self._log_line)
        self.worker.progress.connect(self._on_progress)
        self.worker.track.connect(self._on_track)
        self.worker.failed.connect(self._on_generation_failed)
        self.worker.cancelled.connect(self._on_generation_cancelled)
        self.worker.succeeded.connect(self._on_generation_done)
        self.worker.timing.connect(self._on_batch_timing)
        self._set_busy(True)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        waiting = f" · {len(self.queue)} still queued" if self.queue else ""
        self.progress_text.setText(f"Starting {job['prompt'][:48]!r}…{waiting}")
        self._refresh_queue()
        seed_text = "random seed" if params["seed"] < 0 else f"seed {params['seed']}"
        self._log_line(
            f"—— Generate · {spec.name} · {params['num_tracks']} track(s) · "
            f"{params['duration']:.1f}s · {params['steps']} steps · "
            f"CFG {params['guidance']:g} · {seed_text} · "
            f"{'+'.join(f.upper() for f in params['formats'])} · "
            f"{len(self.queue)} queued after this ——"
        )
        self.worker.start()

    def _cancel_generation(self) -> None:
        """Cancel whichever background job is running (batch or download)."""
        if self.worker is not None:
            self.worker.cancel()
            self.progress_text.setText("Cancelling after the current step…")
            self._log_line("Cancel requested.")
        if self.download_worker is not None:
            self.download_worker.cancel()
            self.progress_text.setText("Stopping the download…")
            self._log_line(
                "Cancel requested — the partial weights are kept for the next attempt."
            )
        self.cancel_button.setEnabled(False)

    def _on_progress(self, percent: int, text: str) -> None:
        """Drive the progress bar and status text from worker signals."""
        if percent < 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(int(percent))
        if text:
            self.progress_text.setText(text)

    def _on_track(self, meta: Dict[str, Any]) -> None:
        """Add a finished track to the table and select a batch's first one."""
        first_track = not self.table.records
        row = self.table.add_result(meta)  # newest first, so it lands on row 0
        newest = self.table.item(row, 0)
        if newest is not None:
            self.table.scrollToItem(newest)
        self.status_label.setText(
            f"{len(self.table.records)} track(s) · last: {Path(meta['path']).name}"
        )
        # Load the first finished track of a batch into the player, then leave the
        # selection alone: re-loading every track would fight the user's choice.
        if first_track or self._loaded_path is None:
            self._suppress_selection_load = True
            self.table.selectRow(row)
            self._suppress_selection_load = False
            self._load_record(meta)

    def _on_generation_cancelled(self, message: str) -> None:
        """Cancellation is a normal outcome: log it, never raise an error box."""
        self._log_line(message or "Cancelled.")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress_text.setText("Cancelled")
        self._pause_queue_after("cancelled")
        self._finish_generation_ui()

    def _on_generation_done(self, count: int) -> None:
        """Finish the batch UI, log the result and start the next job."""
        prompt = str((self._active_job or {}).get("prompt") or "")
        where = f" for {prompt[:60]!r}" if prompt else ""
        self._log_line(
            f"Generation finished · {count} track(s) written to ./outputs{where}"
        )
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.progress_text.setText(f"Done · {count} track(s)")
        self._finish_generation_ui()
        self._pump_queue()

    def _on_batch_timing(self, seconds: float, count: int) -> None:
        """Remember the last batch's speed so the queue can quote an estimate."""
        if count <= 0 or seconds <= 0:
            return
        self._per_track_s = seconds / count
        self._refresh_queue_buttons()
        tracks = self._queued_track_count()
        estimate = (
            f"≈ {fmt_eta(self._per_track_s * tracks)} for the {tracks} waiting track(s)"
            if tracks
            else "nothing waiting"
        )
        self._log_line(
            f"Timing: {self._per_track_s:.2f} s per track from {count} track(s) "
            f"in {seconds:.1f} s — {estimate}."
        )

    def _queued_track_count(self) -> int:
        """Total tracks still waiting, used for the queue estimate."""
        return sum(int(job.get("num_tracks") or 1) for job in self.queue)

    def _on_generation_failed(self, message: str) -> None:
        """Turn a failed batch into a readable dialog and pause the queue."""
        if "cancel" in (message or "").lower():
            # A user cancel is not an error: log it, do not raise a dialog.
            self._log_line(message)
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.progress_text.setText("Cancelled")
            self._finish_generation_ui()
            return
        self._log_line(f"Generation failed: {message}", logging.ERROR)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress_text.setText("Failed")
        self._pause_queue_after("failed")
        self._finish_generation_ui()
        if self._closing:
            return
        if "no hugging face token is entered" in (message or "").lower():
            self._update_token_style()
            self.token_edit.setFocus()
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("Generation failed")
        box.setText(message.splitlines()[0] if message else "Unknown error")
        box.setDetailedText(message)
        box.exec()

    def _finish_generation_ui(self) -> None:
        """Return the controls to idle after a batch, whatever happened."""
        self.worker = None
        self._active_job = None
        self._set_busy(False)
        self._refresh_queue()

    def _busy(self) -> bool:
        """True while a download, model load or batch is still running."""
        return (self.worker is not None and self.worker.isRunning()) or (
            self.download_worker is not None and self.download_worker.isRunning()
        )

    def _set_busy(self, busy: bool) -> None:
        # Generate stays clickable while work runs: pressing it now queues the job
        # for when the current one finishes instead of being refused.
        """Lock or unlock the controls that must not be touched during work."""
        self.cancel_button.setEnabled(busy)  # cancels a download as well as a batch
        self.model_combo.setEnabled(not busy)
        self.download_button.setEnabled(not busy)
        self.refresh_button.setEnabled(not busy)
        self.rescan_outputs_button.setEnabled(not busy)
        if hasattr(self, "status_box"):
            self._set_status_active(busy)
        if not busy:
            self._refresh_model_marks()
            self._refresh_model_status()

    def _set_status_active(self, active: bool) -> None:
        """Show the bar while a job runs; idle collapses to the text line only."""
        self.progress.setVisible(bool(active))
        if not active and not self.progress_text.text().strip():
            self.progress_text.setText("Idle — press Generate to start")

    # ---------------------------------------------------------------- queue
    def _refresh_queue(self) -> None:
        """Redraw the waiting jobs, then the label and buttons that describe them."""
        self.queue_table.blockSignals(True)
        self.queue_table.setRowCount(0)
        for index, job in enumerate(self.queue):
            row = self.queue_table.rowCount()
            self.queue_table.insertRow(row)
            prompt = str(job.get("prompt") or "")
            seed = int(job.get("seed") or -1)
            batch = int(job.get("num_tracks") or 1)
            cells = [
                str(index + 1),
                prompt if len(prompt) <= 70 else prompt[:69] + "…",
                str(job.get("model_name") or ""),
                f"{float(job.get('duration') or 0):.1f} s",
                f"×{batch}",
                str(int(job.get("steps") or 0)),
                f"{float(job.get('guidance') or 0):g}",
                "random" if seed < 0 else str(seed),
                "+".join(str(fmt).upper() for fmt in (job.get("formats") or ())),
                str(job.get("added") or "—"),
            ]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column == 1:
                    item.setToolTip(prompt)
                elif column == 4:
                    item.setToolTip(f"{batch} distinct track(s) in this job")
                elif column == 0:
                    item.setToolTip("Position in the queue")
                elif column == 9:
                    item.setToolTip("Added to the queue at this local time")
                if column != 1 and column != 2:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.queue_table.setItem(row, column, item)
        self.queue_table.blockSignals(False)
        self._refresh_queue_buttons()

    def _refresh_queue_buttons(self) -> None:
        """Keep the hint and the remove/clear/resume buttons in step with the list."""
        # The active job is the source of truth, not the thread: a worker that was
        # just created is not "running" yet, and the label would read "nothing
        # queued" for the first moments of every job.
        active = self._active_job
        waiting = len(self.queue)
        if active is not None:
            text = (
                f"Running: {str(active.get('prompt'))[:50]!r} · "
                f"{int(active.get('num_tracks') or 1)} track(s) · {waiting} waiting"
            )
        elif waiting and self._queue_paused:
            text = f"Paused · {waiting} job(s) waiting — press Resume queue"
        elif waiting:
            text = f"{waiting} job(s) waiting"
        else:
            text = "Nothing queued — press Generate to add a job"
        self.queue_hint.setText(text)
        tracks = self._queued_track_count()
        if self._per_track_s is not None and tracks:
            self.queue_eta.setText(
                f"≈ {fmt_eta(self._per_track_s * tracks)} for {tracks} waiting track(s)"
            )
            self.queue_eta.setToolTip(
                f"From the last finished batch: {self._per_track_s:.2f} s per track. "
                "Waiting jobs only — the running job is not counted."
            )
            self.queue_eta.setVisible(True)
        else:
            self.queue_eta.setVisible(False)
        has_rows = self.queue_table.rowCount() > 0
        selected = self.queue_table.selectionModel().selectedRows() if has_rows else []
        self.clear_queue_button.setEnabled(has_rows)
        self.remove_queued_button.setEnabled(bool(selected))
        paused = bool(waiting) and self._queue_paused
        self.resume_queue_button.setVisible(paused)
        self.resume_queue_button.setEnabled(paused)

    def _pause_queue_after(self, why: str) -> None:
        """Stop waiting jobs from starting by themselves after a cancel or failure.

        Both usually mean the settings deserve a look first, so the rest of the
        queue waits behind **Resume queue** rather than repeating the mistake.
        """
        if not self.queue:
            return
        self._queue_paused = True
        self._log_line(
            f"Queue paused after the job was {why} — {len(self.queue)} job(s) still "
            "waiting. Press Resume queue to continue."
        )

    def _on_queue_selection(self) -> None:
        """Enable Remove only while a waiting job is selected."""
        self._refresh_queue_buttons()

    def _move_queued_rows(self, rows: List[int], target: int, after: bool) -> None:
        """Apply a drag: put the selected jobs at the drop position.

        ``rows`` are the positions the drag started from, ``target`` the row it
        was dropped on, and ``after`` whether it landed in that row's lower half.
        The move is expressed as an insertion into the jobs that are not being
        dragged, so moving a block downwards cannot shift it twice.
        """
        rows = sorted({row for row in rows if 0 <= row < len(self.queue)})
        if not rows or not 0 <= target < len(self.queue):
            self._refresh_queue()
            return
        anchor = target + (1 if after else 0)
        insert_at = sum(1 for index in range(anchor) if index not in rows)
        moving = [self.queue[row] for row in rows]
        rest = [job for index, job in enumerate(self.queue) if index not in rows]
        self.queue = rest[:insert_at] + moving + rest[insert_at:]
        self._refresh_queue()
        order = " · ".join(
            f"{index + 1}. {str(job.get('prompt'))[:28]!r}"
            for index, job in enumerate(self.queue[:5])
        )
        more = f" (+{len(self.queue) - 5} more)" if len(self.queue) > 5 else ""
        self._log_line(f"Queue reordered: {order}{more}")

    def _remove_queued(self) -> None:
        """Drop the highlighted waiting jobs; the running job is untouched."""
        rows = sorted(
            {index.row() for index in self.queue_table.selectionModel().selectedRows()},
            reverse=True,
        )
        if not rows:
            return
        dropped = [self.queue[row] for row in rows if 0 <= row < len(self.queue)]
        for row in rows:
            if 0 <= row < len(self.queue):
                del self.queue[row]
        self._refresh_queue()
        names = ", ".join(repr(str(job.get("prompt"))[:28]) for job in dropped)
        self._log_line(
            f"Removed {len(dropped)} queued job(s) ({names}) — "
            f"{len(self.queue)} still waiting."
        )

    def _clear_queue(self) -> None:
        """Drop every waiting job and leave the running one alone."""
        dropped = list(self.queue)
        self.queue.clear()
        self._queue_paused = False
        self._refresh_queue()
        if dropped:
            names = ", ".join(repr(str(job.get("prompt"))[:28]) for job in dropped[:5])
            more = " …" if len(dropped) > 5 else ""
            self._log_line(f"Cleared {len(dropped)} queued job(s): {names}{more}")

    def _resume_queue(self) -> None:
        """Leave the paused state and start the first waiting job."""
        self._queue_paused = False
        self._log_line(f"Resuming the queue — {len(self.queue)} job(s) waiting.")
        self._pump_queue()

    # ------------------------------------------------------------- outputs
    @staticmethod
    def _resolve_sidecar(sidecar: Path, meta: Dict[str, Any]) -> Optional[Path]:
        """Find the WAV a sidecar describes, tolerating a moved app folder.

        Current sidecars store only ``filename``, resolved next to the sidecar,
        so the track list survives moving or copying the whole folder. Sidecars
        written by older versions store an absolute ``path``: that is used while
        it still exists, otherwise its bare file name is looked up beside the
        sidecar, which recovers tracks generated before the move.
        """
        folder = sidecar.parent
        candidates: List[Path] = []
        filename = str(meta.get("filename") or "").strip()
        if filename:
            candidates.append(folder / filename)
        # A track can exist in several formats: losing one file must not hide the
        # row while a sibling of another format is still there.
        for name in (meta.get("files") or {}).values():
            candidates.append(folder / str(name))
        stored = str(meta.get("path") or "").strip()
        if stored:
            candidates.append(Path(stored))
            candidates.append(folder / Path(stored).name)
        for candidate in candidates:
            try:
                if candidate.is_file():
                    return candidate
            except OSError:
                continue
        return None

    @staticmethod
    def _record_key(meta: Dict[str, Any]) -> str:
        """Identity of a track across rescans: its file-name stem."""
        name = Path(str(meta.get("path") or "")).name or str(meta.get("filename") or "")
        return Path(name).stem.lower()

    def _collect_output_records(self) -> List[Dict[str, Any]]:
        """Read every sidecar in ./outputs whose audio still exists, newest first."""
        try:
            sidecars = sorted(
                paths.OUTPUTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
            )
        except OSError:
            return []
        records: List[Dict[str, Any]] = []
        for sidecar in sidecars[:300]:
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
                if not isinstance(meta, dict):
                    continue
                wav = self._resolve_sidecar(sidecar, meta)
                if wav is None:
                    continue
                # Everything downstream expects an absolute path; the sidecar
                # stays portable and is resolved against ./outputs above.
                meta["path"] = str(wav)
                records.append(meta)
            except Exception:  # noqa: BLE001
                continue
        return records

    def _load_existing_outputs(self) -> None:
        """Populate the table from the sidecars already in ./outputs."""
        records = self._collect_output_records()
        # records is newest-first and add_result() inserts on top, so feed it the
        # oldest first to end up with the newest at the top, like a fresh run.
        for meta in reversed(records):
            self.table.add_result(meta)
        if records:
            self._log_line(f"Loaded {len(records)} existing track(s) from ./outputs")

    def _rescan_outputs(self) -> None:
        """Merge what is in ./outputs now into the table, without a restart.

        Rows outlive their files, so a track deleted in Explorer keeps its row and
        its ``#`` number. Copying the file back (or a batch written by another
        instance) is picked up here: existing rows are refreshed in place and only
        genuinely new tracks are added, so the order never jumps around.
        """
        records = self._collect_output_records()
        previous = list(self.table.records)
        rows = {self._record_key(meta): row for row, meta in enumerate(previous)}
        refreshed = 0
        added = 0
        reappeared = 0
        for meta in reversed(records):  # oldest first, so the newest ends up on top
            row = rows.get(self._record_key(meta))
            if row is None:
                self.table.add_result(meta)
                added += 1
                continue
            elif self.table.update_result(row, meta):
                # Every row is re-drawn so a reappeared file is picked up, but only
                # count it as refreshed when something about it really changed.
                old = previous[row]
                was_missing = bool(old.get("_missing"))
                if old.get("path") != meta.get("path") or (
                    old.get("files") or {}
                ) != (meta.get("files") or {}):
                    refreshed += 1
                elif was_missing and best_preview_path(meta) is not None:
                    # Same path, same sidecar: the file itself came back. A path
                    # comparison cannot see that, so the flag from the last scan does.
                    reappeared += 1
        present = {self._record_key(meta) for meta in records}
        for record in self.table.records:
            if self._record_key(record) in present:
                record.pop("_missing", None)
            else:
                record["_missing"] = True
        # A row that could not be previewed a moment ago may be playable again.
        selected = self.table.selected_record()
        if selected is not None and self._loaded_path is None and best_preview_path(selected):
            self._load_record(selected)
        if added or refreshed or reappeared:
            details = f"{added} new · {refreshed} refreshed"
            if reappeared:
                details += f" · {reappeared} back on disk"
            self._log_line(
                f"Rescanned ./outputs: {len(records)} track(s) on disk · {details}"
            )
        else:
            self._log_line(
                f"Rescanned ./outputs: {len(records)} track(s) on disk · nothing new"
            )

    # -------------------------------------------------------------- preview
    def _on_selection_changed(self) -> None:
        """Load the selected track into the preview and player."""
        if self._suppress_selection_load:
            return
        record = self.table.selected_record()
        if record is None:
            return
        if record.get("path") == self._loaded_path:
            return
        self._load_record(record)

    def _load_record(self, record: Dict[str, Any]) -> None:
        # Preview always uses the highest-quality file that exists, so a track
        # saved as MP3 + WAV is judged on the WAV rather than the lossy copy.
        """Resolve, decode and show one track, explaining failures in amber."""
        files = track_files(record)
        path = best_preview_path(record)
        if not path:
            self._show_preview_error(record, missing=True)
            return
        self._loaded_path = path
        duration_ms = int(float(record.get("duration_s") or 0) * 1000)
        peaks = record.get("peaks")
        if not peaks:
            try:
                peaks = peaks_from_wav(path)
            except Exception:  # noqa: BLE001 - MP3/FLAC have no WAV header
                peaks = []
            record["peaks"] = peaks
        self.waveform.set_audio(peaks, duration_ms)
        shown = Path(path).suffix.lstrip(".").upper()
        extra = f" (best of {len(files)})" if len(files) > 1 else ""
        self._set_player_hint(
            f"{Path(path).name}   ·   preview {shown}{extra}   ·   "
            f"seed {record.get('seed')}   ·   {record.get('sample_rate')} Hz   ·   "
            f"{Path(path).parent}"
        )
        if MEDIA_AVAILABLE:
            # A previous track may have failed to load or been deleted.
            self.play_button.setEnabled(True)
            self.stop_button.setEnabled(True)
        self._scrubbing = True
        self.seek_slider.setRange(0, max(0, duration_ms))
        self.seek_slider.setValue(0)
        self._scrubbing = False
        self.time_label.setText(f"{fmt_time(0)} / {fmt_time(duration_ms)}")
        if self.player is not None:
            self.player.stop()
            self.player.setSource(QUrl.fromLocalFile(path))
            self.play_button.setText("▶ Play")

    def _toggle_play(self) -> None:
        """Start or pause playback of the loaded track."""
        if self.player is None or self._loaded_path is None:
            return
        if not Path(self._loaded_path).exists():
            # Deleted between selecting the row and pressing Play.
            self._show_preview_error({"path": self._loaded_path}, missing=True)
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            if self.player.mediaStatus() == QMediaPlayer.MediaStatus.EndOfMedia or (
                self.player.duration() > 0
                and self.player.position() >= self.player.duration() - 5
            ):
                self.player.setPosition(0)
            self.player.play()

    def _stop_playback(self) -> None:
        """Stop playback and return the position to the start."""
        if self.player is None:
            return
        self.player.stop()
        self.seek_slider.setValue(0)
        self.waveform.set_position(0)
        self.time_label.setText(f"{fmt_time(0)} / {fmt_time(self.seek_slider.maximum())}")

    def _seek_relative(self, delta_ms: int) -> None:
        """Jump playback by a number of seconds, clamped to the track."""
        if self.player is None:
            return
        duration = self.seek_slider.maximum()
        target = min(max(0, self.player.position() + delta_ms), max(0, duration))
        self.player.setPosition(int(target))
        self.seek_slider.setValue(int(target))
        self.waveform.set_position(int(target))

    def _on_slider_pressed(self) -> None:
        """Remember the position so scrubbing can pause and resume."""
        self._scrubbing = True

    def _on_slider_moved(self, value: int) -> None:
        """Follow the slider while the user drags it."""
        self.waveform.set_position(value, force=True)
        self.time_label.setText(f"{fmt_time(value)} / {fmt_time(self.seek_slider.maximum())}")

    def _on_slider_released(self) -> None:
        """Commit the scrubbed position and resume if it was playing."""
        self._scrubbing = False
        if self.player is not None:
            self.player.setPosition(int(self.seek_slider.value()))

    def _on_waveform_seek(self, ms: int) -> None:
        """Seek to the fraction of the track that was clicked."""
        if self.player is not None:
            self.player.setPosition(int(ms))
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(int(ms))
        self.seek_slider.blockSignals(False)
        self.time_label.setText(f"{fmt_time(ms)} / {fmt_time(self.seek_slider.maximum())}")

    def _on_player_position(self, ms: int) -> None:
        """Keep the slider, time label and head in step with the player."""
        if not self._scrubbing:
            self.seek_slider.setValue(int(ms))
        self.waveform.set_position(int(ms))
        self.time_label.setText(f"{fmt_time(ms)} / {fmt_time(self.seek_slider.maximum())}")

    def _on_player_duration(self, ms: int) -> None:
        """Show the track length once the player knows it."""
        if ms > 0:
            self.seek_slider.setRange(0, int(ms))

    def _on_playback_state(self, state) -> None:
        """Swap the Play/Pause label to match the player state."""
        if self.player is None:
            return
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.play_button.setText("⏸ Pause" if playing else "▶ Play")

    def _on_media_status(self, status) -> None:
        """Route decoder errors to the friendly preview message."""
        if self.player is None:
            return
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.play_button.setText("▶ Play")
        elif status == QMediaPlayer.MediaStatus.InvalidMedia and self._loaded_path:
            # Some backends only report a decoder failure through the status.
            self._on_player_error(QMediaPlayer.Error.FormatError, "the player cannot decode it")

    def _on_player_error(self, error, message: str) -> None:
        """Surface a decode/playback failure in the Preview box.

        Qt's only other outlet is the log file, which the user is not looking at
        when they click a track, so the player line carries the short version and
        the log keeps the backend's own wording.
        """
        if self.player is None or self._loaded_path is None:
            return
        self._show_preview_error(
            {"path": self._loaded_path}, reason=message or self.player.errorString()
        )

    def _on_volume_changed(self, value: int) -> None:
        """Apply the volume slider to the player and remember it."""
        if self.audio_output is not None:
            self.audio_output.setVolume(value / 100.0)

    def _on_table_double_click(self, item: QTableWidgetItem) -> None:
        """Double click auditions the track; folders open via the button instead."""
        record = self.table.selected_record()
        if record is None or self.player is None:
            return
        path = record.get("path")
        if not path or not Path(path).exists():
            self._load_record(record)  # explains the missing file in the Preview box
            return
        if self._loaded_path != path:
            self._load_record(record)
        self.player.setPosition(0)
        self.player.play()

    # -------------------------------------------------------------- devices
    def _start_probe(self) -> None:
        """Detect hardware on a worker thread so startup never blocks."""
        self.probe = DeviceProbe(self.device_preference, self)
        self.probe.finished_probe.connect(self._on_probe_done)
        self.probe.start()

    def _on_probe_done(self, info: Dict[str, Any]) -> None:
        """Show the detected device in the status bar and log it."""
        self.device_summary = info.get("summary", "Unknown device")
        self.device_label.setText(self.device_summary)
        self._log_line(f"Hardware: {self.device_summary}")

    def _on_device_changed(self, *_args) -> None:
        """Remember the compute choice and probe it again."""
        self.device_preference = self.device_combo.currentData() or "auto"
        self.settings.setValue("device", self.device_preference)
        self._start_probe()

    # ------------------------------------------------------------- settings
    #: Anything outside this range is treated as a damaged saved width.
    WIDTH_LIMITS = (38, 2000)

    @classmethod
    def _read_widths(cls, raw: Any, expected: int) -> Optional[List[int]]:
        """Parse stored column widths, or return None when they cannot be used.

        A saved row from a build with a different column set has the wrong length
        and is ignored, so adding a column never leaves the table misaligned.
        """
        try:
            values = json.loads(raw) if isinstance(raw, str) else raw
            widths = [int(value) for value in values]
        except (TypeError, ValueError):
            return None
        if not isinstance(values, list) or len(widths) != expected:
            return None
        low, high = cls.WIDTH_LIMITS
        return [min(high, max(low, width)) for width in widths]

    def _restore_widths(self, key: str, table: QTableWidget) -> None:
        """Apply saved column widths, keeping the defaults if the setting is bad."""
        widths = self._read_widths(
            self.settings.value(key, "", type=str), table.columnCount()
        )
        if widths is None:
            return
        header = table.horizontalHeader()
        for column, width in enumerate(widths):
            header.resizeSection(column, width)

    @staticmethod
    def _widths_json(table: QTableWidget) -> str:
        """Serialise a table's live column widths for config/settings.ini."""
        return json.dumps([int(table.columnWidth(c)) for c in range(table.columnCount())])

    def _restore_settings(self) -> None:
        """Reload geometry, controls, disclosures, widths and the queue."""
        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
            self.prompt_edit.setPlainText(self.settings.value("prompt", "", type=str) or "")
            self.negative_edit.setText(self.settings.value("negative", "", type=str) or "")
            self.duration_spin.setValue(self.settings.value("duration", 10.0, type=float))
            self.tracks_spin.setValue(self.settings.value("tracks", 2, type=int))
            self.random_seed_check.setChecked(
                self.settings.value("random_seed", "true", type=str) == "true"
            )
            stored_formats = self.settings.value("formats", "", type=str) or ""
            if stored_formats:
                wanted = {
                    part.strip().lower()
                    for part in stored_formats.split(",")
                    if part.strip()
                }
                for fmt, box in self.format_checks.items():
                    box.blockSignals(True)
                    box.setChecked(fmt in wanted)
                    box.blockSignals(False)
                if not self._selected_formats():
                    self.format_checks["mp3"].setChecked(True)  # never start with none
            self.volume_slider.setValue(self.settings.value("volume", 85, type=int))
            device = self.settings.value("device", "auto", type=str) or "auto"
            index = self.device_combo.findData(device)
            if index >= 0:
                self.device_combo.blockSignals(True)
                self.device_combo.setCurrentIndex(index)
                self.device_combo.blockSignals(False)
                self.device_preference = device
            # Disclosure state is a preference like any other: reopening the app
            # the way it was left beats re-collapsing it every launch.
            self.details_toggle.setChecked(
                self.settings.value("details_open", "false", type=str) == "true"
            )
            self.advanced_toggle.setChecked(
                self.settings.value("advanced_open", "false", type=str) == "true"
            )
            self.log_toggle.setChecked(
                self.settings.value("log_open", "false", type=str) == "true"
            )
            self.log_filter.setChecked(
                self.settings.value("log_warnings_only", "false", type=str) == "true"
            )
            # A widened prompt is a preference like any other, so both tables come
            # back at the widths they were left with.
            self._restore_widths("results_widths", self.table)
            self._restore_widths("queue_widths", self.queue_table)
        except Exception:  # noqa: BLE001 - corrupt ini must not block startup
            pass

    def _save_settings(self) -> None:
        """Write every preference, including widths and waiting jobs."""
        try:
            self.settings.setValue("geometry", self.saveGeometry())
            self.settings.setValue("prompt", self.prompt_edit.toPlainText())
            self.settings.setValue("negative", self.negative_edit.text())
            self.settings.setValue("duration", float(self.duration_spin.value()))
            self.settings.setValue("tracks", int(self.tracks_spin.value()))
            self.settings.setValue(
                "random_seed", "true" if self.random_seed_check.isChecked() else "false"
            )
            self.settings.setValue("volume", int(self.volume_slider.value()))
            self.settings.setValue("device", self.device_preference)
            self.settings.setValue(
                "details_open", "true" if self.details_toggle.isChecked() else "false"
            )
            self.settings.setValue(
                "advanced_open", "true" if self.advanced_toggle.isChecked() else "false"
            )
            self.settings.setValue(
                "log_open", "true" if self.log_toggle.isChecked() else "false"
            )
            self.settings.setValue(
                "log_warnings_only",
                "true" if self.log_filter.isChecked() else "false",
            )
            self.settings.setValue("formats", ",".join(self._selected_formats()))
            self.settings.setValue("results_widths", self._widths_json(self.table))
            self.settings.setValue("queue_widths", self._widths_json(self.queue_table))
            # Only jobs that were still waiting: the running one is cancelled on
            # close and whatever it already wrote stays in ./outputs, so restoring
            # it would duplicate those tracks.
            self.settings.setValue("queue", json.dumps(self.queue, ensure_ascii=False))
            self.settings.sync()
            if self.queue:
                self._log_line(
                    f"Saved {len(self.queue)} queued job(s) for the next launch."
                )
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _coerce_job(raw: Any) -> Optional[Dict[str, Any]]:
        """Rebuild one stored queue entry, or drop it when it is unusable.

        ``config/settings.ini`` is user-editable, so every field is validated instead of
        trusted: a half-written entry must never stop the window from opening.
        """
        if not isinstance(raw, dict):
            return None
        prompt = str(raw.get("prompt") or "").strip()
        if not prompt:
            return None

        def number(key: str, default: float, cast) -> Any:
            try:
                return cast(raw.get(key, default))
            except (TypeError, ValueError):
                return default

        wanted = {str(fmt).lower() for fmt in (raw.get("formats") or [])}
        formats = [fmt for fmt in ("mp3", "wav", "flac") if fmt in wanted] or ["mp3"]
        added = raw.get("added")
        # Queues saved by older builds have no timestamp; the column shows "—".
        added = added.strip()[:19] if isinstance(added, str) else ""
        return {
            "prompt": prompt,
            "negative_prompt": str(raw.get("negative_prompt") or ""),
            "model_id": str(raw.get("model_id") or ""),
            "model_name": str(raw.get("model_name") or ""),
            "duration": max(0.1, float(number("duration", 10.0, float))),
            "steps": max(1, int(number("steps", 50, int))),
            "guidance": float(number("guidance", 7.0, float)),
            "seed": int(number("seed", -1, int)),
            "num_tracks": max(1, int(number("num_tracks", 1, int))),
            "formats": formats,
            "device": str(raw.get("device") or "auto"),
            "added": added,
        }

    def _restore_queue(self) -> None:
        """Bring back the jobs that were still waiting when the app was closed.

        They come back **paused** on purpose: reopening a session should not start
        spending minutes of generation before the user has seen what is there.
        **Resume queue** starts them.
        """
        raw = self.settings.value("queue", "", type=str) or ""
        if not raw.strip():
            return
        try:
            stored = json.loads(raw)
        except (TypeError, ValueError):
            self._log_line("Saved queue could not be read — starting with an empty queue.")
            return
        if not isinstance(stored, list):
            return
        jobs = [job for job in (self._coerce_job(item) for item in stored) if job]
        if not jobs:
            return
        self.queue.extend(jobs)
        self._queue_paused = True
        self._refresh_queue()
        self.progress_text.setText(
            f"Queue restored · {len(jobs)} job(s) waiting — press Resume queue"
        )
        names = ", ".join(repr(str(job["prompt"])[:28]) for job in jobs[:5])
        self._log_line(
            f"Restored {len(jobs)} queued job(s) from config/settings.ini ({names}"
            f"{' …' if len(jobs) > 5 else ''}) — paused until Resume queue."
        )

    # ----------------------------------------------------------------- log
    def _log_line(self, message: str, level: int = logging.INFO) -> None:
        """Append one timestamped line to the log file and the activity panel."""
        text = str(message or "").strip()
        if not text:
            return
        logging.log(level, text)
        line = f"[{datetime.now():%H:%M:%S}] {text}"
        entries = getattr(self, "_log_entries", None)
        if entries is None:
            entries = self._log_entries = []
        entries.append((level, line))
        del entries[:-2000]  # matches the view's block limit
        # The filter only affects the panel: the rotating file log above always
        # receives every level, which is what a bug report needs.
        if not getattr(self, "_log_warnings_only", False) or level >= logging.WARNING:
            try:
                self.log_view.appendPlainText(line)
            except Exception:  # noqa: BLE001 - logging must never raise
                pass
        if not hasattr(self, "status_label") or not hasattr(self, "table"):
            return
        if self.table.records:
            self.status_label.setText(f"{len(self.table.records)} track(s) · {text[:90]}")

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        """Cancel a running job, then close once its thread has really stopped."""
        running = [
            thread
            for thread in (self.worker, self.download_worker)
            if thread is not None and thread.isRunning()
        ]
        if running and not self._closing:
            self._closing = True
            self.progress_text.setText("Stopping the running job…")
            self._log_line("Closing — stopping the running job first.")
            for thread in running:
                try:
                    thread.cancel()  # downloads keep their partial files
                except Exception:  # noqa: BLE001 - a finished thread must not block
                    pass
                thread.finished.connect(self.close)
            # A runtime call that cannot be interrupted must not hold the window
            # hostage: after the grace period the thread is detached and closed.
            self._close_timer = QTimer(self)
            self._close_timer.setSingleShot(True)
            self._close_timer.timeout.connect(self._force_close)
            self._close_timer.start(20000)
            event.ignore()
            return
        self._shutdown()
        super().closeEvent(event)

    def _force_close(self) -> None:
        """Close even though a job did not stop, without destroying a live thread."""
        self._log_line("The running step did not stop in time — closing anyway.")
        for name in ("worker", "download_worker", "probe"):
            thread = getattr(self, name, None)
            if thread is None or not thread.isRunning():
                continue
            try:
                thread.cancel()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
            thread.setParent(None)
            _DETACHED_THREADS.append(thread)
            setattr(self, name, None)
        self.close()

    def _shutdown(self) -> None:
        """Save settings, cancel work and close once the threads have stopped."""
        self._save_settings()
        if getattr(self, "_close_timer", None) is not None:
            self._close_timer.stop()
        for name in ("worker", "download_worker"):
            thread = getattr(self, name, None)
            if thread is None or not thread.isRunning():
                continue
            try:
                thread.cancel()
            except Exception:  # noqa: BLE001
                pass
            if not thread.wait(15000):
                # Never destroy a running QThread: detach and let Python exit.
                thread.setParent(None)
                _DETACHED_THREADS.append(thread)
                setattr(self, name, None)
                logging.warning("A worker did not stop in time and was detached.")
        if self.probe is not None and self.probe.isRunning():
            if not self.probe.wait(5000):
                self.probe.setParent(None)
                _DETACHED_THREADS.append(self.probe)
                self.probe = None
        if self.player is not None:
            self.player.stop()


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
# Set once every class and constant above has been defined.
_STARTUP_IMPORTED = time.perf_counter()


def setup_logging() -> None:
    """Configure the rotating file log and the in-window activity log."""
    paths.ensure_directories()
    handler = RotatingFileHandler(
        paths.LOG_FILE, maxBytes=1_500_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    logging.info(
        "=== Simple Sound Generator v%s starting (pid %s) ===", APP_VERSION, os.getpid()
    )
    logging.info("App root: %s", paths.APP_ROOT)


def startup_timing_line(app_ready: float, window_ready: float, shown: float) -> str:
    """Return the compact boot breakdown that is logged once per start.

    The marks come from ``time.perf_counter()``. Time spent before the first
    import in this file (interpreter start, site-packages scanning) is not part
    of the total, which is why it is described as measured from ``app.py``.
    """

    def span(start: float, end: float) -> str:
        return f"{1000.0 * (end - start):.0f}"

    return (
        f"startup: {span(_STARTUP_T0, shown)} ms total = "
        f"import {span(_STARTUP_T0, _STARTUP_IMPORTED)} "
        f"(bootstrap {span(_STARTUP_T0, _STARTUP_BOOTSTRAP)}, "
        f"qt {span(_STARTUP_BOOTSTRAP, _STARTUP_QT)}, "
        f"deps {span(_STARTUP_QT, _STARTUP_DEPS)}, "
        f"code {span(_STARTUP_DEPS, _STARTUP_IMPORTED)}) + "
        f"qapp {span(_STARTUP_IMPORTED, app_ready)} + "
        f"window {span(app_ready, window_ready)} + "
        f"show {span(window_ready, shown)}"
    )


def _install_excepthook() -> None:
    """Log uncaught exceptions instead of losing them under pythonw."""
    def _hook(exc_type, exc_value, exc_tb):
        message = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logging.error("Unhandled exception:\n%s", message)
        try:
            box = QMessageBox()
            box.setIcon(QMessageBox.Icon.Critical)
            box.setWindowTitle("Unexpected error")
            box.setText(f"{exc_type.__name__}: {exc_value}")
            box.setDetailedText(message)
            box.exec()
        except Exception:  # noqa: BLE001
            print(message, file=sys.stderr)

    sys.excepthook = _hook


def main(argv: Optional[List[str]] = None) -> int:
    """Create the application, apply the theme and show the window."""
    args = list(sys.argv[1:] if argv is None else argv)
    # Identify the build without starting the GUI - and before the bootstrap, so
    # `python code/app.py --version` prints the line and touches nothing on disk.
    if any(argument in ("--version", "-V") for argument in args):
        print(f"{APP_TITLE} v{APP_VERSION}")
        return 0
    paths.bootstrap_env()
    # Windows consoles default to cp1252 while log lines contain “·”, “→” and
    # model names: make the streams tolerant instead of raising UnicodeEncodeError.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - not a real console (pythonw, frozen build)
            pass
    setup_logging()
    app = QApplication([sys.argv[0]] + args)
    app.setApplicationName("Simple Sound Generator")
    app.setApplicationDisplayName(WINDOW_TITLE)
    app.setStyleSheet(QSS)
    app_ready = time.perf_counter()
    _install_excepthook()
    window = MainWindow()
    window_ready = time.perf_counter()
    window.show()
    shown = time.perf_counter()
    # Logged before the event loop runs, so a window that never paints still
    # leaves the cause of the delay in logs/app.log.
    logging.info("%s", startup_timing_line(app_ready, window_ready, shown))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

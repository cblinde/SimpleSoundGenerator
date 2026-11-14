"""Render the main window offscreen to a PNG (dev and CI helper).

The README image and the release asset are produced without a display: Qt runs
with ``QT_QPA_PLATFORM=offscreen``, the hardware probe is disabled (no torch in a
CI image) and every path that would touch real state is redirected to a
temporary folder. Only PySide6, numpy and soundfile are needed - no model
weights, no GPU.

Usage::

    python _dev/screenshot.py [output.png]     # default: docs/screenshot.png
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import wave
from pathlib import Path

# Must be set before PySide6 is imported; CI has no X/Wayland display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))  # app.py imports paths/model_manager as top-level modules

from PySide6.QtWidgets import QApplication  # noqa: E402

import app as A  # noqa: E402

#: Three finished tracks, so the table, the waveform and the status line show
#: what a real session looks like.
SAMPLES = [
    (
        "Warm analog synth pad, slow filter sweep, wide stereo",
        "Stable Audio 3 Small · Music",
        12.0,
        8241,
        "20261114-120102000_warm_analog_synth_pad_t01.wav",
    ),
    (
        "Heavy wooden door creak and latch, close perspective",
        "Stable Audio 3 Small · SFX",
        4.0,
        117,
        "20261114-120147000_heavy_wooden_door_creak_and_lat_t01.wav",
    ),
    (
        "Rain on a metal roof, distant thunder, ambience loop",
        "Stable Audio Open 1.0",
        20.0,
        5503,
        "20261114-120310000_rain_on_a_metal_roof_distant_thu_t01.wav",
    ),
]


def write_sample(path: Path, seconds: float, seed: int, rate: int = 44100) -> None:
    """Write a quiet but visible 24-bit stereo WAV for the waveform preview.

    The audio itself is irrelevant - the app only draws the peak envelope - but
    a shaped signal looks more like a real track than silence.
    """
    frames = bytearray()
    total = int(rate * seconds)
    for index in range(total):
        t = index / rate
        envelope = 0.55 * math.exp(-0.25 * t) * (0.6 + 0.4 * math.sin(2 * math.pi * 0.12 * t + seed))
        left = envelope * math.sin(2 * math.pi * (110 + seed % 90) * t)
        right = envelope * math.sin(2 * math.pi * (164 + seed % 60) * t + 0.7)
        for value in (left, right):
            sample = int(max(-1.0, min(1.0, value)) * (2**23 - 1))
            frames += sample.to_bytes(3, "little", signed=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(3)
        handle.setframerate(rate)
        handle.writeframes(bytes(frames))


def main() -> int:
    """Build the window in its normal look, save a PNG and exit."""
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "screenshot.png"
    output = output if output.is_absolute() else ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    work = Path(tempfile.mkdtemp(prefix="ssg-screenshot-"))
    outputs = work / "outputs"
    outputs.mkdir()
    # Never read or write the real config/settings.ini, outputs/ or logs/.
    A.paths.SETTINGS_FILE = work / "settings.ini"
    A.paths.OUTPUTS_DIR = outputs
    A.paths.LOG_FILE = work / "app.log"

    # The probe imports torch on a QThread; a screenshot needs no hardware
    # detection (and CI has no torch at all).
    A.MainWindow._start_probe = lambda self: None  # type: ignore[method-assign]

    A.setup_logging()
    qt = QApplication.instance() or QApplication([])
    qt.setStyleSheet(A.QSS)
    window = A.MainWindow()

    window.resize(1600, 900)  # wider than the default so every table column fits
    window.prompt_edit.setPlainText("Warm analog synth pad, slow filter sweep, wide stereo")
    window.token_edit.setText("hf_" + "x" * 30)  # rendered as dots; not a credential
    window.device_summary = "CPU · float32"
    window.device_label.setText(window.device_summary)

    for prompt, model_name, duration, seed, name in reversed(SAMPLES):
        path = outputs / name
        write_sample(path, duration, seed)
        window._on_track(
            {
                "path": str(path),
                "filename": name,
                "prompt": prompt,
                "model_name": model_name,
                "duration_s": duration,
                "size_bytes": path.stat().st_size,
                "created": "2026-11-14 12:03",
                "seed": seed,
                "sample_rate": 44100,
            }
        )
        qt.processEvents()

    # Show the newest track in the preview box and its waveform. The preview
    # line drops the temporary folder, which would otherwise read as a path
    # from a real installation.
    if window.table.rowCount():
        window.table.selectRow(0)
        record = window.table.records[0]
        window._load_record(record)
        window._set_player_hint(
            f"{Path(str(record['path'])).name}   ·   preview WAV   ·   "
            f"seed {record['seed']}   ·   44100 Hz   ·   ./outputs"
        )

    window.show()
    for _ in range(5):
        qt.processEvents()

    pixmap = window.grab()
    if pixmap.isNull() or not pixmap.save(str(output), "PNG"):
        print(f"ERROR: could not write {output}", file=sys.stderr)
        return 1

    window._closing = True
    window.close()
    qt.processEvents()
    try:
        shown = output.relative_to(ROOT)
    except ValueError:
        shown = output
    print(f"screenshot: {shown} ({pixmap.width()}x{pixmap.height()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

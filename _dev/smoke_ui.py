"""Dev smoke test: builds the real window under the current Qt binding."""
import sys, tempfile, wave
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths; paths.bootstrap_env()
from PySide6.QtCore import Qt, qVersion
from PySide6.QtWidgets import QApplication, QLabel, QPushButton
import PySide6
q = QApplication([])
import app as A

print("  binding      :", f"PySide6 {PySide6.__version__} / Qt {qVersion()}")
print("  QtMultimedia :", A.QMediaPlayer is not None)
w = A.MainWindow(); w.show(); q.processEvents()
print("  title        :", w.windowTitle())
print("  model order  :", [w.model_combo.itemData(i) for i in range(w.model_combo.count())])
print("  preselected  :", w.model_combo.currentData())
print("  advanced     :", "collapsed" if not w.advanced_panel.isVisible() else "OPEN")
print("  token hint   :", w.token_edit.toolTip().splitlines()[0][:52])
print("  buttons      :", len(w.findChildren(QPushButton)), "| labels:", len(w.findChildren(QLabel)))
print("  enums        :", w.token_edit.echoMode() == A.QLineEdit.EchoMode.Password,
      int(Qt.AlignmentFlag.AlignCenter) > 0,
      w.model_combo.itemData(0, Qt.ItemDataRole.UserRole) is not None)

tmp = Path(tempfile.mkdtemp()); frames = b"\x00" * (44100 * 6 // 10)
def wav(name):
    p = tmp / name
    with wave.open(str(p), "wb") as h:
        h.setnchannels(2); h.setsampwidth(3); h.setframerate(44100); h.writeframes(frames)
    return p
def meta(prompt, p):
    return {"path": str(p), "prompt": prompt, "model_name": "t", "duration_s": 0.5,
            "size_bytes": p.stat().st_size, "created": "now", "seed": 1, "sample_rate": 44100}
for n in ("a.wav", "b.wav", "c.wav"):
    w._on_track(meta("prompt-" + n[0], wav(n))); q.processEvents()
print("  track order  :", [w.table.item(r, 1).text() for r in range(w.table.rowCount())])
print("  records sync :", [m["prompt"] for m in w.table.records] ==
      [w.table.item(r, 1).text() for r in range(w.table.rowCount())])
print("  drag payload :", bool(w.table.selected_record()))
print("  player       :", w.player is not None and w.audio_output is not None,
      "| volume", w.audio_output.volume() if w.audio_output else None)
print("  signal kind  :", type(A.GenerateWorker.track).__name__,
      type(A.MainWindow.seekRequested if hasattr(A.MainWindow, 'seekRequested') else A.DeviceProbe.finished_probe).__name__)
print("  progress bar :", w.progress.isVisible() is False, "| idle status h:", w.status_box.height())
w._save_settings(); w.close(); q.processEvents()
print("  closed       :", not w.isVisible())

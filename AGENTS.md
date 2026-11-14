# AGENTS.md — working notes for coding agents

This file is for agents and humans changing **Simple Sound Generator**, a
portable PySide6 desktop app that generates audio with Stable Audio Open 1.0
and Stable Audio 3. It records the rules and traps learned while building and
testing the app. Read it before editing, and follow the repo's own code and
comments where they are more specific.

`AGENTS.md` is development documentation: it is intentionally **not** in the
delivery file list and must never be added to a shipped zip.

## What the app must keep doing

- **Stay portable and self-contained.** Everything (runtime, weights, caches,
  settings, logs, output) lives inside the app folder. `run.bat` / `run.sh`
  bootstrap Python, `.venv` and `code/requirements.txt` without admin rights or PATH
  changes. Do not introduce writes outside `APP_ROOT` or require a system
  install.
- **Stay importable without ML packages or a GPU.** `code/app.py` must open its
  window and report a missing/partial install instead of crashing. Model code
  imports torch/diffusers lazily, behind guards.
- **Need no git and no preinstalled Python.** The launchers download portable
  CPython and install the SA3 runtime from a GitHub source archive. Never
  reintroduce a `git+https://` requirement or assume `git` exists on the target.
- **Stay PySide6 (LGPL).** Do not swap in a GPL-only UI toolkit or a dependency
  whose licence blocks closed-source distribution. Check `LICENSE` and the
  dependency list before adding a package.
- **Never ship local state.** No `.venv`, `.python`, `config/models/`, `outputs/`,
  `logs/`, `config/settings.ini`, `config/.hf_token`, `__pycache__`, logs or zips inside the
  artifact.
- **Make the first run diagnosable.** The launchers append checkpoints and pip's
  detailed `--log` output to `logs/bootstrap.log`; never rely on the console,
  which closes when the default Windows launch hands off to `pythonw.exe`.
- **Keep destructive work cancellable.** Long operations run on a `QThread`
  with a `threading.Event`; downloads keep partial files, a cancelled batch
  keeps tracks already written, and closing the window cancels first.
- **Keep audio quality choices explicit.** 24-bit WAV/FLAC, 320 kbps MP3; every
  selected format is encoded from the same normalized buffer and shares a stem.

## Layout

| Path | Responsibility |
|---|---|
| `code/app.py` | GUI, queue, playback, logging, settings, workers, entry point |
| `code/audio_engine.py` | model loading, pipelines, sampling, WAV/MP3/FLAC writing |
| `code/model_manager.py` | `config/models.json` registry, local status, downloads, HF token use |
| `code/flash_attn_setup.py` | optional FlashAttention wheel resolution |
| `code/requirements.txt` | pinned dependency set installed into `.venv` |
| `code/paths.py` | `APP_ROOT` paths and cache environment bootstrap (import first) |
| `config/models.json` | editable model registry (gitignored changes are local) |
| `config/models/` | HF/Torch caches and downloaded weights (gitignored) |
| `config/settings.ini`, `config/.hf_token` | preferences and credential (gitignored) |
| `outputs/`, `logs/` | generated audio, the rotating app log and `bootstrap.log` (gitignored) |
| `_dev/build_delivery.py` | dev-only staging + versioned zip builder |
| `_dev/check_versions.py` | CI guard: `APP_VERSION`, zip entries, README/requirements pins |
| `_dev/screenshot.py` | dev/CI-only offscreen renderer for `docs/screenshot.png` |
| `docs/screenshot.png` | window image embedded in the README and attached to releases |
| `.github/workflows/release.yml` | CI: compile/validate, zip build, screenshot, GitHub release |
| `delivery/` | generated staging copy and all kept zips (gitignored) |

`code/paths.py` must be imported before torch/diffusers/transformers so caches land
under `config/models`. It is called at the top of `code/app.py` and again from the engine
and manager so scripted use works too.

## Commands (Windows, Git Bash)

```bash
# compile check — use the project interpreter, not a system python
.venv/Scripts/python.exe -m py_compile code/app.py code/audio_engine.py code/model_manager.py \
    code/flash_attn_setup.py code/paths.py _dev/check_versions.py

# offscreen UI check (write the harness to a file, see below)
PYTHONIOENCODING=utf-8 QT_QPA_PLATFORM=offscreen \
    .venv/Scripts/python.exe -u _dev/check_something.py

# shell script syntax (run.sh cannot actually run here: no .venv/bin/python)
bash -n run.sh

# package (bump APP_VERSION in code/app.py first)
.venv/Scripts/python.exe _dev/build_delivery.py

# verify APP_VERSION, the zip entries and the documented pins
.venv/Scripts/python.exe _dev/check_versions.py
```

There is no `.venv/bin/python` on this Windows checkout. Use
`.venv/Scripts/python.exe`; do not "fix" the bootstrap by creating a POSIX venv.

## Testing recipe and traps

A reliable window-only harness:

```python
import os, sys, tempfile
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, ".")
from PySide6.QtWidgets import QApplication
qt = QApplication.instance() or QApplication([])
import app as A

A.MainWindow._start_probe = lambda self: None   # keep the torch probe out of tests
root = Path(tempfile.mkdtemp())
A.paths.OUTPUTS_DIR = root / "outputs"; A.paths.OUTPUTS_DIR.mkdir()
A.paths.SETTINGS_FILE = root / "config/settings.ini"
A.paths.LOG_FILE = root / "app.log"
A.setup_logging()
w = A.MainWindow()
# ... assertions ...
w.close(); qt.processEvents()
sys.exit(0)
```

Hard-won details:

- **Disable the hardware probe in tests** (`MainWindow._start_probe`). It
  imports torch on a `QThread`; on Windows that can hold the loader lock and
  make `os._exit()` / `ExitProcess` hang after the checks already printed. Close
  the window and exit normally instead.
- **Patch `A.paths.*` before `A.setup_logging()`/`MainWindow()`** so tests never
  read or write the real `config/settings.ini`, `outputs/` or `logs/`.
- **`isVisible()` is False when the top-level window was never shown.** For a
  child widget in an offscreen test use `isHidden()` (or show the window).
- **Fresh window, empty log view.** Startup lines are logged before the view
  exists; only lines logged after construction appear in `log_view`.
- **Running jobs are not queue rows.** Only waiting work lives in `self.queue`;
  the active job is `self._active_job`.
- **PySide6 6.12 enums** often need `.value` when compared with ints.
  `QPushButton` has no `setAutoRaise` — that is `QToolButton` only.
- **Use `-u` and flush before exiting.** Capture the status without losing it
  (`PIPESTATUS`/direct redirect). Don't pipe installs into `tail`.
- **Write harnesses to `_dev/*.py`** rather than `python -c`: long one-liners
  truncate and apostrophes in strings break shell quoting.
- **Source files are CRLF.** Editing tools may normalise; preserve CRLF in
  source files. When comparing a zip against source, normalise
  `\r\n` → `\n` because `read_text()` does.
- **Model generation is not testable here.** No NVIDIA GPU and no SA3 weights.
  Say "compiled/constructed offscreen" — never claim a real render, and never
  claim the GUI was visually inspected unless it was.

## Logging

- `_log_line(message, level=logging.INFO)` writes to the rotating
  `logs/app.log` and the activity panel; it must never raise.
- The panel keeps its own history, `self._log_entries` (last 2000
  `(level, line)` tuples). The **Warnings only** checkbox (`log_filter`) calls
  `_set_log_filter`, which redraws the view from that history.
- The filter affects only the panel: the file log always records every level.
- The launchers write `logs/bootstrap.log`: interpreter/venv decisions, hardware
  choice, pip `--log` output, and launch/failure. It is separate from `app.log`
  and must stay out of the delivery.
- `log_warnings_only` is persisted in `config/settings.ini`; Factory reset turns it
  off. If you add panel state, persist it the same way (`log_open` is the
  pattern) and put it in `_save_settings` and `_restore_settings`.

## Settings keys

`geometry`, `prompt`, `negative`, `duration`, `tracks`, `random_seed`,
`volume`, `device`, `details_open`, `advanced_open`, `log_open`,
`log_warnings_only`, `formats`, `results_widths`, `queue_widths`, `queue`.
Widths are JSON with a 38–2000 clamp; a saved row from a different column set
must be ignored, not applied to the wrong columns. There is deliberately no
`emergency_open` key: the Emergency section and its Factory reset button must
always start collapsed.

## Documentation conventions

- Module-level docstrings explain each file's responsibility.
- Document non-trivial functions/methods (why and non-obvious behaviour), not
  every line. Pure Qt overrides and one-line properties are intentionally left
  undocumented.
- Comments explain decisions and traps, not restatements of the code.
- When a user-visible behaviour changes, update the README in the same change.

## Release checklist

1. Make the change; bump `APP_VERSION` in `code/app.py` (patch for fixes/features).
2. Run the compile check and an offscreen harness covering the changed UI or
   logic; run `bash -n run.sh` if the shell scripts changed.
3. `rm -f delivery/*-v<version>-*.zip` only when deliberately rebuilding that
   version; otherwise never delete old artifacts.
4. `.venv/Scripts/python.exe _dev/build_delivery.py`
5. Verify the newest zip: **12 entries**, `APP_VERSION` matches, no forbidden
   names, and the files contain the change (read them back from the zip, do not
   assume the staging folder matches).
6. Check README claims that go stale: zip size, test counts, column counts,
   setting names. Fix them before finishing.
7. Tidy dev leftovers (`__pycache__`, `_dev/_tmp`, temp harnesses, stray
   `config/settings.ini`/`config/.hf_token`); leave `delivery/` artifacts in place.

The builder reuses a byte-identical zip for the same version and adds a new zip
when content changed. Never hand-edit `delivery/SimpleSoundGenerator/`; it is a
generated copy.

## Known limitations

- Real generation and model loading are unverified in this environment (CPU
  only, SA3 weights absent).
- `run.sh` is syntax-checked only on Windows.
- The GUI has been exercised offscreen, not inspected on a real display.

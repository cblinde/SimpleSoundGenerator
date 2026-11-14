#!/usr/bin/env bash
# ===========================================================================
#  Simple Sound Generator - macOS / Linux bootstrap + launcher
#
#  Creates a fully self-contained runtime next to this script:
#    1. finds a usable Python (system or .python portable, 3.10 - 3.13);
#       if none exists, downloads python-build-standalone into .python/
#    2. creates / reuses .venv with that interpreter
#    3. installs PyTorch for the detected hardware (newest usable CUDA channel,
#       else the CPU wheel) and then requirements.txt
#    4. launches code/app.py - no admin rights, no PATH changes
#
#  Usage:  ./run.sh                 bootstrap + launch
#          ./run.sh --setup-only    bootstrap only
#          ./run.sh --repair        force a full dependency re-check
# ===========================================================================
set -u

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_ROOT"
BOOT_LOG="$APP_ROOT/logs/bootstrap.log"
BOOT_LOG_MAX=2000000

CODE_DIR="$APP_ROOT/code"
VENV_DIR="$APP_ROOT/.venv"
VENV_PY="$VENV_DIR/bin/python"
PORTABLE_DIR="$APP_ROOT/.python"
PORTABLE_PY="$PORTABLE_DIR/python/bin/python3"
REQ="$CODE_DIR/requirements.txt"
PY_VERSION="3.12.16"
PY_RELEASE="20261026"
SETUP_ONLY=0
DEBUG=0
SKIP_FLASH=0
FORCE_FLASH=0
FORCE_SETUP=0

usage() {
  echo "Usage: ./run.sh [--setup-only] [--debug] [--flash-attn] [--no-flash-attn] [--repair]"
  echo "  --setup-only   prepare .python / .venv / requirements.txt and exit"
  echo "  --debug        foreground run with visible output (the default on POSIX)"
  echo "  --flash-attn   retry the optional FlashAttention 2 wheel (automatic by default)"
  echo "  --no-flash-attn  skip FlashAttention 2 entirely"
  echo "  --repair       re-verify every dependency (after pip changes or a broken env)"
  echo "  --version      print the app version and exit (no bootstrap)"
  echo "  (any other arguments are passed through to app.py)"
  echo
  show_version
}

show_version() {
  # Read APP_VERSION straight out of code/app.py: no venv check, no downloads.
  ver=$(sed -n 's/^APP_VERSION[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' \
    "$CODE_DIR/app.py" | head -1)
  echo "Simple Sound Generator v${ver:-unknown}"
}

APP_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --setup-only) SETUP_ONLY=1 ;;
    --debug) DEBUG=1 ;;
    --flash-attn) FORCE_FLASH=1 ;;
    --no-flash-attn) SKIP_FLASH=1 ;;
    --repair) FORCE_SETUP=1 ;;
    --help|-h) usage; exit 0 ;;
    --version|-V) show_version; exit 0 ;;
    *) APP_ARGS+=("$arg") ;;
  esac
done

echo "============================================================"
echo " Simple Sound Generator - portable bootstrap"
echo " root: $APP_ROOT"
echo "============================================================"
echo

mkdir -p "$APP_ROOT/config" "$APP_ROOT/outputs" "$APP_ROOT/logs" || exit 1

# Keep the launcher's own record even when the terminal is gone. pip also
# appends its detailed install log to the same file; tee mirrors the live
# console output there when available.
rotate_boot_log() {
  [ -f "$BOOT_LOG" ] || return 0
  size="$(wc -c < "$BOOT_LOG" 2>/dev/null || echo 0)"
  if [ "${size:-0}" -gt "$BOOT_LOG_MAX" ]; then
    mv -f "$BOOT_LOG" "$BOOT_LOG.1" 2>/dev/null || true
  fi
}
log_boot() {
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$BOOT_LOG" 2>/dev/null || true
}
rotate_boot_log
log_boot "=== Simple Sound Generator launcher start ==="
log_boot "Root: $APP_ROOT"
if command -v tee >/dev/null 2>&1; then
  exec > >(tee -a "$BOOT_LOG") 2>&1
fi


version_ok() {
  "$1" -c 'import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)' >/dev/null 2>&1
}

# ---------------------------------------------------------------- 1. python
PY=""
VENV_OK=0
if [ -x "$VENV_PY" ] && "$VENV_PY" -c "import sys" >/dev/null 2>&1; then
  PY="$VENV_PY"; VENV_OK=1
elif [ -x "$PORTABLE_PY" ] && version_ok "$PORTABLE_PY"; then
  PY="$PORTABLE_PY"
else
  for candidate in python3.12 python3.11 python3.13 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      resolved="$(command -v "$candidate")"
      if version_ok "$resolved"; then PY="$resolved"; break; fi
    fi
  done
fi

detect_triple() {
  os="$(uname -s)"; arch="$(uname -m)"
  case "$os" in
    Linux)
      case "$arch" in
        x86_64|amd64) echo "x86_64-unknown-linux-gnu" ;;
        aarch64|arm64) echo "aarch64-unknown-linux-gnu" ;;
        *) echo "" ;;
      esac ;;
    Darwin)
      case "$arch" in
        arm64) echo "aarch64-apple-darwin" ;;
        x86_64) echo "x86_64-apple-darwin" ;;
        *) echo "" ;;
      esac ;;
    *) echo "" ;;
  esac
}

download_portable_python() {
  triple="$(detect_triple)"
  if [ -z "$triple" ]; then
    echo "ERROR: no portable Python build for $(uname -s)/$(uname -m)."
    echo "       Install Python 3.10-3.13 and re-run."
    return 1
  fi
  url="${PYTHON_STANDALONE_URL:-https://github.com/astral-sh/python-build-standalone/releases/download/$PY_RELEASE/cpython-$PY_VERSION+$PY_RELEASE-$triple-install_only.tar.gz}"
  echo "      No usable Python 3.10-3.13 found - downloading portable CPython $PY_VERSION ..."
  echo "      $url"
  mkdir -p "$PORTABLE_DIR" || return 1
  tarball="$PORTABLE_DIR/python-portable.tar.gz"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 3 -o "$tarball" "$url" || return 1
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$tarball" "$url" || return 1
  else
    echo "ERROR: neither curl nor wget is available to download Python."
    return 1
  fi
  tar -xzf "$tarball" -C "$PORTABLE_DIR" || return 1
  rm -f "$tarball"
  if [ -x "$PORTABLE_DIR/python/bin/python3" ]; then
    PORTABLE_PY="$PORTABLE_DIR/python/bin/python3"
  else
    PORTABLE_PY="$(find "$PORTABLE_DIR" -name python3 -type f -perm -u+x 2>/dev/null | head -1)"
  fi
  [ -n "$PORTABLE_PY" ] || { echo "ERROR: python3 not found inside .python"; return 1; }
  chmod +x "$PORTABLE_PY" 2>/dev/null || true
  # macOS quarantines downloaded binaries; harmless if it fails.
  if command -v xattr >/dev/null 2>&1; then
    xattr -dr com.apple.quarantine "$PORTABLE_DIR" >/dev/null 2>&1 || true
  fi
  PY="$PORTABLE_PY"
  echo "      Portable Python ready: $PY"
}

if [ -z "$PY" ]; then
  download_portable_python || exit 1
fi

echo "[1/4] Interpreter: $PY"
"$PY" -c 'import sys; print("       Python " + sys.version.split()[0])' || exit 1

# ------------------------------------------------------------------ 2. venv
# A virtualenv records the absolute path of the interpreter that created it, so a
# folder copied from another machine (or another OS) can exist but be unusable.
# Only reuse it when its python actually starts here; otherwise rebuild it.
if [ "$VENV_OK" = "1" ]; then
  echo "[2/4] Reusing existing .venv"
else
  if [ -d "$VENV_DIR" ]; then
    echo "      Existing .venv cannot run with this interpreter - recreating it."
    rm -rf "$VENV_DIR"
  fi
  echo "[2/4] Creating virtual environment .venv ..."
  "$PY" -m venv "$VENV_DIR" || { echo "ERROR: could not create .venv"; exit 1; }
  if [ ! -x "$VENV_PY" ]; then
    # Happens when a Windows interpreter is used from a POSIX shell (Git Bash):
    # it creates .venv/Scripts/python.exe instead of .venv/bin/python.
    echo "ERROR: $VENV_DIR/bin/python was not created by $PY"
    echo "       This script targets macOS/Linux - on Windows run run.bat instead."
    exit 1
  fi
  "$VENV_PY" -m pip install --disable-pip-version-check --log "$BOOT_LOG" --quiet --upgrade pip
fi
PY="$VENV_PY"

# --------------------------------------------------------------- 3 packages
# A repeat launch must not pay pip's resolver plus a git fetch of the SA3
# runtime - that is minutes, while the GUI itself is ready in about two
# seconds. The requirements.txt that was last verified is kept in
# .venv/.deps-req; when it still matches and the key packages resolve, the
# whole install block is skipped. --repair, an edited requirements.txt or a
# missing package all fall through to the full check.
SKIP_SETUP=0
if [ "$VENV_OK" = "1" ] && [ "$FORCE_SETUP" != "1" ] && [ -f "$VENV_DIR/.deps-req" ] && [ -f "$REQ" ] && \
   cmp -s "$VENV_DIR/.deps-req" "$REQ" && \
   "$PY" -c 'import importlib.util,sys; sys.exit(0 if all(importlib.util.find_spec(m) for m in ("PySide6","torch","torchaudio","diffusers","transformers","huggingface_hub")) else 1)' >/dev/null 2>&1; then
  SKIP_SETUP=1
fi

if [ "$SKIP_SETUP" = "1" ]; then
  echo "[3/4] Requirements unchanged - skipping pip, PyTorch and FlashAttention checks"
  if [ "$FORCE_FLASH" = "1" ]; then
    echo "      --flash-attn given - re-running the FlashAttention probe"
    "$PY" "$CODE_DIR/flash_attn_setup.py" --force || true
  fi
else
# PyTorch must match the machine, so it is installed *before* requirements.txt.
# It is pinned to 2.7.1 together with torchaudio: the optional Stable Audio 3
# runtime imports torchaudio, and torchaudio has no build past 2.11.1, so the
# newest torch (2.15) cannot be used. cu124 never got 2.7.1, hence cu128/cu126
# only. The wheel tag (2.7.1+cu128) still satisfies requirements.txt's ==2.7.1,
# so the CUDA build is not replaced by the CPU one afterwards.
torch_usable() {
  "$PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1
}

if command -v nvidia-smi >/dev/null 2>&1; then
  if torch_usable; then
    echo "      CUDA-enabled PyTorch already installed."
  elif [ -f "$VENV_DIR/.torch-cpu-only" ]; then
    echo "      No runnable CUDA wheel on this machine - using CPU PyTorch."
  else
    echo "      NVIDIA driver detected - installing CUDA PyTorch (falls back to CPU)."
    torch_ok=0
    # Newest channel first: Blackwell (RTX 50xx) needs cu128 kernels. CUDA has
    # to initialise before a wheel is accepted - a channel newer than the
    # installed driver installs cleanly but cannot run.
    # torch and torchaudio must come from the same channel and version pair.
    for channel in cu128 cu126; do
      if ! "$PY" -m pip install --disable-pip-version-check --log "$BOOT_LOG" \
          --index-url "https://download.pytorch.org/whl/$channel" "torch==2.7.1" "torchaudio==2.7.1"; then
        continue
      fi
      if torch_usable; then
        echo "      CUDA PyTorch installed from channel $channel."
        rm -f "$VENV_DIR/.torch-cpu-only"
        torch_ok=1
        break
      fi
      echo "      $channel wheel installed but CUDA did not initialise - trying a lower channel."
    done
    if [ "$torch_ok" != "1" ]; then
      echo "      WARNING: no CUDA wheel could be activated - continuing with CPU PyTorch."
      # Remember the failure so later launches do not re-download multi-GB wheels.
      : > "$VENV_DIR/.torch-cpu-only"
    fi
  fi
fi

if ! "$PY" -c 'import torch' >/dev/null 2>&1; then
  if [ "$(uname -s)" = "Darwin" ]; then
    # The cpu index has no macOS wheels; PyPI builds use Metal (MPS) or CPU.
    echo "      Installing PyTorch from PyPI (macOS MPS/CPU build)."
  else
    echo "      Installing CPU PyTorch from download.pytorch.org/whl/cpu ..."
    "$PY" -m pip install --disable-pip-version-check --log "$BOOT_LOG" \
        --index-url https://download.pytorch.org/whl/cpu "torch==2.7.1" "torchaudio==2.7.1" || \
      echo "      WARNING: CPU wheel index unavailable - requirements.txt will install the PyPI build."
  fi
fi

echo "[3/4] Installing / verifying requirements.txt ..."
"$PY" -m pip install --disable-pip-version-check --log "$BOOT_LOG" -r "$REQ" || {
  echo "ERROR: dependency installation failed. Check network/proxy settings and re-run."
  exit 1
}

# Optional Stable Audio 3 runtime (not on PyPI). Installed from the GitHub
# source archive, so no git executable is required. --no-deps protects the
# torch/torchaudio pair installed above; its own deps are pinned in
# requirements.txt. Failure only disables SA3 checkpoints, so keep going.
echo "      Installing / verifying the optional Stable Audio 3 runtime ..."
"$PY" -m pip install --disable-pip-version-check --log "$BOOT_LOG" --no-deps --quiet \
    "https://github.com/Stability-AI/stable-audio-3/archive/refs/heads/main.zip" || \
  echo "      WARNING: Stable Audio 3 runtime not installed (network) - SA3 checkpoints will not load."

# Optional FlashAttention 2: a wheel only works for the exact torch + CUDA +
# Python build, so resolve a prebuilt one instead of compiling. Speed only.
if [ "$SKIP_FLASH" = "1" ]; then
  echo "      FlashAttention 2 skipped (--no-flash-attn)."
else
  if [ "$FORCE_FLASH" = "1" ]; then
    "$PY" "$CODE_DIR/flash_attn_setup.py" --force || true
  else
    "$PY" "$CODE_DIR/flash_attn_setup.py" || true
  fi
fi
"$PY" -c 'import torch; print("       torch " + torch.__version__ + " | CUDA: " + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no (CPU)"))' 2>/dev/null || \
  echo "      WARNING: torch did not import cleanly - see messages above."

# Remember this exact requirements.txt so the next start can skip pip.
cp "$REQ" "$VENV_DIR/.deps-req" 2>/dev/null || true
fi

if [ "$SETUP_ONLY" = "1" ]; then
  echo
  echo "Setup complete. Start the app with: ./run.sh"
  exit 0
fi

# ------------------------------------------------------------------- 4. app
echo "[4/4] Launching code/app.py ..."
log_boot "Launching code/app.py"
echo
exec "$PY" "$CODE_DIR/app.py" ${APP_ARGS[@]+"${APP_ARGS[@]}"}

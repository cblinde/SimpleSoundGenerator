"""
paths.py
========

Portable path and environment bootstrap.

This module **must be imported before** ``torch``, ``diffusers``,
``transformers`` or ``huggingface_hub`` so that every cache, download and
weight file lands inside the application folder instead of the user profile.

Layout (everything self-contained under ``APP_ROOT``)::

    APP_ROOT/
        run.bat run.sh  README.md  LICENSE
        code/       app.py, audio_engine.py, model_manager.py,
                    flash_attn_setup.py, paths.py, requirements.txt
        config/     models.json, settings.ini, .hf_token, models/
        outputs/    generated audio + JSON sidecars
        logs/       rotating text logs
        .python/    portable CPython (created by the bootstrap)
        .venv/      virtual environment (created by the bootstrap)

Credentials
-----------
The Hugging Face token lives in ``config/.hf_token`` (kept inside the app
folder, never logged, never echoed). :func:`_apply_token_env` mirrors it into
``HF_TOKEN`` and ``HUGGING_FACE_HUB_TOKEN``, because ``huggingface_hub`` and
``transformers`` authenticate from the environment only: the downloader's own
``token=`` argument does not cover the Stable Audio 3 text encoder, which is
fetched through ``transformers`` and otherwise answers 401 on a gated repo.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

APP_ROOT: Path = Path(__file__).resolve().parent.parent
CODE_DIR: Path = Path(__file__).resolve().parent
CONFIG_DIR: Path = APP_ROOT / "config"

MODELS_DIR: Path = CONFIG_DIR / "models"
OUTPUTS_DIR: Path = APP_ROOT / "outputs"
LOGS_DIR: Path = APP_ROOT / "logs"

SETTINGS_FILE: Path = CONFIG_DIR / "settings.ini"
REGISTRY_FILE: Path = CONFIG_DIR / "models.json"
HF_TOKEN_FILE: Path = CONFIG_DIR / ".hf_token"
LOG_FILE: Path = LOGS_DIR / "app.log"

#: Directories inside ``config/models/`` that belong to the cache, not to a model.
CACHE_DIR_NAMES = {"hub", "torch", "xet", "xdg", "datasets", "assets", ".cache"}

_configured = False


def ensure_directories() -> None:
    """Create the code/config/data tree, migrating a pre-1.7 layout once."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_layout()
    for directory in (MODELS_DIR, OUTPUTS_DIR, LOGS_DIR, MODELS_DIR / "hub", MODELS_DIR / "torch"):
        directory.mkdir(parents=True, exist_ok=True)


def _migrate_legacy_layout() -> None:
    """Move settings, token, registry and weights from the pre-1.7 layout.

    Earlier builds kept these next to the launchers. Upgrading in place must
    not lose a downloaded model or a typed token, so each item moves into
    ``config/`` when the new location does not exist yet. Best effort: a
    read-only or cross-device move must never block startup.
    """
    for legacy_name, destination in (
        ("settings.ini", SETTINGS_FILE),
        (".hf_token", HF_TOKEN_FILE),
        ("models.json", REGISTRY_FILE),
    ):
        legacy = APP_ROOT / legacy_name
        if legacy.is_file() and not destination.exists():
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                legacy.replace(destination)
            except OSError:
                try:
                    shutil.move(str(legacy), str(destination))
                except (OSError, shutil.Error):
                    pass
    legacy_models = APP_ROOT / "models"
    if legacy_models.is_dir() and not MODELS_DIR.exists():
        try:
            legacy_models.replace(MODELS_DIR)
        except OSError:
            try:
                shutil.move(str(legacy_models), str(MODELS_DIR))
            except (OSError, shutil.Error):
                pass


def bootstrap_env() -> None:
    """Pin every ML cache to ``config/models``. Safe to call repeatedly.

    Called from ``code/app.py`` before the heavy imports and again from
    ``audio_engine`` / ``model_manager`` so the modules also work when used
    from a script or a test harness.
    """
    global _configured
    ensure_directories()
    if _configured:
        return

    models = str(MODELS_DIR)
    os.environ.update(
        {
            # Hugging Face: downloads, blobs and the auth token file.
            "HF_HOME": models,
            "HF_HUB_CACHE": os.path.join(models, "hub"),
            "HF_XET_CACHE": os.path.join(models, "xet"),
            # Older transformers/diffusers honour these two names.
            "TRANSFORMERS_CACHE": os.path.join(models, "hub"),
            "DIFFUSERS_CACHE": os.path.join(models, "hub"),
            # torch.hub / torchvision weights.
            "TORCH_HOME": os.path.join(models, "torch"),
            # Anything that falls back to the XDG cache (Linux desktop files).
            "XDG_CACHE_HOME": os.path.join(models, "xdg"),
            # Quality-of-life / determinism flags.
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "TOKENIZERS_PARALLELISM": "false",
            # Apple Silicon: fall back to CPU for unimplemented ops instead of
            # crashing deep inside a kernel.
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        }
    )
    _apply_token_env(read_hf_token())
    _configured = True


def _apply_token_env(token: str) -> None:
    """Expose the app's token to huggingface_hub / transformers / diffusers.

    Those libraries authenticate from ``HF_TOKEN`` (or the legacy
    ``HUGGING_FACE_HUB_TOKEN``) in the environment. ``config/.hf_token`` is our own
    convention: a token typed into the app used to reach only
    ``snapshot_download(token=...)``, so anything the model runtime fetched by
    itself - notably the T5Gemma text encoder inside a gated checkpoint -
    went out unauthenticated and came back 401 on a perfectly valid token.
    """
    token = (token or "").strip()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    else:
        os.environ.pop("HF_TOKEN", None)
        os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)


def read_hf_token() -> str:
    """Return the Hugging Face token from ``HF_TOKEN`` or ``config/.hf_token``."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    if token.strip():
        return token.strip()
    try:
        if HF_TOKEN_FILE.exists():
            return HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    return ""


def write_hf_token(token: str) -> None:
    """Persist a token locally (kept inside the app folder, never logged)."""
    token = (token or "").strip()
    try:
        if token:
            HF_TOKEN_FILE.write_text(token + "\n", encoding="utf-8")
            try:  # best effort: owner-only on POSIX
                os.chmod(HF_TOKEN_FILE, 0o600)
            except OSError:
                pass
        elif HF_TOKEN_FILE.exists():
            HF_TOKEN_FILE.unlink()
    except OSError:
        pass
    _apply_token_env(token)


def human_size(num_bytes: float) -> str:
    """Format a byte count as a compact human readable string."""
    try:
        size = float(num_bytes)
    except (TypeError, ValueError):
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

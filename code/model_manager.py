"""
model_manager.py
================

Local model registry, storage scanning and weight downloading.

* The registry comes from ``config/models.json`` (editable, shipped with the app).
  If the file is missing or unreadable the in-code :data:`DEFAULT_MODELS` are
  used and a fresh ``config/models.json`` is written.
* Every model is stored in exactly one place: ``config/models/<model id>/``.
  Caches (``HF_HOME``, ``TORCH_HOME``) are redirected there too by ``paths``.
* Folders dropped into ``config/models`` by hand are auto-discovered and appear in
  the dropdown, so a locally converted Stable Audio 3 checkpoint works without
  editing any JSON.

Nothing in this module imports torch, so it stays cheap to import from the GUI
process.

Downloading
-----------
:meth:`ModelManager.download` wraps ``snapshot_download`` with a cancellable
tqdm subclass: the file in flight stays as ``*.incomplete`` for the next attempt,
files that already completed are skipped, and ``.ready`` is written only after a
transfer that was not cancelled. The token comes from
``paths.read_hf_token()`` (``HF_TOKEN`` first, then ``config/.hf_token``), and
``paths`` additionally exports it as ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN``
because the Stable Audio 3 text encoder fetches through ``transformers``, which
reads only the environment.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable, Dict, List, Optional

import paths

LogFn = Callable[[str], None]
ProgressFn = Callable[[str], None]

DEFAULT_IGNORE = ["*.ckpt", "*.png", "*.jpg", "*.jpeg", "*.mp4", "*.gif"]


class ModelError(RuntimeError):
    """Base class for registry/transport problems."""


class ModelAccessError(ModelError):
    """The repository is gated (or the token is missing / invalid)."""


class DownloadCancelled(Exception):
    """The user aborted the transfer.

    Not a :class:`ModelError`: nothing went wrong. The bytes already received
    stay in the ``*.incomplete`` files inside the model folder, and
    ``huggingface_hub`` resumes them with an HTTP range request on the next
    call, so cancelling costs only the part of the current file in flight.
    """


# ---------------------------------------------------------------------------
# Built-in registry (mirrors models.json; used as a fallback)
# ---------------------------------------------------------------------------
DEFAULT_MODELS: List[dict] = [
    {
        "id": "stable-audio-3-small-sfx",
        "name": "Stable Audio 3 Small \u00b7 SFX",
        "repo_id": "stabilityai/stable-audio-3-small-sfx",
        "architecture": (
            "Stable Audio 3 \u00b7 rectified-flow DiT + SAME-Small autoencoder + T5Gemma, "
            "official stable-audio-3 runtime"
        ),
        "backend": "stable_audio_3",
        "use_case": "One-shot transients, impacts, foley and UI sounds \u2014 CPU friendly",
        "summary": (
            "Sound-effects specialised sibling of Stable Audio 3 Small: 433M parameters, "
            "8 distilled steps, 44.1 kHz stereo, up to 120 s. Short prompts of 1\u201310 s "
            "give tight, transient-rich hits."
        ),
        "highlights": [
            "Tuned for short transients and foley rather than musical structure",
            "8 distilled steps \u2014 seconds per clip on a modest GPU",
            "Runs on CPU or CUDA",
            "Ships its own T5Gemma text encoder (~3.5 GB total)",
        ],
        "max_duration": 120.0,
        "sample_rate": 44100,
        "duration_arg": "duration",
        "default_steps": 8,
        "default_guidance": 1.0,
        "supports_negative_prompt": False,
        "supports_guidance": False,
        "gated": True,
        "approx_size_gb": 3.5,
        "min_vram_gb": 0.0,
        "required_files": ["model.safetensors", "model_config.json"],
        "ignore_patterns": ["*.png"],
        "notes": (
            "Needs the official runtime: pip install "
            "\"https://github.com/Stability-AI/stable-audio-3/archive/refs/heads/main.zip\" (see README)."
        ),
    },
    {
        "id": "stable-audio-3-small-music",
        "name": "Stable Audio 3 Small \u00b7 Music",
        "repo_id": "stabilityai/stable-audio-3-small-music",
        "architecture": (
            "Stable Audio 3 \u00b7 rectified-flow DiT + SAME-Small autoencoder + T5Gemma, "
            "official stable-audio-3 runtime"
        ),
        "backend": "stable_audio_3",
        "use_case": "Background music beds, loops and 1\u20132 minute cues \u2014 CPU friendly",
        "summary": (
            "433M distilled checkpoint from Stability AI's Stable Audio 3 family. Eight "
            "ping-pong denoising steps with guidance baked into the weights, 44.1 kHz "
            "stereo, up to 120 s. Small enough to run on CPU-only laptops."
        ),
        "highlights": [
            "44.1 kHz stereo, up to 120 s per clip",
            "8 distilled steps, CFG baked in (guidance 1.0, no negative prompt)",
            "Runs on CPU or CUDA (no FlashAttention required)",
            "Ships its own T5Gemma text encoder (~3.5 GB total)",
        ],
        "max_duration": 120.0,
        "sample_rate": 44100,
        "duration_arg": "duration",
        "default_steps": 8,
        "default_guidance": 1.0,
        "supports_negative_prompt": False,
        "supports_guidance": False,
        "gated": True,
        "approx_size_gb": 3.5,
        "min_vram_gb": 0.0,
        "required_files": ["model.safetensors", "model_config.json"],
        "ignore_patterns": ["*.png"],
        "notes": (
            "Needs the official runtime: pip install "
            "\"https://github.com/Stability-AI/stable-audio-3/archive/refs/heads/main.zip\" (see README)."
        ),
    },
    {
        "id": "stable-audio-3-medium",
        "name": "Stable Audio 3 Medium",
        "repo_id": "stabilityai/stable-audio-3-medium",
        "architecture": (
            "Stable Audio 3 \u00b7 rectified-flow DiT (1.4B) + SAME-Large autoencoder + "
            "T5Gemma, official stable-audio-3 runtime"
        ),
        "backend": "stable_audio_3",
        "use_case": "Long-form music cues and complex sound design \u2014 up to 380 s, GPU",
        "summary": (
            "The high-quality Stable Audio 3 checkpoint: 1.4B distilled DiT with the "
            "SAME-Large autoencoder, 8 ping-pong steps, 44.1 kHz stereo and up to 380 s "
            "per generation. Expects a CUDA GPU and FlashAttention 2."
        ),
        "highlights": [
            "44.1 kHz stereo, up to 380 s \u2014 best long-form music quality",
            "8 distilled steps; guidance baked in (no CFG / negative prompt)",
            "Needs CUDA + FlashAttention 2 for the documented speed",
            "Largest download in the registry (~10.4 GB incl. text encoder)",
        ],
        "max_duration": 380.0,
        "sample_rate": 44100,
        "duration_arg": "duration",
        "default_steps": 8,
        "default_guidance": 1.0,
        "supports_negative_prompt": False,
        "supports_guidance": False,
        "gated": True,
        "approx_size_gb": 10.4,
        "min_vram_gb": 12.0,
        "required_files": ["model.safetensors", "model_config.json"],
        "ignore_patterns": ["*.png"],
        "notes": (
            "Needs the official runtime plus FlashAttention 2 on GPU; on CPU it will not "
            "fit comfortably. Prefer the Small checkpoints without an NVIDIA GPU."
        ),
    },
    {
        "id": "stable-audio-open-1.0",
        "name": "Stable Audio Open 1.0",
        "repo_id": "stabilityai/stable-audio-open-1.0",
        "architecture": "Stable Audio Open 1.0 \u00b7 latent DiT + T5, diffusers StableAudioPipeline",
        "backend": "diffusers",
        "use_case": "Short SFX, foley, loops, ambience and stems \u2014 up to 47 s per clip",
        "summary": (
            "Stability AI's open text-to-audio model. A T5 text encoder conditions a latent "
            "diffusion transformer that decodes to 44.1 kHz stereo. Excellent at foley, "
            "textures and short musical loops; it does not model long song structure."
        ),
        "highlights": [
            "44.1 kHz stereo, variable length up to 47 s",
            "Negative prompt + CFG supported (recommended: 100 steps, CFG 7.0)",
            "CUDA: bf16/fp16; CPU: fp32 fallback (slow \u2014 minutes per clip)",
            "Gated repo: accept the licence on huggingface.co and paste an HF token",
        ],
        "max_duration": 47.0,
        "sample_rate": 44100,
        "duration_arg": "audio_end_in_s",
        "default_steps": 100,
        "default_guidance": 7.0,
        "supports_negative_prompt": True,
        "supports_guidance": True,
        "gated": True,
        "approx_size_gb": 4.4,
        "min_vram_gb": 6.0,
        "required_files": [
            "model_index.json",
            "transformer",
            "vae",
            "text_encoder",
            "projection_model",
        ],
        "allow_patterns": [
            "*.json",
            "tokenizer/*",
            "text_encoder/*",
            "transformer/*",
            "vae/*",
            "projection_model/*",
            "scheduler/*",
        ],
        "ignore_patterns": ["*.ckpt"],
        "notes": (
            "The root model.safetensors / model.ckpt are the original stable-audio-tools "
            "checkpoints; they are skipped to save ~4 GB and are not used by diffusers."
        ),
    }
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class ModelStatus:
    """Result of scanning ``config/models/<id>/``."""

    state: str = "missing"  # ready | partial | missing | downloading
    size_bytes: int = 0
    detail: str = ""

    @property
    def ready(self) -> bool:
        return self.state == "ready"


@dataclass
class ModelSpec:
    """Description of one selectable model."""

    id: str
    name: str
    repo_id: str
    architecture: str = ""
    backend: str = "diffusers"
    use_case: str = ""
    summary: str = ""
    highlights: List[str] = field(default_factory=list)
    max_duration: float = 47.0
    sample_rate: int = 44100
    duration_arg: str = "audio_end_in_s"
    default_steps: int = 100
    default_guidance: float = 7.0
    supports_negative_prompt: bool = True
    supports_guidance: bool = True
    gated: bool = True
    approx_size_gb: float = 0.0
    min_vram_gb: float = 0.0
    required_files: List[str] = field(default_factory=list)
    allow_patterns: Optional[List[str]] = None
    ignore_patterns: List[str] = field(default_factory=lambda: list(DEFAULT_IGNORE))
    notes: str = ""
    local: bool = False

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelSpec":
        """Build a spec from one models.json entry, filling in defaults."""
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in (data or {}).items() if k in known}
        clean.setdefault("id", clean.get("repo_id") or "model")
        clean.setdefault("name", clean["id"])
        clean.setdefault("repo_id", "")
        return cls(**clean)

    @property
    def short_label(self) -> str:
        """Name without the family prefix, for compact UI text."""
        return f"{self.name}"

    @property
    def supports_offline_load(self) -> bool:
        """True when every required file is already on disk."""
        return bool(self.repo_id) or self.local


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------
class ModelManager:
    """Registry + local storage for model weights."""

    def __init__(
        self,
        root: Optional[Path] = None,
        registry_file: Optional[Path] = None,
        log: Optional[LogFn] = None,
    ) -> None:
        """Load the registry and remember the application root."""
        paths.bootstrap_env()
        if root is not None:
            self.root = Path(root)
            self.models_dir = self.root / "models"
            fallback_registry = self.root / "models.json"
        else:
            self.root = paths.APP_ROOT
            self.models_dir = paths.MODELS_DIR
            fallback_registry = paths.REGISTRY_FILE
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = Path(registry_file) if registry_file else fallback_registry
        self._log: LogFn = log or (lambda *_: None)
        self._specs: Optional[List[ModelSpec]] = None

    # -- registry ----------------------------------------------------------
    def specs(self, refresh: bool = False) -> List[ModelSpec]:
        """Return every selectable model (registry + auto-discovered folders)."""
        if self._specs is None or refresh:
            self._specs = self._load_specs()
        return list(self._specs)

    def get(self, model_id: str) -> Optional[ModelSpec]:
        """Return the spec with this id, or None."""
        for spec in self.specs():
            if spec.id == model_id:
                return spec
        return None

    def _load_specs(self) -> List[ModelSpec]:
        """Merge shipped, extra and auto-discovered local models."""
        data: Dict = {}
        if self.registry_file.exists():
            try:
                data = json.loads(self.registry_file.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("models.json must contain a JSON object")
            except Exception as exc:  # noqa: BLE001 - never let a bad file block startup
                self._log(f"models.json is unreadable ({exc}); falling back to built-in registry.")
                data = {}
        else:
            data = {}
            try:
                payload = {
                    "_comment": (
                        "Editable model registry. 'models' replaces the built-in list; "
                        "'extra_models' is appended. Required keys: id, name, repo_id."
                    ),
                    "models": DEFAULT_MODELS,
                    "extra_models": [],
                }
                self.registry_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            except OSError:
                pass

        entries: List[dict] = []
        entries.extend(data.get("models") or DEFAULT_MODELS)
        entries.extend(data.get("extra_models") or [])

        specs: List[ModelSpec] = []
        seen: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                spec = ModelSpec.from_dict(entry)
            except Exception as exc:  # noqa: BLE001
                self._log(f"Skipping malformed model entry {entry!r}: {exc}")
                continue
            if spec.id in seen:
                continue
            seen.add(spec.id)
            specs.append(spec)

        specs.extend(self._discover_local(seen))
        return specs

    def _discover_local(self, known_ids: set[str]) -> List[ModelSpec]:
        """Register model folders that exist in config/models but not in the registry."""
        found: List[ModelSpec] = []
        try:
            children = sorted(p for p in self.models_dir.iterdir() if p.is_dir())
        except OSError:
            return found
        for child in children:
            if child.name in known_ids or child.name in paths.CACHE_DIR_NAMES or child.name.startswith("."):
                continue
            is_diffusers = (child / "model_index.json").exists()
            is_bundle = (child / "model_config.json").exists() or (child / "model.safetensors").exists()
            if not (is_diffusers or is_bundle):
                continue
            arch = "Local folder"
            backend = "diffusers" if is_diffusers else "stable_audio_3"
            if is_diffusers:
                try:
                    index = json.loads((child / "model_index.json").read_text(encoding="utf-8"))
                    arch = f"Local diffusers pipeline \u00b7 {index.get('_class_name', 'unknown')}"
                except Exception:  # noqa: BLE001
                    pass
            found.append(
                ModelSpec(
                    id=child.name,
                    name=f"{child.name} (local)",
                    repo_id="",
                    architecture=arch,
                    backend=backend,
                    use_case="Discovered in config/models",
                    summary=(
                        "This folder was found in config/models and is not listed in config/models.json. "
                        "It is loaded directly from disk; sampling defaults are generic."
                    ),
                    highlights=[
                        f"Path: {child}",
                        "Added automatically by the local model scanner",
                    ],
                    max_duration=120.0,
                    default_steps=100,
                    default_guidance=1.0,
                    supports_negative_prompt=is_diffusers,
                    supports_guidance=is_diffusers,
                    gated=False,
                    required_files=["model_index.json"] if is_diffusers else ["model.safetensors"],
                    ignore_patterns=[],
                    notes="Edit config/models.json to give this folder proper metadata and defaults.",
                    local=True,
                )
            )
            if len(found) >= 24:
                break
        return found

    # -- storage -----------------------------------------------------------
    def local_dir(self, spec: ModelSpec) -> Path:
        """Folder in config/models that holds this model's files."""
        return self.models_dir / spec.id

    def is_diffusers_format(self, spec: ModelSpec) -> bool:
        """True when the folder contains model_index.json."""
        return (self.local_dir(spec) / "model_index.json").exists()

    def status(self, spec: ModelSpec) -> ModelStatus:
        """Summarise whether the model is ready, partial or missing."""
        directory = self.local_dir(spec)
        if not directory.exists():
            return ModelStatus("missing", 0, "Not downloaded")
        size = _dir_size(directory)
        missing = [item for item in spec.required_files if not (directory / item).exists()]
        if not missing:
            return ModelStatus("ready", size, f"Installed \u00b7 {paths.human_size(size)}")
        if size > 0:
            return ModelStatus(
                "partial",
                size,
                f"Partial download ({paths.human_size(size)}) \u00b7 missing {', '.join(missing[:3])}",
            )
        return ModelStatus("missing", size, "Not downloaded")

    def expected_size(self, spec: ModelSpec) -> str:
        """Human-readable download size for the model."""
        return f"~{spec.approx_size_gb:.1f} GB" if spec.approx_size_gb else "unknown size"

    # -- download ----------------------------------------------------------
    def download(
        self,
        spec: ModelSpec,
        log: Optional[LogFn] = None,
        progress: Optional[ProgressFn] = None,
        token: Optional[str] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        """Fetch ``spec.repo_id`` into ``config/models/<id>/`` (resumable).

        Cancelling (``cancel_event`` set) raises :class:`DownloadCancelled` and
        keeps everything already on disk, so the next call continues where it
        stopped; files that are already complete are never fetched again.
        """
        log = log or self._log
        if cancel_event is not None and cancel_event.is_set():
            raise DownloadCancelled()
        if not spec.repo_id:
            raise ModelError(
                f"'{spec.name}' has no repository id. Put the weights in {self.local_dir(spec)} manually."
            )
        try:
            from huggingface_hub import snapshot_download  # imported lazily: heavy
        except Exception as exc:  # noqa: BLE001
            raise ModelError(
                "huggingface_hub is missing from the environment. Re-run the bootstrap "
                "(run.bat / run.sh) so requirements.txt is installed."
            ) from exc

        destination = self.local_dir(spec)
        destination.mkdir(parents=True, exist_ok=True)
        token = (token or paths.read_hf_token() or None)

        log(f"Downloading {spec.repo_id} \u2192 {destination}")
        log(f"Expected size: {self.expected_size(spec)}" + (" (gated \u2014 token required)" if spec.gated else ""))

        kwargs: Dict = {"repo_id": spec.repo_id, "local_dir": str(destination)}
        if spec.allow_patterns:
            kwargs["allow_patterns"] = spec.allow_patterns
        if spec.ignore_patterns:
            kwargs["ignore_patterns"] = spec.ignore_patterns
        if token:
            kwargs["token"] = token

        tqdm_class = _make_progress_tqdm(progress, cancel_event)
        if tqdm_class is not None:
            kwargs["tqdm_class"] = tqdm_class

        if cancel_event is not None and cancel_event.is_set():
            raise DownloadCancelled()
        try:
            try:
                snapshot_download(**kwargs)
            except TypeError:
                # Older/newer hub without tqdm_class support: the progress hook
                # cannot abort the transfer there, so honour a cancel first.
                kwargs.pop("tqdm_class", None)
                if cancel_event is not None and cancel_event.is_set():
                    raise DownloadCancelled()
                snapshot_download(**kwargs)
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, DownloadCancelled) or _is_cancel_exception(exc):
                log(
                    "Download cancelled - partial files are kept, so the next "
                    "attempt continues from here."
                )
                raise DownloadCancelled(str(exc) or "cancelled") from exc
            message = str(exc)
            lowered = message.lower()
            if any(tag in lowered for tag in ("401", "403", "gated", "authenticated", "authorization")):
                raise ModelAccessError(
                    f"Access to '{spec.repo_id}' was refused.\n\n"
                    "This repository is gated. Accept the licence on its Hugging Face page "
                    "(https://huggingface.co/" + spec.repo_id + "), then create a read token at "
                    "https://huggingface.co/settings/tokens and paste it into the HF token field "
                    "in the app (it is stored in config/.hf_token)."
                ) from exc
            if "no space left" in lowered or "disk" in lowered:
                raise ModelError(
                    f"Not enough disk space in {destination}. "
                    f"Free up {self.expected_size(spec)} and retry."
                ) from exc
            raise ModelError(f"Download failed: {message}") from exc

        if cancel_event is not None and cancel_event.is_set():
            log("Download cancelled at the end - partial files are kept for the next attempt.")
            raise DownloadCancelled("cancelled")

        try:
            (destination / ".ready").write_text(
                json.dumps({"repo_id": spec.repo_id, "downloaded": _utc_stamp()}, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

        status = self.status(spec)
        log(f"Download finished \u00b7 {status.detail}")
        return destination


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _dir_size(directory: Path) -> int:
    """Total size of the files under a directory, ignoring errors."""
    total = 0
    try:
        for root, _dirs, files in os.walk(directory):
            for name in files:
                try:
                    total += (Path(root) / name).stat().st_size
                except OSError:
                    continue
    except OSError:
        return total
    return total


def _utc_stamp() -> str:
    """UTC timestamp written to .ready so downloads can be identified."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_cancel_exception(exc: BaseException) -> bool:
    """True when ``exc`` (or the exception it wraps) is our cancellation signal."""
    if type(exc).__name__ == "DownloadCancelled":
        return True
    cause = exc.__cause__ or exc.__context__
    if cause is not None and type(cause).__name__ == "DownloadCancelled":
        return True
    return "DownloadCancelled" in str(exc)


def _make_progress_tqdm(progress: Optional[ProgressFn], cancel_event=None):
    """A tqdm subclass that forwards progress into the GUI and can abort.

    Raising from ``update`` (or ``__init__``) is what stops the transfer:
    ``huggingface_hub`` aborts the current file immediately, leaving the part it
    already wrote in ``*.incomplete`` for the next attempt to resume.
    """
    if progress is None and cancel_event is None:
        return None
    try:
        from tqdm.auto import tqdm
    except Exception:  # noqa: BLE001
        return None

    def _check_cancel() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise DownloadCancelled()

    class _ProgressTqdm(tqdm):  # type: ignore[misc]
        """tqdm subclass that forwards byte progress to a callback."""
        def __init__(self, *args, **kwargs):  # noqa: D107
            """Attach the progress callback to this bar."""
            super().__init__(*args, **kwargs)
            _check_cancel()  # stops between files as well as mid-file

        def update(self, n=1):  # noqa: D102
            """Advance the bar and report the new byte count."""
            _check_cancel()
            value = super().update(n)
            if progress is None:
                return value
            try:
                total = getattr(self, "total", None) or 0
                if total:
                    percent = int(min(100, max(0, self.n / total * 100)))
                    progress(f"{self.desc or 'weights'}: {percent}% of {paths.human_size(total)}")
            except Exception:  # noqa: BLE001 - progress must never break a download
                pass
            return value

    return _ProgressTqdm


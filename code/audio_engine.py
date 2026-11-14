"""
audio_engine.py
===============

Stable Audio inference: device routing, pipeline loading, memory policy,
batched track generation and MP3/WAV/FLAC export.

Two architecture families are supported behind a small adapter layer:

``diffusers`` backend
    Any checkpoint published in diffusers format, e.g.
    ``stabilityai/stable-audio-open-1.0`` → :class:`diffusers.StableAudioPipeline`.
    A locally *converted* Stable Audio 3 checkpoint (one that has a
    ``model_index.json`` whose ``_class_name`` is ``StableAudio3Pipeline``) is
    picked up automatically once the installed diffusers exposes that class.

``stable_audio_3`` backend
    The native Stable Audio 3 bundles (``stable-audio-3-small-music``,
    ``-small-sfx``, ``-medium``). Those are *not* diffusers-format, so the
    official ``stable_audio_3`` runtime is used when it is installed. If it is
    missing the engine raises a :class:`GenerationError` that contains the exact
    install command instead of crashing.

Hardware policy
---------------
``cuda``  → ``bfloat16`` (or ``float16`` when bf16 is unsupported), attention
            slicing and CPU offload when VRAM is short, ``torch.inference_mode()``.
``mps``   → ``float32`` (fp16 is noisy on Apple Silicon), CPU fallback for
            unimplemented ops via ``PYTORCH_ENABLE_MPS_FALLBACK=1``.
``cpu``   → ``float32``, no slicing (it only slows CPU inference down).

Out-of-memory conditions (VRAM *or* host RAM) are detected, the pipeline is
degraded to sequential CPU offload where that is possible, and the track is
retried once. A second failure raises :class:`OutOfMemoryGenerationError` with
actionable advice instead of an opaque traceback.

Cancellation
------------
``load()`` and ``generate()`` accept an optional ``cancel_event`` (a
``threading.Event``). It is polled before the expensive stages of a load (folder
check, checkpoint read, device placement), before every track and after every
denoising step, so Cancel stops at the next safe point rather than at the end. A
track that finishes *after* the event was set is discarded instead of written.
:class:`GenerationCancelled` is the internal signal; the GUI worker maps it to a
"cancelled" outcome, which is never shown as an error.

File names
----------
``<YYYYMMDD-HHMMSSmmm>_<first 30 chars of the prompt>[_tNN].<ext>`` (one file per
ticked format, shared stem), with a
``-2``/``-3`` suffix when that name already exists. Only ``a-z0-9-_`` survive,
so nothing illegal or over-long reaches the filesystem and a batch never
overwrites an earlier track.
"""

from __future__ import annotations

import gc
import inspect
import json
import os
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

import paths

paths.bootstrap_env()

try:
    import torch
except Exception as exc:  # noqa: BLE001 - surfaced as a readable GUI error
    raise RuntimeError(
        "PyTorch could not be imported. Run run.bat / run.sh once so the bootstrap "
        "installs requirements.txt into .venv, then start the app again."
    ) from exc

LogFn = Callable[[str], None]
ProgressFn = Callable[[int, str], None]
TrackFn = Callable[["TrackResult"], None]

SA3_INSTALL_HINT = (
    "This checkpoint uses the native Stable Audio 3 architecture, which is not "
    "distributed in diffusers format.\n\n"
    "It needs the official stable_audio_3 runtime, which imports torchaudio — and "
    "torchaudio is published only up to 2.11.1, so it cannot run on newer PyTorch.\n\n"
    "Re-run run.bat / run.sh: the bootstrap installs torch 2.7.1 + torchaudio 2.7.1 "
    "(CUDA when available) and fetches the runtime itself. Do not install torch by "
    "hand — it would overwrite the build matched to your GPU.\n\n"
    "Notes: the medium checkpoint needs CUDA + FlashAttention 2; the small "
    "checkpoints run on CPU. Alternatively place a diffusers-converted SA3 "
    "folder inside config/models — it is detected automatically."
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class GenerationError(RuntimeError):
    """Any recoverable inference failure with a user-facing message."""


class OutOfMemoryGenerationError(GenerationError):
    """VRAM or host RAM was exhausted."""


class GenerationCancelled(Exception):
    """Raised inside the denoising loop when the user presses Cancel."""


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DeviceInfo:
    """Chosen compute device plus the dtype and summary shown in the UI."""
    device: str
    label: str
    dtype_name: str
    torch_dtype: Any
    total_vram_gb: Optional[float] = None

    @property
    def summary(self) -> str:
        vram = f" · {self.total_vram_gb:.1f} GB VRAM" if self.total_vram_gb else ""
        return f"{self.label} — {self.device}, {self.dtype_name}{vram}"

    def as_dict(self) -> Dict[str, Any]:
        """Serialise the device choice for the probe thread's signal."""
        return {
            "device": self.device,
            "label": self.label,
            "dtype": self.dtype_name,
            "vram_gb": self.total_vram_gb,
            "summary": self.summary,
        }


def detect_device(preference: str = "auto") -> DeviceInfo:
    """Pick the best available device, honouring an explicit preference.

    ``preference`` is one of ``auto``, ``cuda``, ``mps``, ``cpu``. A preference
    that is unavailable silently falls back through cuda → mps → cpu so the app
    never dies on a machine without the requested accelerator.
    """
    preference = (preference or "auto").lower()
    if preference in {"cuda", "mps", "cpu"}:
        order = [preference]
    else:
        order = ["cuda", "mps", "cpu"]
    # Always keep CPU as the final safety net.
    if "cpu" not in order:
        order.append("cpu")

    bf16 = False
    try:
        bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:  # noqa: BLE001
        bf16 = False

    for candidate in order:
        try:
            if candidate == "cuda" and torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return DeviceInfo(
                    device="cuda",
                    label=torch.cuda.get_device_name(0),
                    dtype_name="bfloat16" if bf16 else "float16",
                    torch_dtype=torch.bfloat16 if bf16 else torch.float16,
                    total_vram_gb=props.total_memory / (1024 ** 3),
                )
            if candidate == "mps":
                mps = getattr(torch.backends, "mps", None)
                if mps is not None and mps.is_available():
                    return DeviceInfo(
                        device="mps",
                        label="Apple Silicon GPU",
                        dtype_name="float32",
                        torch_dtype=torch.float32,
                    )
            if candidate == "cpu":
                return DeviceInfo(
                    device="cpu",
                    label=os.cpu_count() and f"CPU ({os.cpu_count()} threads)" or "CPU",
                    dtype_name="float32",
                    torch_dtype=torch.float32,
                )
        except Exception:  # noqa: BLE001 - never let probing break startup
            continue

    return DeviceInfo("cpu", "CPU", "float32", torch.float32)


# ---------------------------------------------------------------------------
# Output formats
# ---------------------------------------------------------------------------
# Offered in the GUI in this order, which is also the order they are written.
# MP3 leads because it is the default: a 3-minute clip is ~7 MB at 320 kbps
# instead of the ~31 MB its 24-bit WAV needs, from the same float buffer.
OUTPUT_FORMATS = ("mp3", "wav", "flac")
DEFAULT_OUTPUT_FORMATS = ("mp3",)
# 320 kbps CBR is the highest rate MPEG-1 Layer III defines at 44.1 kHz, so this
# is as good as MP3 gets; lameenc's quality 0 picks its slowest, best search.
MP3_BITRATE_KBPS = 320


def _clean_formats(values: Any) -> tuple:
    """Keep known formats in canonical order; fall back to the default set.

    Anything unknown is dropped rather than written, and an empty selection
    becomes MP3, so the engine can never be asked to produce no file at all.
    """
    wanted = {str(value).strip().lower() for value in (values or ())}
    ordered = tuple(fmt for fmt in OUTPUT_FORMATS if fmt in wanted)
    return ordered or DEFAULT_OUTPUT_FORMATS


# ---------------------------------------------------------------------------
# Request / result types
# ---------------------------------------------------------------------------
@dataclass
class GenerationParams:
    """User-facing generation parameters."""

    prompt: str = ""
    negative_prompt: str = ""
    duration: float = 10.0
    steps: int = 100
    guidance: float = 7.0
    seed: int = -1
    num_tracks: int = 1
    formats: tuple = DEFAULT_OUTPUT_FORMATS

    def sanitized(self, spec: Optional[Any] = None) -> "GenerationParams":
        """Clamp values into a safe range for the selected model."""
        max_duration = float(getattr(spec, "max_duration", 47.0) or 47.0)
        return replace(
            self,
            prompt=(self.prompt or "").strip(),
            negative_prompt=(self.negative_prompt or "").strip(),
            duration=float(min(max(float(self.duration), 1.0), max_duration)),
            steps=int(min(max(int(self.steps), 1), 500)),
            guidance=float(min(max(float(self.guidance), 0.0), 30.0)),
            num_tracks=int(min(max(int(self.num_tracks), 1), MAX_TRACKS)),
            formats=_clean_formats(self.formats),
        )


@dataclass
class TrackResult:
    """One generated WAV file."""

    path: str
    prompt: str
    negative_prompt: str
    model_id: str
    model_name: str
    seed: int
    duration_s: float
    sample_rate: int
    channels: int
    size_bytes: int
    created: str
    device: str
    elapsed_s: float
    files: Dict[str, str] = None  # type: ignore[assignment]  # format → file name
    peaks: List[float] = None  # type: ignore[assignment]

    def as_meta(self, include_peaks: bool = False) -> Dict[str, Any]:
        """Build the JSON sidecar metadata for one finished track."""
        meta = {
            "path": self.path,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "seed": self.seed,
            "duration_s": round(self.duration_s, 3),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "size_bytes": self.size_bytes,
            "created": self.created,
            "device": self.device,
            "elapsed_s": round(self.elapsed_s, 2),
            "files": dict(self.files or {}),
            "peaks": list(self.peaks or []) if include_peaks else None,
        }
        if meta["peaks"] is None:
            meta.pop("peaks")
        return meta


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def compute_peaks(audio: np.ndarray, buckets: int = 1200) -> List[float]:
    """Downsample an ``(samples, channels)`` array to normalized peak buckets."""
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 2:
        mono = data.mean(axis=1)
    else:
        mono = data.reshape(-1)
    if mono.size == 0:
        return []
    buckets = max(1, int(buckets))
    if mono.size < buckets:
        return [float(min(1.0, abs(v))) for v in mono]
    step = mono.size // buckets
    trimmed = mono[: step * buckets].reshape(buckets, step)
    peaks = np.max(np.abs(trimmed), axis=1)
    peak = float(np.max(peaks)) or 1.0
    return [float(min(1.0, v / peak)) for v in peaks]


def _tensor_to_numpy(value: Any) -> np.ndarray:
    """Normalize pipeline output into ``(samples, channels)`` float32."""
    if torch.is_tensor(value):
        array = value.detach().to("cpu").float().numpy()
    else:
        array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim == 1:
        array = array[:, None]
    elif array.ndim == 3:
        array = array[0]
    if array.ndim == 2 and array.shape[0] in (1, 2) and array.shape[1] > array.shape[0]:
        array = array.T  # channels-first → samples-first
    return np.ascontiguousarray(array, dtype=np.float32)


def _is_oom(exc: BaseException) -> bool:
    """Recognise VRAM and host-RAM out-of-memory failures."""
    if isinstance(exc, MemoryError):
        return True
    try:
        if isinstance(exc, torch.cuda.OutOfMemoryError):  # type: ignore[attr-defined]
            return True
    except Exception:  # noqa: BLE001
        pass
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "out of memory",
            "not enough memory",
            "cannot allocate memory",
            "defaultcpuallocator",
            "insufficient memory",
            "cuda error: out of memory",
        )
    )


def _check_cancel(cancel_event: Any) -> None:
    """Raise :class:`GenerationCancelled` when the user pressed Cancel."""
    if cancel_event is not None and cancel_event.is_set():
        raise GenerationCancelled("cancelled by user")


def _localize_text_encoder(
    model_config: Dict[str, Any], directory: Path
) -> Optional[str]:
    """Point a T5Gemma conditioner at the checkpoint's own text-encoder folder.

    Stable Audio 3 stores the text encoder inside the checkpoint repo
    (``t5gemma-b-b-ul2/``) but its config asks transformers for
    ``repo_id=…, subfolder=…``. That is a fresh Hub download: it fails with a
    401 on a gated repo when no token is exported, and it re-downloads even when
    the files already sit in ``config/models``. Setting ``model_path`` to the local
    folder (and dropping ``subfolder``, which would be appended twice) makes the
    conditioner load straight from disk.
    """
    try:
        conditioners = model_config["model"]["conditioning"]["configs"]
    except (KeyError, TypeError):
        return None
    if not isinstance(conditioners, list):
        return None
    for entry in conditioners:
        if not isinstance(entry, dict) or entry.get("type") != "t5gemma":
            continue
        config = entry.get("config")
        if not isinstance(config, dict) or config.get("model_path"):
            continue
        subfolder = str(config.get("subfolder") or "t5gemma-b-b-ul2")
        candidate = Path(directory) / subfolder
        if not (candidate / "config.json").exists():
            return None
        config["model_path"] = str(candidate)
        config.pop("subfolder", None)  # model_path already points inside it
        return str(candidate)
    return None


def _gated_access_hint(exc: BaseException, repo_id: str = "") -> str:
    """Actionable text when a gated/private repository refused the request.

    Hugging Face answers a gated checkpoint without a token with a bare
    ``OSError: You are trying to access a gated repo`` plus a 401, which says
    nothing about the missing token or where to get one.
    """
    text = str(exc).lower()
    if not any(
        marker in text
        for marker in (
            "gated",
            "401",
            "403",
            "unauthorised",
            "unauthorized",
            "forbidden",
            "access to",
        )
    ):
        return ""
    where = (
        f"https://huggingface.co/{repo_id}"
        if repo_id
        else "the model page on huggingface.co"
    )
    if paths.read_hf_token():
        return (
            "The saved Hugging Face token was refused by a gated repository.\n\n"
            f"Make sure the account that owns the token accepted the licence at {where}, "
            "then paste a token with read access into the HF token field."
        )
    return (
        "No Hugging Face token is entered, and this checkpoint is in a gated repo.\n\n"
        f"1. Sign in at {where} and accept the licence.\n"
        "2. Create a read token at https://huggingface.co/settings/tokens.\n"
        "3. Paste it into the HF token field on the left (it stays outlined red until "
        "a token is typed, and Save/Generate stores it in config/.hf_token)."
    )


def _accepted_kwargs(func: Any) -> Optional[set]:
    """Parameter names of a pipeline ``__call__`` (``None`` when unknown)."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return None
    names = set(signature.parameters)
    if not names:
        return None
    return names


# Windows refuses these stem names even with an extension.
_RESERVED_STEMS = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def _prompt_slug(text: str, limit: int = 30) -> str:
    """Filesystem-safe version of the prompt for use in a file name.

    Keeps ASCII letters/digits, turns every other run of characters into a single
    dash (so ``\"Warm pad, 80s!\"`` does not produce quotes, commas or spaces), and
    caps the result at ``limit`` characters so path limits stay comfortable.
    """
    out: List[str] = []
    for char in (text or "").lower():
        if char.isascii() and char.isalnum():
            out.append(char)
        elif not out or out[-1] != "-":
            out.append("-")
    slug = "".join(out).strip("-_")[:limit].strip("-_")
    if not slug:
        return "track"
    if slug in _RESERVED_STEMS:
        slug = f"{slug}-audio"
    return slug


# Upper bound for a single batch. There is no technical limit here (tracks are
# generated one at a time and streamed to disk as they finish), so this only
# keeps a typo from queuing days of work.
MAX_TRACKS = 999

# ``StableAudioModel.generate()`` derives the generated length from ``duration``
# but then clamps it to its ``sample_size`` argument, which defaults to 5,292,032
# samples — exactly 120 s at 44.1 kHz. Passing only ``duration`` therefore caps
# every longer request at two minutes, so the sample budget is set explicitly.
SA3_DURATION_PADDING_SEC = 6.0  # matches the runtime's duration_padding_sec default

# ``stable_audio_3.models.transformer.apply_attn()`` takes the fused attention
# kernel whenever ``flash_attn_func`` imported - it never inspects the tensors'
# device. flash_attn is a CUDA extension, so a CPU run dies on its first
# attention call with "NotImplementedError: ... 'flash_attn::_flash_attn_varlen_forward'
# ... from the 'CPU' backend". The package ships a complete SDPA fallback that is
# simply never reached; blanking these globals routes to it.
_SA3_FLASH_SYMBOLS = ("flash_attn_func", "flash_attn_kvpacked_func", "flash_attn_varlen_func")
_SA3_FLASH_BACKUP: Dict[str, Any] = {}


def _sa3_attention_patch(device: str) -> Optional[str]:
    """Point stable_audio_3 at an attention kernel that works on ``device``.

    Returns ``"disabled"`` when the CUDA-only fused path was switched off for a
    non-CUDA run, ``"restored"`` when it was put back for a CUDA run, and
    ``None`` when there was nothing to do.
    """
    try:
        from stable_audio_3.models import transformer as sa3_transformer  # type: ignore
    except Exception:  # noqa: BLE001 - package layout moved, nothing to patch
        return None

    if device == "cuda":
        restored = False
        for name, value in list(_SA3_FLASH_BACKUP.items()):
            if getattr(sa3_transformer, name, None) is None:
                setattr(sa3_transformer, name, value)
                restored = True
            _SA3_FLASH_BACKUP.pop(name, None)
        return "restored" if restored else None

    disabled = False
    for name in _SA3_FLASH_SYMBOLS:
        value = getattr(sa3_transformer, name, None)
        if value is None:
            continue
        _SA3_FLASH_BACKUP[name] = value
        setattr(sa3_transformer, name, None)
        disabled = True
    return "disabled" if disabled else None


def _sa3_sample_size(
    duration: float, sample_rate: int, max_samples: Optional[int]
) -> int:
    """Samples to allocate for a ``duration`` request in the SA3 runtime.

    The extra padding gives the model context past the requested end, which is
    what the runtime itself does; the result is never larger than the checkpoint
    was built for.
    """
    target = int((float(duration) + SA3_DURATION_PADDING_SEC) * int(sample_rate))
    if max_samples:
        target = min(target, int(max_samples))
    return max(1, target)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class AudioEngine:
    """Loads one model at a time and generates batches of WAV files."""

    def __init__(
        self,
        device_preference: str = "auto",
        log: Optional[LogFn] = None,
    ) -> None:
        """Store the model spec and start with nothing loaded."""
        self.log: LogFn = log or (lambda *_: None)
        self.device_info: DeviceInfo = detect_device(device_preference)
        self._pipeline: Any = None
        self._sa3_model: Any = None
        self._backend: Optional[str] = None
        self._spec: Any = None
        self._sample_rate: int = 44100
        self._sa3_max_samples: Optional[int] = None
        self._offload_level: int = 0
        self._load_marks: list = []  # [(stage label, perf_counter), ...] for the load summary
        self.log(f"Compute device: {self.device_info.summary}")

    # -- introspection -----------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        """True when a pipeline is resident in memory."""
        return self._backend is not None

    @property
    def loaded_model_id(self) -> Optional[str]:
        """Id of the loaded checkpoint, or None."""
        return getattr(self._spec, "id", None)

    @property
    def sample_rate(self) -> int:
        """Sample rate of the loaded model, falling back to the spec."""
        return self._sample_rate

    @property
    def max_generation_duration(self) -> Optional[float]:
        """Longest clip the loaded checkpoint itself supports, in seconds."""
        if self._sa3_max_samples and self._sample_rate:
            return self._sa3_max_samples / float(self._sample_rate)
        return None

    def _mark(self, label: str) -> None:
        """Record the end of a load stage for the one-line summary."""
        self._load_marks.append((label, time.perf_counter()))

    def _load_timing_line(self, spec: Any) -> str:
        """One line describing where a model load spent its time."""
        marks = self._load_marks
        total = marks[-1][1] - marks[0][1]
        parts = []
        previous = marks[0][1]
        for label, at in marks[1:]:
            parts.append(f"{label} {at - previous:.2f}")
            previous = at
        return (
            f"load: {total:.2f} s = {' + '.join(parts)} "
            f"({getattr(spec, 'name', '?')} · {self._backend or '?'} · "
            f"{self.device_info.device})"
        )

    def _track_timing_line(
        self, index: int, total: int, result: Any, sampling: float, wall: float, seed: int
    ) -> str:
        """One line separating sampling from saving for a single track."""
        realtime = result.duration_s / sampling if sampling > 0 else 0.0
        return (
            f"generation {index + 1}/{total}: {wall:.2f} s = "
            f"sampling {sampling:.2f} + save {wall - sampling:.2f} "
            f"| {realtime:.2f}x realtime ({result.duration_s:.1f} s audio, "
            f"seed {seed}, {self.device_info.device})"
        )

    # -- loading -----------------------------------------------------------
    def load(
        self,
        spec: Any,
        model_dir: Path,
        offload: str = "auto",
        progress: Optional[ProgressFn] = None,
        cancel_event: Any = None,
    ) -> str:
        """Load the pipeline for ``spec`` from ``model_dir``.

        ``cancel_event`` is polled between the expensive stages (checkpoint read,
        device placement) so Cancel stops a load as early as the runtime allows
        instead of looking dead for minutes.

        Returns a short human readable summary of what was loaded.
        """
        self._load_marks = [("start", time.perf_counter())]
        _check_cancel(cancel_event)
        directory = Path(model_dir)
        if not directory.exists():
            raise GenerationError(
                f"Model folder {directory} does not exist. Download the weights first."
            )

        self.unload()
        self._spec = spec
        diffusers_format = (directory / "model_index.json").exists()
        class_name = _pipeline_class_name(directory)
        _check_cancel(cancel_event)

        if diffusers_format:
            summary = self._load_diffusers(
                directory, class_name, offload, progress, cancel_event
            )
        elif getattr(spec, "backend", "") == "stable_audio_3":
            summary = self._load_stable_audio_3(spec, directory, progress, cancel_event)
        else:
            raise GenerationError(
                f"{directory.name} has no model_index.json, and its backend "
                f"'{getattr(spec, 'backend', '?')}' is not supported by this build."
            )
        self._mark("ready")
        self.log(self._load_timing_line(spec))
        return summary

    def _load_diffusers(
        self,
        directory: Path,
        class_name: Optional[str],
        offload: str,
        progress: Optional[ProgressFn],
        cancel_event: Any = None,
    ) -> str:
        """Load a diffusers checkpoint and place it on the chosen device."""
        try:
            import diffusers
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(
                "The 'diffusers' package is not importable. Re-run the bootstrap so "
                "requirements.txt is installed."
            ) from exc
        self._mark("imports")

        if progress:
            progress(-1, f"Loading pipeline from {directory.name}…")
        self.log(f"Loading diffusers pipeline from {directory}")
        self.log(f"model_index.json → _class_name = {class_name or 'unknown'}")

        pipeline_cls = getattr(diffusers, class_name, None) if class_name else None
        using_auto = False
        if pipeline_cls is None:
            auto_cls = getattr(diffusers, "AutoPipelineForText2Audio", None)
            if auto_cls is None:
                raise GenerationError(
                    f"diffusers does not expose '{class_name or 'a text-to-audio pipeline'}'. "
                    "Upgrade diffusers (see requirements.txt)."
                )
            pipeline_cls = auto_cls
            using_auto = True

        _check_cancel(cancel_event)
        kwargs: Dict[str, Any] = {"torch_dtype": self.device_info.torch_dtype}
        if any(directory.glob("**/*.safetensors")):
            kwargs["use_safetensors"] = True

        try:
            pipeline = pipeline_cls.from_pretrained(str(directory), **kwargs)
        except Exception as exc:  # noqa: BLE001
            if _is_oom(exc):
                raise OutOfMemoryGenerationError(self._oom_hint(exc)) from exc
            hint = ""
            if class_name and "StableAudio3" in class_name and using_auto:
                hint = (
                    "\n\nThis folder is a Stable Audio 3 (diffusers) checkpoint but the "
                    "installed diffusers has no StableAudio3Pipeline. Install diffusers "
                    "from source: pip install \"git+https://github.com/huggingface/diffusers\" "
                    "— or use the native SA3 backend instead (see README)."
                )
            raise GenerationError(
                f"Failed to load the pipeline from {directory}.\n"
                f"{type(exc).__name__}: {exc}{hint}"
            ) from exc

        self._mark("weights")
        self._pipeline = pipeline
        self._backend = "diffusers"
        self._sample_rate = _pipeline_sample_rate(pipeline) or 44100

        try:
            pipeline.set_progress_bar_config(disable=True)
        except Exception:  # noqa: BLE001
            pass

        _check_cancel(cancel_event)
        self._apply_memory_policy(pipeline, offload)
        self._mark("placement")

        mode = {0: "full GPU/CPU", 1: "model CPU offload", 2: "sequential CPU offload"}[
            self._offload_level
        ]
        return (
            f"{self._spec.name if self._spec else directory.name} loaded "
            f"({class_name or 'auto pipeline'}, {self.device_info.dtype_name}, {mode}, "
            f"{self._sample_rate} Hz)"
        )

    def _apply_memory_policy(self, pipeline: Any, offload: str) -> None:
        """Pick an offload level from VRAM and the model's needs."""
        info = self.device_info
        mode = (offload or "auto").lower()
        self._offload_level = 0

        if info.device == "cuda":
            if mode == "auto":
                needed = float(getattr(self._spec, "min_vram_gb", 0) or 8.0)
                vram = info.total_vram_gb or 0.0
                if vram and vram < needed * 0.75:
                    mode = "sequential"
                elif vram and vram < needed:
                    mode = "model"
                else:
                    mode = "none"
            try:
                if mode in {"sequential", "sequential_cpu_offload"}:
                    pipeline.enable_sequential_cpu_offload()
                    self._offload_level = 2
                elif mode in {"model", "model_cpu_offload", "cpu_offload"}:
                    pipeline.enable_model_cpu_offload()
                    self._offload_level = 1
                else:
                    pipeline.to("cuda")
            except Exception as exc:  # noqa: BLE001 - fall back to plain GPU/CPU
                self.log(f"Offload setup failed ({exc}); falling back to a plain device move.")
                try:
                    pipeline.to(info.device)
                except Exception:  # noqa: BLE001
                    pipeline.to("cpu")
                    self.device_info = replace(
                        info, device="cpu", dtype_name="float32", torch_dtype=torch.float32
                    )
                    self._offload_level = 0
            # Attention slicing keeps peak VRAM down; cheap on GPU, costly on CPU.
            try:
                pipeline.enable_attention_slicing("max" if self._offload_level == 2 else "auto")
            except Exception:  # noqa: BLE001
                pass
        elif info.device == "mps":
            try:
                pipeline.to("mps")
                pipeline.enable_attention_slicing()
            except Exception as exc:  # noqa: BLE001
                self.log(f"MPS placement failed ({exc}); using CPU instead.")
                self.device_info = replace(
                    info, device="cpu", dtype_name="float32", torch_dtype=torch.float32
                )
                pipeline.to("cpu")
        else:
            pipeline.to("cpu")
            if getattr(self._spec, "min_vram_gb", 0):
                self.log(
                    "Running on CPU with float32. Expect minutes per clip — this is the "
                    "development fallback, CUDA is the tuned target."
                )

    def _load_stable_audio_3(
        self,
        spec: Any,
        directory: Path,
        progress: Optional[ProgressFn],
        cancel_event: Any = None,
    ) -> str:
        """Load a native SA3 checkpoint through Stability's runtime."""
        try:
            from stable_audio_3 import StableAudioModel  # type: ignore
            from stable_audio_3.loading_utils import load_diffusion_cond  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(SA3_INSTALL_HINT) from exc
        self._mark("imports")
        _check_cancel(cancel_event)

        # stable_audio_3 chooses its attention kernel at import time: with
        # flash_attn importable it takes the fused CUDA path even when the tensors
        # live on the CPU, which raises NotImplementedError. FlashAttention is a
        # CUDA-only extension, so a CPU run has to be pointed at the SDPA fallback.
        target = self.device_info.device
        _sa3_attention_patch(target)
        try:
            import flash_attn  # noqa: F401 - presence is what matters

            flash_label = getattr(flash_attn, "__version__", "?")
        except Exception:  # noqa: BLE001
            flash_label = ""
        if flash_label and target == "cuda":
            self.log(
                f"FlashAttention {flash_label} available - SA3 attention uses the "
                "fused CUDA kernel"
            )
        elif flash_label:
            self.log(
                f"FlashAttention {flash_label} is CUDA-only - running on {target}, so "
                "SA3 attention uses the standard SDPA path."
            )
        else:
            self.log(
                "FlashAttention 2 not installed - SA3 uses standard attention "
                "(slower, more VRAM). flash_attn_setup.py installs it."
            )

        # StableAudioModel.from_pretrained() accepts registry keys only ("small-sfx",
        # "medium" …) and re-resolves them through the Hub, so a folder that was
        # downloaded or copied into config/models has to be opened from its own files.
        config_path = directory / "model_config.json"
        ckpt_path = directory / "model.safetensors"
        missing = [p.name for p in (config_path, ckpt_path) if not p.exists()]
        if missing:
            raise GenerationError(
                f"{directory.name} is missing {', '.join(missing)}. Press \"Download "
                "weights\" to fetch the complete native checkpoint (it includes the "
                "T5Gemma text encoder), or copy those files into the folder."
            )

        _check_cancel(cancel_event)
        if progress:
            progress(-1, f"Loading Stable Audio 3 model from {directory.name}…")

        device = self.device_info.device
        # Half precision only pays off, and only works cleanly, on CUDA.
        model_half = bool(
            device == "cuda"
            and self.device_info.torch_dtype in {torch.float16, torch.bfloat16}
        )
        self.log(f"stable_audio_3: loading {ckpt_path.name} from {directory}")
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                model_config = json.load(handle)
            # The checkpoint's own length budget, in samples. generate() silently
            # clamps to it, so remember it to report the real limit.
            self._sa3_max_samples = int(model_config.get("sample_size") or 0) or None
            local_encoder = _localize_text_encoder(model_config, directory)
            if local_encoder:
                self.log(f"stable_audio_3: text encoder loaded from {local_encoder}")
            backbone = load_diffusion_cond(
                model_config, str(ckpt_path), device=device, model_half=model_half
            )
            backbone.use_lora = False
            backbone.lora_names = []
            model = StableAudioModel(backbone, model_config, device, model_half)
            self._mark("weights")
        except Exception as exc:  # noqa: BLE001
            raw = (
                f"stable_audio_3 could not open {ckpt_path.name}: "
                f"{type(exc).__name__}: {exc}"
            )
            hint = _gated_access_hint(exc, getattr(spec, "repo_id", ""))
            # Put the actionable line first: the dialog headline is line 1.
            raise GenerationError(f"{hint}\n\n{raw}" if hint else raw) from exc

        _check_cancel(cancel_event)
        self._sa3_model = model
        self._backend = "stable_audio_3"
        self._sample_rate = int(getattr(spec, "sample_rate", 44100) or 44100)
        try:
            self._sample_rate = int(backbone.sample_rate)
        except Exception:  # noqa: BLE001 - keep the registry value
            pass
        self._mark("placement")
        return (
            f"{getattr(spec, 'name', directory.name)} loaded "
            f"(stable_audio_3 runtime, {device}, {'fp16' if model_half else 'fp32'})"
        )

    def unload(self) -> None:
        """Release the pipeline and empty the accelerator caches."""
        self._pipeline = None
        self._sa3_model = None
        self._backend = None
        self._spec = None
        self._sa3_max_samples = None
        self._offload_level = 0
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mps = getattr(torch, "mps", None)
            if mps is not None and hasattr(mps, "empty_cache"):
                mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # -- generation --------------------------------------------------------
    def generate(
        self,
        spec: Any,
        params: GenerationParams,
        out_dir: Path,
        on_track: Optional[TrackFn] = None,
        on_progress: Optional[ProgressFn] = None,
        cancel_event: Any = None,
    ) -> List[TrackResult]:
        """Generate ``params.num_tracks`` distinct WAV files into ``out_dir``.

        Seeds are deterministic: a fixed seed ``s`` produces ``s, s+1, s+2 …``
        across the batch, while seed ``-1`` draws a fresh random seed per track.
        """
        if not self.is_loaded:
            raise GenerationError("No model is loaded. Select a model and press Generate.")

        parameters = params.sanitized(spec)
        if not parameters.prompt:
            raise GenerationError("The prompt is empty — describe the audio you want.")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        total = parameters.num_tracks
        results: List[TrackResult] = []
        self.log(
            f"Generating {total} track(s) · {parameters.duration:.1f}s · "
            f"{parameters.steps} steps · CFG {parameters.guidance:g} · "
            f"seed {parameters.seed if parameters.seed >= 0 else 'random'}"
        )
        batch_started = time.perf_counter()
        limit = self.max_generation_duration
        if limit is not None and parameters.duration > limit:
            self.log(
                f"This checkpoint tops out at {limit:.0f}s, so the "
                f"{parameters.duration:.0f}s request comes back as about {limit:.0f}s."
            )

        for index in range(total):
            if cancel_event is not None and cancel_event.is_set():
                self.log("Cancelled before the next track.")
                break

            if parameters.seed >= 0:
                seed = (parameters.seed + index) % (2 ** 31 - 1)
            else:
                seed = int.from_bytes(os.urandom(4), "little") % (2 ** 31 - 1)

            self._emit(on_progress, int(100 * index / total),
                       f"Track {index + 1}/{total} · seed {seed} · denoising…")
            started = time.perf_counter()
            try:
                audio, sample_rate = self._run_one(
                    spec, parameters, seed, index, total, on_progress, cancel_event
                )
            except GenerationCancelled:
                self.log(f"Track {index + 1} cancelled.")
                break
            if cancel_event is not None and cancel_event.is_set():
                # Cancelled while the runtime was inside a non-interruptible
                # call: do not save a track the user asked to stop.
                self.log(f"Track {index + 1} finished but was discarded (cancelled).")
                break
            elapsed = time.perf_counter() - started

            base = out_dir / self._filename(parameters.prompt, index, total)
            base = self._unique_path(base)
            written = self._write_formats(base, audio, sample_rate, parameters.formats)
            written_paths = list(written.values())
            path = written_paths[0]  # the first requested format represents the track
            size = sum(p.stat().st_size for p in written_paths if p.exists())
            result = TrackResult(
                path=str(path),
                prompt=parameters.prompt,
                negative_prompt=parameters.negative_prompt,
                model_id=getattr(spec, "id", ""),
                model_name=getattr(spec, "name", ""),
                seed=seed,
                duration_s=audio.shape[0] / float(sample_rate or 1),
                sample_rate=int(sample_rate),
                channels=int(audio.shape[1]) if audio.ndim == 2 else 1,
                size_bytes=size,
                created=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                device=self.device_info.device,
                elapsed_s=elapsed,
                files={fmt: p.name for fmt, p in written.items()},
                peaks=compute_peaks(audio),
            )
            self._write_sidecar(path, result)
            results.append(result)
            finished = time.perf_counter()
            self.log(
                f"Saved {' + '.join(p.name for p in written_paths)} · "
                f"{paths.human_size(size)} · "
                f"{result.duration_s:.1f}s audio in {elapsed:.1f}s"
            )
            self.log(
                self._track_timing_line(
                    index, total, result, elapsed, finished - started, seed
                )
            )
            if on_track is not None:
                on_track(result)

        # One closing line for the whole batch: per-track lines are useful for a
        # single clip but become noise at 20+ tracks, and the average is what
        # tells you whether a longer batch is worth queueing.
        batch_elapsed = time.perf_counter() - batch_started
        if results:
            audio_seconds = sum(result.duration_s for result in results)
            per_track = batch_elapsed / len(results)
            realtime = audio_seconds / batch_elapsed if batch_elapsed > 0 else 0.0
            self.log(
                f"batch: {len(results)}/{total} track(s) in {batch_elapsed:.2f} s = "
                f"{per_track:.2f} s/track | {audio_seconds:.1f} s audio | "
                f"{realtime:.2f}x realtime"
            )
        else:
            self.log(
                f"batch: 0/{total} track(s) in {batch_elapsed:.2f} s (nothing saved)"
            )

        self._emit(on_progress, 100, f"Finished · {len(results)} track(s)")
        return results

    def _run_one(
        self,
        spec: Any,
        params: GenerationParams,
        seed: int,
        index: int,
        total: int,
        on_progress: Optional[ProgressFn],
        cancel_event: Any,
    ) -> tuple:
        """Generate a single waveform, retrying once after an OOM."""
        attempts = 0
        while True:
            try:
                return self._invoke(spec, params, seed, index, total, on_progress, cancel_event)
            except GenerationCancelled:
                raise
            except Exception as exc:  # noqa: BLE001
                if not _is_oom(exc):
                    if isinstance(exc, GenerationError):
                        raise
                    raise GenerationError(
                        f"Generation failed: {type(exc).__name__}: {exc}"
                    ) from exc
                if attempts >= 1 or not self._degrade_after_oom():
                    raise OutOfMemoryGenerationError(self._oom_hint(exc)) from exc
                attempts += 1
                self.log("Recovered from an out-of-memory condition — retrying the track.")

    def _invoke(
        self,
        spec: Any,
        params: GenerationParams,
        seed: int,
        index: int,
        total: int,
        on_progress: Optional[ProgressFn],
        cancel_event: Any,
    ) -> tuple:
        """Call the pipeline with only the kwargs its signature accepts."""
        duration = float(params.duration)
        if self._backend == "diffusers" and self._pipeline is not None:
            generator = self._make_generator(seed)
            accepted = _accepted_kwargs(self._pipeline.__call__)
            kwargs: Dict[str, Any] = {}

            def allow(name: str) -> bool:
                return accepted is None or name in accepted

            if allow("prompt"):
                kwargs["prompt"] = params.prompt
            duration_arg = getattr(spec, "duration_arg", "audio_end_in_s")
            if allow(duration_arg):
                kwargs[duration_arg] = duration
            elif allow("duration"):
                kwargs["duration"] = duration
            if allow("num_inference_steps"):
                kwargs["num_inference_steps"] = int(params.steps)
            if (
                getattr(spec, "supports_guidance", True)
                and params.guidance > 1.0
                and allow("guidance_scale")
            ):
                kwargs["guidance_scale"] = float(params.guidance)
            if (
                getattr(spec, "supports_negative_prompt", True)
                and params.negative_prompt
                and params.guidance > 1.0
                and allow("negative_prompt")
            ):
                kwargs["negative_prompt"] = params.negative_prompt
            if allow("num_waveforms_per_prompt"):
                kwargs["num_waveforms_per_prompt"] = 1
            if allow("generator"):
                kwargs["generator"] = generator
            if allow("callback_on_step_end"):
                kwargs["callback_on_step_end"] = self._step_callback(
                    index, total, params.steps, on_progress, cancel_event
                )

            with torch.inference_mode():
                output = self._pipeline(**kwargs)
            audios = getattr(output, "audios", None)
            if audios is None:
                audios = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(audios) and audios.ndim >= 3:
                audios = audios[0]
            return _tensor_to_numpy(audios), self._sample_rate

        if self._backend == "stable_audio_3" and self._sa3_model is not None:
            generate_fn = getattr(self._sa3_model, "generate")
            accepted = _accepted_kwargs(generate_fn)
            wanted: Dict[str, Any] = {
                "prompt": params.prompt,
                "duration": duration,
                "steps": int(params.steps),
                "cfg_scale": float(params.guidance),
                "seed": int(seed),
                "batch_size": 1,
                "sample_size": _sa3_sample_size(
                    duration, self._sample_rate, self._sa3_max_samples
                ),
            }
            if getattr(spec, "supports_negative_prompt", False) and params.negative_prompt:
                wanted["negative_prompt"] = params.negative_prompt
            kwargs = {
                key: value
                for key, value in wanted.items()
                if accepted is None or key in accepted
            }
            try:
                with torch.inference_mode():
                    output = generate_fn(**kwargs)
            except Exception as exc:  # noqa: BLE001
                raise GenerationError(
                    f"stable_audio_3 generate() failed: {type(exc).__name__}: {exc}"
                ) from exc
            audio = _tensor_to_numpy(output)
            return audio, self._sample_rate

        raise GenerationError("Internal error: no backend is loaded.")

    def _make_generator(self, seed: int) -> Any:
        """Create a seeded torch generator for reproducible tracks."""
        device = self.device_info.device
        gen_device = "cuda" if device == "cuda" else "cpu"
        try:
            return torch.Generator(device=gen_device).manual_seed(int(seed))
        except Exception:  # noqa: BLE001
            return torch.Generator().manual_seed(int(seed))

    def _step_callback(
        self,
        index: int,
        total: int,
        steps: int,
        on_progress: Optional[ProgressFn],
        cancel_event: Any,
    ):
        """Report denoising progress and honour cancel between steps."""
        steps_total = max(1, int(steps))

        def _callback(_pipe, step_index, _timestep, callback_kwargs):
            if cancel_event is not None and cancel_event.is_set():
                raise GenerationCancelled("cancelled by user")
            if on_progress is not None:
                done = index + (step_index + 1) / steps_total
                percent = int(min(99, 100 * done / max(1, total)))
                on_progress(
                    percent,
                    f"Track {index + 1}/{total} · step {step_index + 1}/{steps_total}",
                )
            return callback_kwargs

        return _callback

    # -- OOM handling ------------------------------------------------------
    def _degrade_after_oom(self) -> bool:
        """Try to shed memory so the same request can succeed. True if changed."""
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        try:
            mps = getattr(torch, "mps", None)
            if mps is not None and hasattr(mps, "empty_cache"):
                mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass

        pipeline = self._pipeline
        if pipeline is None or self._offload_level >= 2:
            return False
        try:
            if self.device_info.device == "cuda":
                self.log("Enabling sequential CPU offload to reduce peak VRAM…")
                pipeline.enable_sequential_cpu_offload()
            else:
                self.log("Reducing memory pressure — moving the pipeline to CPU…")
                pipeline.to("cpu")
                self.device_info = replace(
                    self.device_info,
                    device="cpu",
                    dtype_name="float32",
                    torch_dtype=torch.float32,
                )
            self._offload_level = 2
            return True
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not reduce memory usage automatically: {exc}")
            return False

    def _oom_hint(self, exc: BaseException) -> str:
        """Turn an out-of-memory error into advice the user can act on."""
        device = self.device_info.device
        if device == "cuda":
            advice = (
                "Close other GPU applications (browser tabs, games, Blender/Resolve) and try "
                "again, reduce the duration, or pick a smaller model. Sequential CPU offload "
                "was already attempted."
            )
        elif device == "mps":
            advice = (
                "Shorten the clip or switch to CPU in the Compute device selector. Apple "
                "Silicon shares RAM between CPU and GPU, so a long clip can exhaust it."
            )
        else:
            advice = (
                "The machine ran out of system RAM. Shorten the duration, lower the step "
                "count, close other applications, or use a smaller model."
            )
        return (
            "Out of memory while generating.\n\n"
            f"{advice}\n\nTechnical detail: {type(exc).__name__}: {exc}"
        )

    # -- file output -------------------------------------------------------
    def _filename(self, prompt: str, index: int, total: int) -> str:
        """``<timestamp ms>_<first 30 chars of the prompt>[_tNN]`` plus ``.wav``.

        The suffix is a placeholder: the caller swaps it for every requested
        format, so MP3/WAV/FLAC siblings share one stem and one sidecar.

        The timestamp carries milliseconds so two batches started in the same
        second with the same prompt normally get different names.
        """
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S%f")[:-3]
        suffix = f"_t{index + 1:02d}" if total > 1 else ""
        return f"{stamp}_{_prompt_slug(prompt)}{suffix}.wav"

    def _unique_path(self, path: Path) -> Path:
        """Return ``path``, or ``…-2.wav``/``…-3.wav`` when the name is taken.

        Windows resolves ``datetime.now()`` to roughly a millisecond, so the name
        alone cannot guarantee uniqueness; the filesystem is the real authority.
        """
        if not self._name_taken(path):
            return path
        stem, suffix, counter = path.stem, path.suffix, 2
        while True:
            candidate = path.with_name(f"{stem}-{counter}{suffix}")
            if not self._name_taken(candidate):
                return candidate
            counter += 1

    @staticmethod
    def _name_taken(path: Path) -> bool:
        """True when the track name is used by any output format or its sidecar.

        A batch can write MP3 + WAV + FLAC side by side, so uniquifying only the
        WAV would let ``-2`` variants collide on the other extensions.
        """
        stem = path.with_suffix("")
        return any(stem.with_suffix(f".{ext}").exists() for ext in (*OUTPUT_FORMATS, "json"))

    def _write_formats(
        self, base: Path, audio: np.ndarray, sample_rate: int, formats: Any
    ) -> Dict[str, Path]:
        """Write every requested format from one normalized buffer.

        Normalizing once (rather than per file) keeps the exports at identical
        level, and the MP3 is encoded straight from the full-resolution samples,
        so it is only ever one encode away from the lossless source.
        """
        data = np.asarray(audio, dtype=np.float32)
        if data.size:
            peak = float(np.max(np.abs(data)))
            if peak > 1.0:
                self.log(f"Peak {peak:.2f} exceeded full scale — normalizing before export.")
                data = data / peak
        written: Dict[str, Path] = {}
        for fmt in _clean_formats(formats):
            target = base.with_suffix(f".{fmt}")
            self._write_audio(target, data, sample_rate, fmt)
            written[fmt] = target
        return written

    def _write_audio(
        self, path: Path, data: np.ndarray, sample_rate: int, fmt: str
    ) -> None:
        """Write one file in ``fmt`` (``wav``/``flac`` via soundfile, ``mp3`` via LAME)."""
        if fmt == "mp3":
            self._write_mp3(path, data, sample_rate)
            return
        try:
            import soundfile as sf
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(
                f"soundfile is not installed, so {fmt.upper()} export is unavailable. "
                "Re-run the bootstrap."
            ) from exc
        try:
            if fmt == "flac":
                sf.write(
                    str(path), data, int(sample_rate), format="FLAC", subtype="PCM_24"
                )
            else:
                sf.write(str(path), data, int(sample_rate), subtype="PCM_24")
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"Could not write {path.name}: {exc}") from exc

    def _write_mp3(self, path: Path, data: np.ndarray, sample_rate: int) -> None:
        """Encode 320 kbps CBR at LAME's slowest/best setting.

        lameenc wraps LAME and is LGPL-3.0 (the MP3 patents expired in 2017), so
        it is licensed like the PySide6 the app already ships.
        """
        try:
            import lameenc
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(
                "MP3 export needs the 'lameenc' package, which is not installed.\n\n"
                "Re-run run.bat / run.sh (it installs requirements.txt), or untick MP3."
            ) from exc

        channels = 1 if data.ndim == 1 else int(data.shape[1])
        pcm = np.clip(np.asarray(data, dtype=np.float32), -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2")  # LAME takes interleaved 16-bit PCM

        encoder = lameenc.Encoder()
        encoder.set_bit_rate(MP3_BITRATE_KBPS)
        encoder.set_in_sample_rate(int(sample_rate))
        encoder.set_channels(channels)
        encoder.set_quality(0)  # 0 = slowest, best quality (7 = fastest)
        try:
            payload = encoder.encode(pcm.tobytes())
            payload += encoder.flush()
            path.write_bytes(payload)
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"Could not write {path.name}: {exc}") from exc

    def _write_sidecar(self, path: Path, result: TrackResult) -> None:
        """Write the metadata JSON beside the WAV.

        Only the file name is recorded, never the absolute path: the sidecar
        travels with ``./outputs``, so a moved (or copied) application folder
        still lists its earlier tracks. ``app._load_existing_outputs`` resolves
        the name against the folder the sidecar lives in, and still accepts the
        absolute ``path`` written by older versions.
        """
        try:
            meta = result.as_meta(include_peaks=True)
            meta.pop("path", None)
            meta = {"filename": path.name, **meta}
            path.with_suffix(".json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
        except OSError:
            pass  # metadata is a convenience, never a hard failure

    # -- misc --------------------------------------------------------------
    @staticmethod
    def _emit(callback: Optional[ProgressFn], percent: int, text: str) -> None:
        """Forward a log line to the worker signal when one was supplied."""
        if callback is None:
            return
        try:
            callback(percent, text)
        except Exception:  # noqa: BLE001 - UI callbacks must not kill generation
            pass


# ---------------------------------------------------------------------------
# module helpers
# ---------------------------------------------------------------------------
def _pipeline_class_name(directory: Path) -> Optional[str]:
    """Read _class_name from a diffusers model_index.json."""
    index = Path(directory) / "model_index.json"
    try:
        return json.loads(index.read_text(encoding="utf-8")).get("_class_name")
    except Exception:  # noqa: BLE001
        return None


def _pipeline_sample_rate(pipeline: Any) -> Optional[int]:
    """Read a pipeline's sample rate from its config, if present."""
    vae = getattr(pipeline, "vae", None)
    if vae is None:
        return None
    for candidate in (
        getattr(vae, "sampling_rate", None),
        getattr(getattr(vae, "config", None), "sampling_rate", None),
    ):
        try:
            if candidate:
                return int(candidate)
        except (TypeError, ValueError):
            continue
    return None

#!/usr/bin/env python3
"""Best-effort FlashAttention 2 installer for this app's environment.

FlashAttention 2 is a compiled CUDA extension: a wheel only works when its
Python tag, PyTorch version and CUDA version match the interpreter exactly, so
``pip install flash-attn`` on Windows either finds nothing (the sdist needs MSVC
plus the CUDA toolkit and compiles for a very long time) or quietly breaks the
environment. This script instead *resolves a prebuilt wheel* for the stack that
is actually installed, installs it with ``--no-deps`` so torch can never be
touched, and verifies it with a real GPU kernel launch.

Why it is worth having: ``stable_audio_3/models/transformer.py`` imports
``flash_attn_func`` / ``flash_attn_varlen_func`` and ``flash_attn.bert_padding``.
When they import, the attention layers take the fused CUDA kernel (faster and
measurably lighter on VRAM); when they do not, the model prints "flash_attn not
installed, disabling Flash Attention" and falls back to plain SDPA. It is an
optimisation, never a requirement, so every failure here is non-fatal.

Usage:
    python flash_attn_setup.py                 # install if missing, verify
    python flash_attn_setup.py --check         # report status only
    python flash_attn_setup.py --force         # ignore the failure marker
    python flash_attn_setup.py --dry-run       # resolve a wheel, do not install
    python flash_attn_setup.py --dry-run --simulate "cu128 torch2.7.1 cp311 win_amd64"
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Repositories that publish prebuilt flash-attn wheels. Order matters only for
# tie-breaking: the scorer below decides on version/ABI fit, not on this list.
REPOS = (
    "sdbds/flash-attention-for-windows",
    "kingbri1/flash-attention",
    "Dao-AILab/flash-attention",
    "mjun0812/flash-attention-prebuild-wheels",
)
MAX_RELEASES = 6

# e.g. flash_attn-<version>+cu128torch2.7.1cxx11abiFALSEfullbackward-cp311-cp311-win_amd64.whl
ASSET_RE = re.compile(
    r"^flash_attn-(?P<version>\d[^-+]*)\+cu(?P<cu>\d+)torch(?P<torch>\d+\.\d+(?:\.\d+)?)"
    r"[^-]*-(?P<py>cp\d{3})-cp\d{3}-(?P<plat>[a-z0-9_]+)\.whl$",
    re.IGNORECASE,
)

MARKER_NAME = ".no-flash-attn"


# --------------------------------------------------------------------------- env
def _python_tag() -> str:
    """Return the CPython wheel tag (cp311) for a version tuple."""
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _platform_tag() -> Optional[str]:
    """Return the wheel platform tag for the running OS and arch."""
    if sys.platform.startswith("win"):
        return "win_amd64"
    if sys.platform.startswith("linux"):
        return "linux_x86_64"
    return None


def _cuda_tag(raw: str) -> str:
    """'12.8' -> '128' (the tag used in wheel names)."""
    digits = re.sub(r"\D", "", raw or "")
    return digits or "0"


def detect_environment(simulate: str = "") -> Dict[str, str]:
    """Describe the running stack, or parse a ``--simulate`` description."""
    if simulate:
        env = {"cu": "", "torch": "", "py": "", "plat": "", "gpu": "simulated"}
        for token in simulate.split():
            low = token.lower()
            if low.startswith("cu"):
                env["cu"] = _cuda_tag(low[2:] or "0")
            elif low.startswith("torch"):
                env["torch"] = low[5:]
            elif low.startswith("cp"):
                env["py"] = low
            elif "linux" in low:
                env["plat"] = "linux_x86_64"
            elif "win" in low:
                env["plat"] = "win_amd64"
        return env

    env = {"cu": "0", "torch": "0", "py": _python_tag(), "plat": _platform_tag() or "", "gpu": ""}
    try:
        import torch  # imported lazily: this script must stay cheap to skip
    except Exception as exc:  # noqa: BLE001
        env["gpu"] = f"torch unavailable ({exc.__class__.__name__})"
        return env
    env["torch"] = torch.__version__.split("+")[0]
    if not getattr(torch.version, "cuda", None):
        env["gpu"] = "PyTorch has no CUDA build"
        return env
    env["cu"] = _cuda_tag(torch.version.cuda)
    try:
        if not torch.cuda.is_available():
            env["gpu"] = f"CUDA {torch.version.cuda} present but no device is usable"
            return env
        major, minor = torch.cuda.get_device_capability(0)
        env["gpu"] = f"{torch.cuda.get_device_name(0)} (sm_{major}{minor})"
        if (major, minor) < (8, 0):
            env["gpu"] += " - unsupported by FlashAttention 2 (needs sm_80+)"
    except Exception as exc:  # noqa: BLE001
        env["gpu"] = f"device probe failed ({exc.__class__.__name__})"
    return env


def _gpu_usable(env: Dict[str, str]) -> bool:
    """True when the GPU is Ampere or newer, which flash-attn requires."""
    gpu = env.get("gpu", "")
    return bool(gpu) and "unsupported" not in gpu and "no device" not in gpu and "no CUDA" not in gpu \
        and "unavailable" not in gpu and "failed" not in gpu


# ---------------------------------------------------------------------- resolving
def _github_get(url: str) -> Optional[object]:
    """GET a GitHub API URL with a short timeout."""
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "simple-sound-generator-flash-attn-setup",
            "Accept": "application/vnd.github+json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        return None


def _version_key(version: str) -> Tuple[int, ...]:
    """Sort key for dotted version strings."""
    return tuple(int(part) for part in re.findall(r"\d+", version))


def _minor(version: str) -> str:
    """Parse the minor version from a torch build string."""
    parts = version.split(".")
    return ".".join(parts[:2])


def _cuda_fits(asset_cu: str, env_cu: str) -> bool:
    """Accept 'cu128' for 12.8 and the generic 'cu12' Linux naming."""
    asset_cu, env_cu = str(asset_cu), str(env_cu)
    if asset_cu == env_cu:
        return True
    return len(asset_cu) == 2 and env_cu.startswith(asset_cu)


def collect_assets(env: Dict[str, str], log=print) -> List[Dict[str, object]]:
    """Every wheel asset from the known repos that fits this stack."""
    fits: List[Dict[str, object]] = []
    for repo in REPOS:
        releases = _github_get(f"https://api.github.com/repos/{repo}/releases?per_page={MAX_RELEASES}")
        if not isinstance(releases, list):
            continue
        for release in releases:
            for asset in release.get("assets") or []:
                name = asset.get("name") or ""
                match = ASSET_RE.match(name)
                if not match:
                    continue
                info = match.groupdict()
                if env["plat"] and info["plat"] != env["plat"]:
                    continue
                if env["py"] and info["py"] != env["py"]:
                    continue
                if env["cu"] and not _cuda_fits(info["cu"], env["cu"]):
                    continue
                if env["torch"] and _minor(info["torch"]) != _minor(env["torch"]):
                    continue
                fits.append(
                    {
                        **info,
                        "name": name,
                        "repo": repo,
                        "url": asset.get("browser_download_url"),
                        "size_mb": round((asset.get("size") or 0) / (1024 * 1024), 1),
                        "exact_torch": info["torch"] == env["torch"],
                        "exact_cuda": info["cu"] == env["cu"],
                    }
                )
    return fits


def choose_wheel(fits: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    """Prefer the exact torch build, then the exact CUDA tag, then newest."""
    if not fits:
        return None
    return max(
        fits,
        key=lambda item: (
            bool(item["exact_torch"]),
            bool(item["exact_cuda"]),
            _version_key(str(item["version"])),
        ),
    )


# ---------------------------------------------------------------------- install
def _pip_install(url: str) -> Tuple[bool, str]:
    """Install one wheel with --no-deps, streaming its output."""
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-deps",
        url,
    ]
    print(f"      pip install --no-deps {url.rsplit('/', 1)[-1]}")
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{exc.__class__.__name__}: {exc}"
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()
        return False, tail[-1] if tail else f"pip exited with {result.returncode}"
    return True, ""


def verify_runtime() -> Tuple[bool, str]:
    """Import flash_attn and run a real kernel launch on the GPU."""
    try:
        import torch
        import flash_attn
        from flash_attn import flash_attn_func
    except Exception as exc:  # noqa: BLE001
        return False, f"import failed: {exc.__class__.__name__}: {exc}"
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    try:
        q = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.float16)
        out = flash_attn_func(q, q, q)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        return False, f"kernel launch failed: {exc.__class__.__name__}: {exc}"
    if tuple(out.shape) != tuple(q.shape):
        return False, f"unexpected output shape {tuple(out.shape)}"
    return True, getattr(flash_attn, "__version__", "unknown")


def _installed_version() -> Optional[str]:
    """Return the installed flash_attn version, or None."""
    try:
        import flash_attn

        return getattr(flash_attn, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return None


def _marker_path() -> Path:
    """One marker per environment, stored inside the venv (or next to python)."""
    return Path(sys.prefix) / MARKER_NAME


def _write_marker(reason: str) -> None:
    """Record a failed attempt so later launches skip it."""
    try:
        _marker_path().write_text(reason.strip() + "\n", encoding="utf-8")
    except OSError:
        pass


def _read_marker() -> str:
    """Read the recorded failure marker, if any."""
    try:
        return _marker_path().read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _clear_marker() -> None:
    """Forget a recorded failure after a successful install."""
    try:
        _marker_path().unlink()
    except OSError:
        pass


# --------------------------------------------------------------------------- main
def main(argv: Optional[List[str]] = None) -> int:
    """Resolve, install and smoke-test a matching FlashAttention wheel."""
    parser = argparse.ArgumentParser(description="Install a matching prebuilt FlashAttention 2 wheel.")
    parser.add_argument("--check", action="store_true", help="report status and exit")
    parser.add_argument("--force", action="store_true", help="ignore the failure marker and retry")
    parser.add_argument("--dry-run", action="store_true", help="resolve a wheel without installing")
    parser.add_argument(
        "--simulate",
        default="",
        metavar="ENV",
        help='pretend to run on another stack, e.g. "cu128 torch2.7.1 cp311 win_amd64"',
    )
    args = parser.parse_args(argv)

    env = detect_environment(args.simulate)
    label = f"torch {env['torch']} | cu{env['cu']} | {env['py']} | {env['plat'] or sys.platform}"
    print(f"      FlashAttention 2 · {label}")

    if args.check:
        version = _installed_version()
        if version:
            ok, detail = verify_runtime()
            print(f"      installed {version} · GPU kernel {'OK' if ok else 'FAILED: ' + detail}")
            return 0 if ok else 1
        print(f"      not installed{(' · ' + _read_marker()) if _read_marker() else ''}")
        return 1

    if not _gpu_usable(env):
        print(f"      skipped: {env.get('gpu') or 'no usable CUDA device'}")
        return 0

    if _installed_version():
        ok, detail = verify_runtime()
        if ok:
            print(f"      already active ({detail}) - nothing to do")
            _clear_marker()
            return 0
        print(f"      installed but not usable ({detail}) - reinstalling")

    if not args.force and not args.dry_run and not args.simulate:
        marker = _read_marker()
        if marker:
            print(f"      skipped: previous attempt failed ({marker.splitlines()[0]})")
            print("      retry with: python flash_attn_setup.py --force")
            return 0

    fits = collect_assets(env)
    wheel = choose_wheel(fits)
    if wheel is None:
        reason = f"no prebuilt wheel for {label}"
        print(f"      {reason}")
        if not args.dry_run:
            _write_marker(reason)
        return 0

    exact = "exact" if wheel["exact_torch"] else f'torch {wheel["torch"]} build'
    print(f'      wheel: {wheel["name"]} ({wheel["size_mb"]} MB, {exact})')
    if args.dry_run:
        print(f'      url: {wheel["url"]}')
        return 0

    ok, detail = _pip_install(str(wheel["url"]))
    if not ok:
        reason = f"install failed: {detail}"
        print(f"      {reason}")
        _write_marker(reason)
        return 0

    ok, detail = verify_runtime()
    if not ok:
        reason = f"installed but unusable: {detail}"
        print(f"      {reason}")
        _write_marker(reason)
        return 0

    _clear_marker()
    print(f"      FlashAttention {detail} active - SA3 attention now uses it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

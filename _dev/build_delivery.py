"""Build a clean, versioned delivery copy of Simple Sound Generator.

Dev-only helper: it never touches the working folder (``.venv/``, ``config/models/``,
``outputs/`` and ``logs/`` stay where they are, so the app keeps running here).
It writes exactly the files a user needs into::

    delivery/SimpleSoundGenerator/                                  <- the copy
    delivery/SimpleSoundGenerator-v<version>-<date>-<time>.zip      <- the artifact

Artifact policy — nothing built earlier is ever thrown away:

* a new version always adds a new zip and leaves every older zip in place;
* rebuilding the same version with unchanged files reuses the existing zip
  instead of piling up byte-identical duplicates;
* rebuilding the same version after a code change adds a new zip next to the
  old one (same version, different content), so both stay available.

The version comes from ``APP_VERSION`` in ``code/app.py``.

Run it with the project interpreter::

    .venv\\Scripts\\python.exe _dev\\build_delivery.py
"""

from __future__ import annotations

import hashlib
import re
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Everything a user needs. Nothing else is copied.
SHIP: list[str] = [
    "README.md",
    "LICENSE",
    ".gitignore",
    "run.bat",
    "run.sh",
    "code/app.py",
    "code/audio_engine.py",
    "code/model_manager.py",
    "code/paths.py",
    "code/flash_attn_setup.py",
    "code/requirements.txt",
    "config/models.json",
]

#: Anything matching these must never reach a delivery.
FORBIDDEN_DIRS = {
    ".venv", ".python", "models", "outputs", "logs", "__pycache__",
    ".freebuff", ".git", "delivery", "_dev",
}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".bak", ".zip", ".log", ".ini", ".tmp"}
FORBIDDEN_NAMES = {".hf_token", "settings.ini", "does-not-exist.json"}

VERSION_RE = re.compile(r'^APP_VERSION\s*=\s*"([^"]+)"', re.MULTILINE)
ARTIFACT_GLOB = "SimpleSoundGenerator-v*.zip"


def fail(message: str) -> None:
    print(f"BUILD FAILED: {message}", file=sys.stderr)
    raise SystemExit(1)


def app_version() -> str:
    """Read ``APP_VERSION`` straight from code/app.py (no PyQt/PySide import needed)."""
    match = VERSION_RE.search((ROOT / "code" / "app.py").read_text(encoding="utf-8"))
    if not match:
        fail('APP_VERSION = "..." not found in code/app.py')
    version = match.group(1).strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+[-+.\w]*", version):
        fail(f"APP_VERSION {version!r} is not a plain version string")
    return version


def artifact_path(delivery: Path, version: str) -> Path:
    """Timestamped zip name; a suffix keeps rapid rebuilds distinct."""
    base = f"SimpleSoundGenerator-v{version}-{datetime.now():%Y%m%d-%H%M%S}"
    candidate = delivery / f"{base}.zip"
    counter = 2
    while candidate.exists():
        candidate = delivery / f"{base}-{counter}.zip"
        counter += 1
    return candidate


def ship_fingerprint(target: Path) -> str:
    """Content hash of the staged file set (names included, order fixed)."""
    digest = hashlib.sha256()
    for name in SHIP:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((target / name).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def artifact_matches(zip_path: Path, fingerprint: str) -> bool:
    """True when this existing zip holds exactly the staged content."""
    try:
        with zipfile.ZipFile(zip_path) as archive:
            names = set(archive.namelist())
            digest = hashlib.sha256()
            for name in SHIP:
                arcname = f"SimpleSoundGenerator/{name}"
                if arcname not in names:
                    return False
                digest.update(name.encode("utf-8"))
                digest.update(b"\0")
                digest.update(archive.read(arcname))
                digest.update(b"\0")
            return digest.hexdigest() == fingerprint
    except (OSError, zipfile.BadZipFile):
        return False


def main() -> int:
    missing = [name for name in SHIP if not (ROOT / name).is_file()]
    if missing:
        fail(f"missing files: {', '.join(missing)}")

    version = app_version()
    delivery = ROOT / "delivery"
    target = delivery / "SimpleSoundGenerator"

    # Only the staging folder is rebuilt. Existing zips are never deleted: every
    # previously built version stays available side by side.
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    for name in SHIP:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)

    # Guard against a stray file sneaking in through the ship list itself.
    problems: list[str] = []
    for path in target.rglob("*"):
        relative = path.relative_to(target)
        if any(part in FORBIDDEN_DIRS for part in relative.parts):
            problems.append(f"{relative} (forbidden directory)")
        elif path.is_file() and (
            path.name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES
        ):
            problems.append(f"{relative} (forbidden file)")
    if problems:
        fail("refusing to ship: " + "; ".join(problems))

    fingerprint = ship_fingerprint(target)
    same_version = sorted(delivery.glob(f"SimpleSoundGenerator-v{version}-*.zip"))
    identical = [zip_path for zip_path in same_version
                 if artifact_matches(zip_path, fingerprint)]

    if identical:
        zip_path = identical[0]
        reused = True
    else:
        zip_path = artifact_path(delivery, version)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in SHIP:
                archive.write(target / name, Path("SimpleSoundGenerator") / name)
        reused = False

    total = sum((target / name).stat().st_size for name in SHIP)
    print(f"version: {version}")
    print(f"folder : {target.relative_to(ROOT)}")
    if reused:
        print(f"zip    : {zip_path.relative_to(ROOT)} (already up to date, reused)")
    else:
        print(f"zip    : {zip_path.relative_to(ROOT)}  ({zip_path.stat().st_size / 1024:.0f} KiB)")
    print(f"files  : {len(SHIP)}  ({total / 1024:.0f} KiB uncompressed)")

    retained = sorted(delivery.glob(ARTIFACT_GLOB))
    print(f"\nartifacts kept in delivery/: {len(retained)}")
    for existing in retained:
        marker = " <- this build" if existing == zip_path else ""
        print(f"  {existing.name}  ({existing.stat().st_size / 1024:.0f} KiB){marker}")

    print("\nNo .venv/, config/models/, config/settings.ini, outputs/, logs/ or config/.hf_token is included:")
    print("the first run on the target machine builds its own runtime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

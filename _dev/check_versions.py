"""Cross-check the delivery zip and the version claims around it.

Runs in CI after ``_dev/build_delivery.py`` and can be used by hand::

    python _dev/check_versions.py
    python _dev/check_versions.py --zip delivery/SimpleSoundGenerator-v<version>-<timestamp>.zip

It fails when

* the zip name or the zipped ``code/app.py`` disagrees with ``APP_VERSION``;
* the zip does not contain exactly the 12 shipped files, or contains extras;
* a shipped text file differs from the working tree (staging drift);
* a pin in ``code/requirements.txt`` is missing, or the README quotes a
  different version for torch, torchaudio or diffusers;
* the launchers disagree on ``PY_VERSION`` or ``PY_RELEASE``, or the README's
  portable CPython version disagrees with them;
* a comment in ``requirements.txt`` restates a different torch pin;
* the screenshot job pins PySide6, numpy or soundfile differently from
  requirements.txt.

The check is version-agnostic: bumping ``APP_VERSION``, a pin or the portable
Python version only requires updating the documentation, never this file.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = "SimpleSoundGenerator"

#: The delivery contract, duplicated here on purpose: adding a shipped file
#: must be a deliberate edit to both lists (and to the release checklist).
EXPECTED_ENTRIES = (
    ".gitignore",
    "LICENSE",
    "README.md",
    "code/app.py",
    "code/audio_engine.py",
    "code/flash_attn_setup.py",
    "code/model_manager.py",
    "code/paths.py",
    "code/requirements.txt",
    "config/models.json",
    "run.bat",
    "run.sh",
)

VERSION_RE = re.compile(r'^APP_VERSION\s*=\s*"([^"]+)"', re.MULTILINE)
PIN_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([0-9][0-9A-Za-z.+-]*)")
PY_VERSION_PATTERNS = {
    "run.sh": r'^PY_VERSION="([^"]+)"',
    "run.bat": r'^set "PY_VERSION=([^"]+)"',
}

#: Both launchers build the same python-build-standalone URL, so a different
#: release stamp would fetch a different snapshot (or 404).
PY_RELEASE_PATTERNS = {
    "run.sh": r'^PY_RELEASE="([^"]+)"',
    "run.bat": r'^set "PY_RELEASE=([^"]+)"',
}

#: README sentences that state a pinned version. Every match must equal the
#: requirements pin; each package must still be stated at least once, so a
#: README that drops its version claims cannot pass silently.
README_PIN_PATTERNS = {
    "torch": (
        r"PyTorch\s+(\d+\.\d+\.\d+)",
        r"torch==(\d+\.\d+\.\d+)",
        r"torch\s*/\s*torchaudio\s+(\d+\.\d+\.\d+)",
        r"\*\*(\d+\.\d+\.\d+) together with torchaudio",
        r"torch (\d+\.\d+\.\d+)\+cpu",
    ),
    "torchaudio": (
        r"torchaudio==(\d+\.\d+\.\d+)",
        r"torch\s*/\s*torchaudio\s+(\d+\.\d+\.\d+)",
        r"together with torchaudio\s+(\d+\.\d+\.\d+)",
        r"torchaudio (\d+\.\d+\.\d+)\+cpu",
    ),
    "diffusers": (
        r"pinned diffusers\s+(\d+\.\d+\.\d+)",
    ),
}

#: Comments in requirements.txt that restate the torch pin. External bounds
#: (torchaudio 2.11.1, the newest torch 2.15) are deliberately not checked:
#: they describe PyPI, not the pin.
REQUIREMENTS_PIN_PATTERNS = {
    "torch": (
        r"PyTorch is pinned to\s+(\d+\.\d+\.\d+)",
        r"The tag already says\s*#?\s*(\d+\.\d+\.\d+)",
        r"e\.g\. (\d+\.\d+\.\d+)\+cu",
        r"never got\s+(\d+\.\d+\.\d+)",
    ),
}

#: The screenshot job installs a torch-free subset of requirements. Its
#: hardcoded specs must stay equal to the pins above.
SCREENSHOT_PACKAGES = ("PySide6", "numpy", "soundfile")
WORKFLOW_PIN_RE = re.compile(r'"([A-Za-z0-9_.-]+)==([^"]+)"')


def fail(errors: list[str]) -> int:
    """Print every collected mismatch and return the CI failure status."""
    print("VERSION CHECK FAILED:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    return 1


def normalise(data: bytes) -> bytes:
    """Compare text across CRLF/LF checkouts; the zip should hold the source bytes."""
    return data.replace(b"\r\n", b"\n")


def requirement_pins() -> dict[str, str]:
    """Parse ``name==version`` pins, ignoring comments and range suffixes."""
    pins: dict[str, str] = {}
    for line in (ROOT / "code" / "requirements.txt").read_text(encoding="utf-8").splitlines():
        match = PIN_RE.match(line.split("#", 1)[0].strip())
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


def check_readme(pins: dict[str, str], errors: list[str]) -> None:
    """Require the README's pin claims to agree with requirements.txt."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for package, patterns in README_PIN_PATTERNS.items():
        found = [match for pattern in patterns for match in re.findall(pattern, text)]
        if not found:
            errors.append(f"README no longer states the pinned {package} version")
            continue
        for value in sorted(set(found)):
            if value != pins[package]:
                errors.append(
                    f"README says {package} {value}, but requirements.txt pins {pins[package]}"
                )


def check_python_version(errors: list[str]) -> str | None:
    """Keep the portable CPython pin and its release stamp aligned."""
    versions: dict[str, str] = {}
    for name, pattern in PY_VERSION_PATTERNS.items():
        text = (ROOT / name).read_text(encoding="utf-8")
        match = re.search(pattern, text, re.MULTILINE)
        if not match:
            errors.append(f"no PY_VERSION assignment found in {name}")
        else:
            versions[name] = match.group(1)

    releases: dict[str, str] = {}
    for name, pattern in PY_RELEASE_PATTERNS.items():
        text = (ROOT / name).read_text(encoding="utf-8")
        match = re.search(pattern, text, re.MULTILINE)
        if not match:
            errors.append(f"no PY_RELEASE assignment found in {name}")
        else:
            releases[name] = match.group(1)

    if len(set(versions.values())) > 1:
        errors.append(
            "launchers disagree on PY_VERSION: "
            + ", ".join(f"{name} {value}" for name, value in versions.items())
        )
    if len(set(releases.values())) > 1:
        errors.append(
            "launchers disagree on PY_RELEASE: "
            + ", ".join(f"{name} {value}" for name, value in releases.items())
        )
    if not versions:
        return None

    expected = versions.get("run.sh") or next(iter(versions.values()))

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    found = re.findall(r"CPython\s+(\d+\.\d+\.\d+)", readme)
    if not found:
        errors.append("README no longer states the portable CPython version")
    for value in sorted(set(found)):
        if value != expected:
            errors.append(f"README says CPython {value}, but run.sh uses {expected}")
    return expected


def check_requirements_comments(pins: dict[str, str], errors: list[str]) -> None:
    """Require pin restatements in requirements.txt comments to match the pins."""
    text = (ROOT / "code" / "requirements.txt").read_text(encoding="utf-8")
    for package, patterns in REQUIREMENTS_PIN_PATTERNS.items():
        found = [match for pattern in patterns for match in re.findall(pattern, text)]
        if not found:
            errors.append(f"requirements.txt comments no longer restate the {package} pin")
            continue
        for value in sorted(set(found)):
            if value != pins[package]:
                errors.append(
                    f"requirements.txt comment says {package} {value}, "
                    f"but the pin is {pins[package]}"
                )


def check_screenshot_pins(pins: dict[str, str], errors: list[str]) -> None:
    """Keep the screenshot job's hardcoded install pins in sync with requirements."""
    text = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    found: dict[str, set[str]] = {}
    for name, version in WORKFLOW_PIN_RE.findall(text):
        found.setdefault(name.lower(), set()).add(version)
    for package in SCREENSHOT_PACKAGES:
        pin = pins.get(package.lower())
        if pin is None:
            errors.append(f"requirements.txt has no concrete {package} pin")
            continue
        values = found.get(package.lower(), set())
        if not values:
            errors.append(f"workflow no longer pins {package} for the screenshot job")
        for value in sorted(values):
            if value != pin:
                errors.append(
                    f"workflow pins {package} {value}, but requirements.txt pins {pin}"
                )


def newest_zip(version: str) -> Path | None:
    """The likeliest artifact to check: newest zip carrying this APP_VERSION."""
    candidates = [
        path
        for path in (ROOT / "delivery").glob(f"{PACKAGE}-v{version}-*.zip")
        if path.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def check_zip(path: Path, version: str, errors: list[str]) -> None:
    """Verify the artifact's name, entry set, version and staged file content."""
    if not path.is_file():
        errors.append(f"zip not found: {path}")
        return

    name_re = re.compile(
        rf"^{PACKAGE}-v{re.escape(version)}-\d{{8}}-\d{{6}}(?:-\d+)?\.zip$"
    )
    if not name_re.fullmatch(path.name):
        errors.append(f"zip name does not encode APP_VERSION {version}: {path.name}")

    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                errors.append("zip contains duplicate entries")

            entries: dict[str, bytes] = {}
            for name in names:
                if not name.startswith(f"{PACKAGE}/"):
                    errors.append(f"zip entry outside {PACKAGE}/: {name}")
                    continue
                relative = name[len(PACKAGE) + 1:]
                if not relative or relative.endswith("/"):
                    errors.append(f"zip contains a directory entry: {name}")
                    continue
                entries[relative] = archive.read(name)
    except (OSError, zipfile.BadZipFile) as exc:
        errors.append(f"cannot read {path}: {exc}")
        return

    missing = sorted(set(EXPECTED_ENTRIES) - set(entries))
    extra = sorted(set(entries) - set(EXPECTED_ENTRIES))
    if missing:
        errors.append("zip is missing: " + ", ".join(missing))
    if extra:
        errors.append("zip contains unexpected entries: " + ", ".join(extra))

    zipped_app = entries.get("code/app.py")
    if zipped_app is None:
        errors.append("zip has no code/app.py, so APP_VERSION cannot be verified")
    else:
        match = VERSION_RE.search(zipped_app.decode("utf-8", "replace"))
        if not match:
            errors.append("zipped code/app.py has no APP_VERSION")
        elif match.group(1) != version:
            errors.append(
                f"zipped code/app.py says {match.group(1)}, but APP_VERSION is {version}"
            )

    for relative in EXPECTED_ENTRIES:
        if relative not in entries:
            continue
        source = (ROOT / relative).read_bytes()
        if normalise(entries[relative]) != normalise(source):
            errors.append(f"{relative} inside the zip differs from the working tree")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip",
        type=Path,
        help="artifact to inspect (default: newest delivery zip for APP_VERSION)",
    )
    args = parser.parse_args(argv)

    errors: list[str] = []
    app_text = (ROOT / "code" / "app.py").read_text(encoding="utf-8")
    match = VERSION_RE.search(app_text)
    if not match:
        return fail(['APP_VERSION = "..." not found in code/app.py'])
    version = match.group(1).strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+[-+.\w]*", version):
        return fail([f"APP_VERSION {version!r} is not a plain version string"])

    pins = requirement_pins()
    for package in README_PIN_PATTERNS:
        if package not in pins:
            errors.append(f"requirements.txt has no concrete {package} pin")
    python_version = check_python_version(errors)
    if not errors:
        check_readme(pins, errors)
        check_requirements_comments(pins, errors)
        check_screenshot_pins(pins, errors)

    zip_path = args.zip or newest_zip(version)
    if zip_path is None:
        errors.append("no delivery zip found; run _dev/build_delivery.py first")
    else:
        check_zip(zip_path, version, errors)

    if errors:
        return fail(errors)

    print(
        f"OK: APP_VERSION {version}; {zip_path.name} has {len(EXPECTED_ENTRIES)} entries; "
        f"portable CPython {python_version}; README and requirements.txt agree on "
        + ", ".join(f"{name} {pins[name]}" for name in README_PIN_PATTERNS)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

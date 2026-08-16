from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys
from typing import Callable, Iterable


_EXCLUDED_NAMES = {"INCOMPLETE", ".DS_Store"}
_EXCLUDED_SOURCE_DIRECTORIES = {"__pycache__"}
_EXCLUDED_SOURCE_SUFFIXES = {".pyc", ".pyo"}
_EXCLUDED_ACT_DIRECTORIES = {"act", "act_smoke", "act_pilot", "checkpoints"}
_NON_ACT_REQUIREMENTS = {"datasets", "socksio"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_files(
    root: Path,
    files: Iterable[Path],
    *,
    content_filter: Callable[[Path, bytes], bytes] | None = None,
) -> str:
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda value: value.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        if content_filter is None:
            size = path.stat().st_size
            content_hash = bytes.fromhex(sha256_file(path))
        else:
            content = content_filter(path, path.read_bytes())
            size = len(content)
            content_hash = hashlib.sha256(content).digest()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        digest.update(content_hash)
    return digest.hexdigest()


def _act_source_content(path: Path, content: bytes) -> bytes:
    if not path.name.startswith("requirements-lerobot-"):
        return content
    retained: list[bytes] = []
    for line in content.splitlines(keepends=True):
        requirement = line.strip().split(b";", 1)[0]
        name = requirement
        for separator in (b"[", b"<", b">", b"=", b"!", b"~"):
            name = name.split(separator, 1)[0]
        if name.decode("utf-8").lower().replace("_", "-") in _NON_ACT_REQUIREMENTS:
            continue
        retained.append(line)
    return b"".join(retained)


def fingerprint_tree(root: str | Path) -> str:
    directory = Path(root)
    if not directory.is_dir():
        raise ValueError(f"fingerprint root must be a directory: {directory}")
    files = (
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.name not in _EXCLUDED_NAMES
        and not any(part in _EXCLUDED_ACT_DIRECTORIES for part in path.relative_to(directory).parts)
    )
    return _fingerprint_files(directory, files)


def standard_dataset_fingerprint(root: str | Path) -> str:
    directory = Path(root)
    if not directory.is_dir():
        raise ValueError(f"standard dataset root must be a directory: {directory}")
    files: list[Path] = []
    for name in ("data", "videos"):
        subtree = directory / name
        if subtree.is_dir():
            files.extend(
                path
                for path in subtree.rglob("*")
                if path.is_file() and path.name not in _EXCLUDED_NAMES
            )
    meta = directory / "meta"
    for name in ("info.json", "stats.json", "tasks.parquet"):
        path = meta / name
        if path.is_file():
            files.append(path)
    episodes = meta / "episodes"
    if episodes.is_dir():
        files.extend(
            path
            for path in episodes.rglob("*")
            if path.is_file() and path.name not in _EXCLUDED_NAMES
        )
    if not files:
        raise ValueError(f"standard dataset contains no fingerprintable files: {directory}")
    return _fingerprint_files(directory, files)


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def runtime_versions(*, requested_device: str = "auto") -> dict[str, object]:
    import numpy
    import torch

    from interaction_vla.device import resolve_device

    try:
        ffmpeg = subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()[0]
    except (FileNotFoundError, subprocess.SubprocessError, IndexError):
        ffmpeg = "unavailable"
    try:
        resolved_device = str(resolve_device(requested_device))
    except RuntimeError as error:
        resolved_device = f"unavailable: {error}"
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    cuda_device_name = (
        str(torch.cuda.get_device_name(0)) if cuda_device_count > 0 else None
    )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "lerobot": _distribution_version("lerobot"),
        "torch": torch.__version__,
        "torchvision": _distribution_version("torchvision"),
        "mujoco": _distribution_version("mujoco"),
        "numpy": numpy.__version__,
        "ffmpeg": ffmpeg,
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "cuda_device_count": cuda_device_count,
        "cuda_device_name": cuda_device_name,
    }


def git_commit(root: str | Path = ".") -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(root),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unavailable"


def source_fingerprint(root: str | Path = ".") -> str:
    repository = Path(root)
    candidates = (
        repository / "interaction_vla" / "lerobot_bridge",
        repository / "configs",
    )
    files: list[Path] = []
    for directory in candidates:
        if directory.is_dir():
            files.extend(
                path
                for path in directory.rglob("*")
                if path.is_file()
                and path.name not in _EXCLUDED_NAMES
                and path.suffix not in _EXCLUDED_SOURCE_SUFFIXES
                and not any(
                    part in _EXCLUDED_SOURCE_DIRECTORIES
                    for part in path.relative_to(directory).parts
                )
                and (directory.name != "configs" or path.name.startswith("lerobot_"))
            )
    for name in (
        "requirements-lerobot-macos.txt",
        "requirements-lerobot-macos.lock.txt",
        "requirements-lerobot-linux-cuda.txt",
    ):
        path = repository / name
        if path.is_file():
            files.append(path)
    return _fingerprint_files(repository, files, content_filter=_act_source_content)

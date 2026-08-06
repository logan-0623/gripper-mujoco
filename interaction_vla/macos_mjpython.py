from __future__ import annotations

from pathlib import Path
import sys
import sysconfig


def ensure_mjpython_libpython_link(
    *,
    prefix: str | Path,
    libdir: str | Path,
    library_name: str,
) -> Path:
    source = Path(libdir) / library_name
    target = Path(prefix) / "lib" / library_name
    if not source.is_file():
        raise FileNotFoundError(f"libpython library does not exist: {source}")
    if (target.exists() or target.is_symlink()) and target.resolve() == source.resolve():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source)
    return target


def main() -> int:
    libdir = sysconfig.get_config_var("LIBDIR")
    library_name = sysconfig.get_config_var("LDLIBRARY")
    if not libdir or not library_name:
        raise RuntimeError("active Python does not expose LIBDIR and LDLIBRARY")
    target = ensure_mjpython_libpython_link(
        prefix=sys.prefix,
        libdir=libdir,
        library_name=library_name,
    )
    print(f"mjpython libpython ready: {target} -> {target.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

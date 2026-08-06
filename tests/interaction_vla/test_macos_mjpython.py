from importlib import import_module
from pathlib import Path

import pytest


def test_ensure_mjpython_libpython_link_creates_virtualenv_link(
    tmp_path: Path,
) -> None:
    module = import_module("interaction_vla.macos_mjpython")
    prefix = tmp_path / "venv"
    libdir = tmp_path / "uv-python" / "lib"
    source = libdir / "libpython3.12.dylib"
    source.parent.mkdir(parents=True)
    source.touch()

    target = module.ensure_mjpython_libpython_link(
        prefix=prefix,
        libdir=libdir,
        library_name=source.name,
    )

    assert target == prefix / "lib" / source.name
    assert target.is_symlink()
    assert target.resolve() == source.resolve()


def test_ensure_mjpython_libpython_link_is_idempotent(tmp_path: Path) -> None:
    module = import_module("interaction_vla.macos_mjpython")
    prefix = tmp_path / "venv"
    libdir = tmp_path / "uv-python" / "lib"
    source = libdir / "libpython3.12.dylib"
    source.parent.mkdir(parents=True)
    source.touch()

    first = module.ensure_mjpython_libpython_link(
        prefix=prefix,
        libdir=libdir,
        library_name=source.name,
    )
    second = module.ensure_mjpython_libpython_link(
        prefix=prefix,
        libdir=libdir,
        library_name=source.name,
    )

    assert second == first
    assert second.resolve() == source.resolve()


def test_ensure_mjpython_libpython_link_rejects_missing_source(
    tmp_path: Path,
) -> None:
    module = import_module("interaction_vla.macos_mjpython")

    with pytest.raises(FileNotFoundError, match="libpython"):
        module.ensure_mjpython_libpython_link(
            prefix=tmp_path / "venv",
            libdir=tmp_path / "missing",
            library_name="libpython3.12.dylib",
        )


def test_main_uses_the_active_python_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = import_module("interaction_vla.macos_mjpython")
    prefix = tmp_path / "venv"
    libdir = tmp_path / "uv-python" / "lib"
    source = libdir / "libpython3.12.dylib"
    source.parent.mkdir(parents=True)
    source.touch()
    values = {"LIBDIR": str(libdir), "LDLIBRARY": source.name}
    monkeypatch.setattr(module.sys, "prefix", str(prefix))
    monkeypatch.setattr(module.sysconfig, "get_config_var", values.__getitem__)

    result = module.main()

    target = prefix / "lib" / source.name
    assert result == 0
    assert target.resolve() == source.resolve()
    assert str(target) in capsys.readouterr().out

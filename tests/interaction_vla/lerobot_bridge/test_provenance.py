from interaction_vla.lerobot_bridge.provenance import (
    fingerprint_tree,
    sha256_file,
    source_fingerprint,
    standard_dataset_fingerprint,
)


def test_tree_fingerprint_changes_with_content_not_mtime(tmp_path) -> None:
    path = tmp_path / "meta.json"
    path.write_text('{"value": 1}', encoding="utf-8")
    first = fingerprint_tree(tmp_path)
    path.touch()
    assert fingerprint_tree(tmp_path) == first
    path.write_text('{"value": 2}', encoding="utf-8")
    assert fingerprint_tree(tmp_path) != first
    assert len(sha256_file(path)) == 64


def test_source_fingerprint_ignores_python_cache_files(tmp_path) -> None:
    source = tmp_path / "interaction_vla" / "lerobot_bridge"
    source.mkdir(parents=True)
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    baseline = source_fingerprint(tmp_path)

    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "module.cpython-312.pyc").write_bytes(b"runtime cache")

    assert source_fingerprint(tmp_path) == baseline


def test_source_fingerprint_ignores_graph_download_dependencies(tmp_path) -> None:
    source = tmp_path / "interaction_vla" / "lerobot_bridge"
    source.mkdir(parents=True)
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    requirements = tmp_path / "requirements-lerobot-macos.txt"
    lock = tmp_path / "requirements-lerobot-macos.lock.txt"
    requirements.write_text(
        "torch>=2.10,<2.11\nlerobot[dataset,training]==0.6.1\n",
        encoding="utf-8",
    )
    lock.write_text(
        "torch==2.10.0\nlerobot==0.6.1\n",
        encoding="utf-8",
    )
    baseline = source_fingerprint(tmp_path)

    requirements.write_text(
        "torch>=2.10,<2.11\ndatasets==4.8.5\nsocksio==1.0.0\n"
        "lerobot[dataset,training]==0.6.1\n",
        encoding="utf-8",
    )
    lock.write_text(
        "socksio==1.0.0\ntorch==2.10.0\nlerobot==0.6.1\n",
        encoding="utf-8",
    )

    assert source_fingerprint(tmp_path) == baseline

    lock.write_text(
        "socksio==1.0.0\ntorch==2.11.0\nlerobot==0.6.1\n",
        encoding="utf-8",
    )
    assert source_fingerprint(tmp_path) != baseline


def test_standard_dataset_fingerprint_binds_video_but_not_bridge_metadata(
    tmp_path,
) -> None:
    video = tmp_path / "videos" / "agent" / "file.mp4"
    data = tmp_path / "data" / "file.parquet"
    info = tmp_path / "meta" / "info.json"
    bridge = tmp_path / "meta" / "bridge_provenance.json"
    for path, content in (
        (video, b"video"),
        (data, b"rows"),
        (info, b"info"),
        (bridge, b"bridge"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    baseline = standard_dataset_fingerprint(tmp_path)

    bridge.write_bytes(b"changed bridge metadata")
    assert standard_dataset_fingerprint(tmp_path) == baseline
    video.write_bytes(b"changed video")
    assert standard_dataset_fingerprint(tmp_path) != baseline

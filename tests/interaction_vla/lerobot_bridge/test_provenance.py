from interaction_vla.lerobot_bridge.provenance import fingerprint_tree, sha256_file


def test_tree_fingerprint_changes_with_content_not_mtime(tmp_path) -> None:
    path = tmp_path / "meta.json"
    path.write_text('{"value": 1}', encoding="utf-8")
    first = fingerprint_tree(tmp_path)
    path.touch()
    assert fingerprint_tree(tmp_path) == first
    path.write_text('{"value": 2}', encoding="utf-8")
    assert fingerprint_tree(tmp_path) != first
    assert len(sha256_file(path)) == 64

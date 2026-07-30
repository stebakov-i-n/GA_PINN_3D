import json

from cfd.manifests import load_manifest_entries, normalize_manifest_key, summarize_entries


def test_normalize_manifest_key_handles_windows_and_posix():
    assert normalize_manifest_key(r"0018_H_AO_COA\1_1.stl") == "0018_H_AO_COA/1_1.stl"
    assert normalize_manifest_key("0018_H_AO_COA/1_1.stl") == "0018_H_AO_COA/1_1.stl"


def test_manifest_adapter_lists_only_accepted_with_split(tmp_path):
    split = tmp_path / "full_split.json"
    selection = tmp_path / "geometry_selection.json"
    split.write_text(json.dumps({"train": ["case_a"], "val": ["case_b"], "test": ["case_c"]}), encoding="utf-8")
    selection.write_text(
        json.dumps(
            {
                r"case_a\1_1.stl": "accept",
                "case_a/1_2.stl": "reject",
                "case_b/1_1.stl": "accept",
                "case_z/1_1.stl": "accept",
            }
        ),
        encoding="utf-8",
    )
    entries, warnings = load_manifest_entries(split, selection)
    assert [e.normalized_key for e in entries] == ["case_a/1_1.stl", "case_b/1_1.stl"]
    assert summarize_entries(entries)["by_split"] == {"train": 1, "validation": 1, "test": 0}
    assert any("absent from split" in w for w in warnings)

"""Source generators must work without ignored historical run outputs."""
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from pdn_lab.data import sha256
from pdn_lab.schema import PROJECT_ROOT, read_jsonl, write_json, write_jsonl


@pytest.fixture
def preparation_scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT_ROOT))
    modules = [
        importlib.import_module("scripts.legacy_stress_cases"),
        importlib.import_module("scripts.audit_stress_cases"),
        importlib.import_module("scripts.prepare_hard_data"),
    ]
    original_open = Path.open

    def without_local_reports(path, *args, **kwargs):
        if {"reports", "runs"} & set(path.parts):
            raise AssertionError(f"Preparation depends on a local run artifact: {path}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", without_local_reports)
    return modules


def record_digest(records):
    # LF serialization makes this fixture check independent of the host OS.
    text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_legacy_source_restores_the_frozen_exclusion_set(preparation_scripts):
    legacy, audit, _ = preparation_scripts
    records = legacy.legacy_cases()
    assert len(records) == 42371
    assert record_digest(records) == "067b2973be1ff90fbdc561b9c8f65239cf00ae9d7ef607a7898ff302ba162c2f"
    unicode_cards = [
        record for record in records
        if record["category"] == "unicode_number_separators"
        and record["entities"][0][0] == "CARD"
    ]
    assert len(unicode_cards) == 6
    assert all(not audit.luhn(record["text"]) for record in unicode_cards)


def test_audit_reconstructs_frozen_cases_without_reports(preparation_scripts, tmp_path):
    legacy, audit, _ = preparation_scripts
    output = tmp_path / "audited"
    result = audit.prepare(output)
    records = read_jsonl(output / "cases.jsonl")
    assert result["texts"] == len(records) == 42377
    assert len(result["corrected_annotations"]) == 6
    assert record_digest(records) == "1cd378796743084e735ef4de2de4ee7799ef072b35b41a04581f38b52645f6ca"

    legacy_path = tmp_path / "frozen-input.jsonl"
    write_jsonl(legacy_path, legacy.legacy_cases())
    explicit_output = tmp_path / "explicit"
    assert audit.prepare(explicit_output, legacy_path) == result
    assert (explicit_output / "cases.jsonl").read_bytes() == (output / "cases.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        audit.prepare(output)


def test_hard_data_preparation_needs_only_original_data(preparation_scripts, tmp_path):
    _, _, hard = preparation_scripts
    source = tmp_path / "source"
    processed = source / "processed"
    original_bytes = {}
    for split in ("train", "validation", "test"):
        record = {
            "id": f"portable-fixture/{split}", "source": "fixture",
            "text": f"Исходная отрицательная запись раздела {split}.", "entities": [],
        }
        # A custom input root and a preserved CRLF test file exercise portability.
        data = (json.dumps(record, ensure_ascii=False) + "\r\n").encode("utf-8")
        processed.mkdir(parents=True, exist_ok=True)
        (processed / f"{split}.jsonl").write_bytes(data)
        original_bytes[split] = data
    write_json(processed / "manifest.json", {
        "schema_version": 1,
        "files": {split: sha256(processed / f"{split}.jsonl") for split in original_bytes},
    })
    output = tmp_path / "extended"
    manifest = hard.prepare(source, output)
    assert manifest["augmentation_counts"] == {"train": 4500, "validation": 900, "holdout": 1000}
    assert manifest["statistics"]["train"]["texts"] == 4501
    assert manifest["statistics"]["validation"]["texts"] == 901
    assert (output / "processed/test.jsonl").read_bytes() == original_bytes["test"]
    text_sets = [
        {record["text"] for record in read_jsonl(output / "processed" / f"{split}.jsonl")}
        for split in ("train", "validation", "test", "holdout")
    ]
    for index, texts in enumerate(text_sets):
        assert all(texts.isdisjoint(other) for other in text_sets[index + 1:])
    for split, expected in manifest["files"].items():
        assert sha256(output / "processed" / f"{split}.jsonl") == expected
    with pytest.raises(FileExistsError):
        hard.prepare(source, output)
    assert all((processed / f"{split}.jsonl").read_bytes() == data for split, data in original_bytes.items())

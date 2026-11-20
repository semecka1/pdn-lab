"""Reconstruct the frozen v1 cases without keeping local run reports.

Six Unicode card examples changed after v1: their original annotation accepted
4111 4111 4111 4111 incorrectly. Those original texts are necessary to
reproduce the exact augmentation exclusion set. The audit script fixes their
annotations separately.
"""
from __future__ import annotations

from pathlib import Path

from pdn_lab.schema import read_jsonl
from pdn_lab.stress_cases import generate_cases


def legacy_cases(path: Path | None = None) -> list[dict]:
    if path is not None:
        return read_jsonl(path)
    records = generate_cases()
    changed = 0
    for record in records:
        if record["category"] != "unicode_number_separators":
            continue
        label, start, end = record["entities"][0]
        if label != "CARD":
            continue
        value = record["text"][start:end]
        separator = value[4]
        if value != separator.join(["4111", "1111", "1111", "1111"]):
            raise ValueError("The source Unicode card fixture changed unexpectedly")
        original = separator.join(["4111"] * 4)
        record["text"] = record["text"][:start] + original + record["text"][end:]
        record["note"] = "Типографские/многострочные разделители шире заявленных ASCII-форматов."
        changed += 1
    if changed != 6:
        raise ValueError(f"Expected six historical Unicode card fixtures, got {changed}")
    return records

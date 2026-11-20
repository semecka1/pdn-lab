"""Независимость и целостность ручного контекста, без загрузки модели/GPU."""
from collections import Counter
from pathlib import Path
import runpy

from pdn_lab.schema import entities_of, to_bio


GENERATOR = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "prepare_context_challenge_v3.py"))


def test_context_challenge_is_reproducible_and_manually_annotated():
    first = GENERATOR["build_cases"]()
    assert first == GENERATOR["build_cases"]()
    assert len(first) == len({record["text"] for record in first}) == 176
    assert set(Counter(record["category"] for record in first).values()) == {16}
    for record in first:
        to_bio(record["text"], entities_of(record))


def test_context_challenge_preserves_positive_counterexamples():
    records = GENERATOR["build_cases"]()
    animal = next(record for record in records if record["text"] == "Лев спит в вольере зоопарка.")
    human = next(record for record in records if record["text"] == "Лев работает инженером.")
    assert animal["entities"] == []
    assert human["entities"] == [["NAME", 0, 3]]
    quote = next(record for record in records if record["text"] == "Редактор написал: «Автор — САЛТЫКОВ-ЩЕДРИН».")
    assert [(entity.label, quote["text"][entity.start:entity.end]) for entity in entities_of(quote)] == [("SURNAME", "САЛТЫКОВ-ЩЕДРИН")]


def test_marker_offsets_account_for_unicode_and_repeated_surfaces():
    text, entities = GENERATOR["unmark"]("«Петров»; [[SURNAME|Петров]] — сотрудник; [[NAME|Лев]].")
    assert [(label, text[start:end]) for label, start, end in entities] == [("SURNAME", "Петров"), ("NAME", "Лев")]
    assert entities[0][1] == text.index("Петров", text.index(";") + 1)

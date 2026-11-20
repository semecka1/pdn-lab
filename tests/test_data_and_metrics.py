import hashlib

import pytest

from pdn_lab.data import TEMPLATES, generate_synthetic, prepare
from pdn_lab.metrics import evaluate_predictions
from pdn_lab.schema import ATTRIBUTES, Entity, entities_of, to_bio


def test_false_positives_on_o_are_counted():
    records = [{"id": "1", "source": "manual", "text": "Иван пришёл", "entities": [["NAME", 0, 4]]}]
    report = evaluate_predictions(records, [[Entity("NAME", 0, 4), Entity("NAME", 5, 11)]])
    assert report["token_weighted_f1_without_o"] == pytest.approx(2 / 3)
    assert report["strict_entities_by_attribute"]["NAME"]["fp"] == 1


def test_wrong_boundary_is_error_even_when_type_is_right():
    records = [{"id": "1", "source": "manual", "text": "Анна Мария", "entities": [["NAME", 0, 10]]}]
    report = evaluate_predictions(records, [[Entity("NAME", 0, 4)]])
    assert report["strict_entity_micro"]["f1"] == 0


def test_synthetic_reproducible_with_independent_templates():
    first = generate_synthetic("test", 100, 42)
    assert first == generate_synthetic("test", 100, 42)
    assert first != generate_synthetic("test", 100, 43)
    for split_a, split_b in (("train", "test"), ("train", "validation"), ("validation", "test")):
        assert not set(TEMPLATES[split_a]) & set(TEMPLATES[split_b])
    for record in first:
        to_bio(record["text"], entities_of(record))


def test_preparation_counts_integrity_and_no_text_leakage(tmp_path):
    manifest = prepare(tmp_path, train_count=100, validation_count=100, test_count=100, synthetic_only=True)
    assert all(manifest["statistics"]["test"]["texts_per_attribute"][label] > 50 for label in ATTRIBUTES)
    for split, digest in manifest["files"].items():
        assert hashlib.sha256((tmp_path / "processed" / f"{split}.jsonl").read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        prepare(tmp_path, synthetic_only=True)


def test_too_few_test_texts_rejected(tmp_path):
    with pytest.raises(ValueError, match="больше 50"):
        prepare(tmp_path, train_count=10, validation_count=10, test_count=20, synthetic_only=True)

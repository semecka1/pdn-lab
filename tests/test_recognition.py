import pytest

from pdn_lab.detector import Detector
from pdn_lab.rules import find_structured
from pdn_lab.schema import Entity, from_bio, to_bio
from pdn_lab.validators import make_card, make_inn, snils_check, valid_card, valid_inn, valid_phone, valid_snils


@pytest.mark.parametrize("number", ["7707083893", "500100732259", "7736207543"])
def test_known_inn(number):
    assert valid_inn(number)
    assert not valid_inn(number[:-1] + str((int(number[-1]) + 1) % 10))


@pytest.mark.parametrize("number", ["0000000000", "000000000000", "111111111111", "50010073225"])
def test_inn_bad_format_and_degenerate(number):
    assert not valid_inn(number)


def test_snils_external_example_and_boundaries():
    assert valid_snils("112-233-445 95")
    assert not valid_snils("112-233-445 96")
    assert not valid_snils("000-000-000 00")
    assert not valid_snils("001-001-998 00")
    assert snils_check("999999999") == 1
    assert snils_check("900000059") == 0  # сумма 100
    assert snils_check("900000068") == 0  # сумма 101
    assert snils_check("900000159") == 2  # сумма 103


@pytest.mark.parametrize("number", ["4111 1111 1111 1111", "5555-5555-5555-4444", "378282246310005"])
def test_known_luhn_examples(number):
    assert valid_card(number)
    assert not valid_card(number[:-1] + str((int(number[-1]) + 1) % 10))


def test_phone_russian_formats():
    assert valid_phone("+7 (912) 345-67-89")
    assert valid_phone("8 495 123 45 67")
    assert not valid_phone("+1 202 555 0123")
    assert not valid_phone("+7 000 000 00 00")


def test_original_offsets_with_emoji_and_crlf():
    text = "🙂\r\nПочта: student+lab@example.org. Телефон +7 (912) 345-67-89. СНИЛС 112-233-445 95."
    found = find_structured(text)
    assert [(e.label, text[e.start:e.end]) for e in found] == [("EMAIL", "student+lab@example.org"), ("PHONE", "+7 (912) 345-67-89"), ("SNILS", "112-233-445 95")]
    tokens, tags = to_bio(text, found)
    assert from_bio(tokens, tags) == found


def test_invalid_checksums_do_not_fall_back_to_phone():
    text = "ИНН 500100732258; СНИЛС 79123344500; карта 4111111111111112."
    assert find_structured(text) == []


def test_long_numeric_run_is_not_split_into_valid_shorter_numbers():
    assert find_structured("Счёт 999941111111111111119999") == []
    assert find_structured("abc500100732259def") == []


def test_name_and_rule_overlap_resolved_once():
    class FakeNames:
        def predict_many(self, texts, batch_size):
            return [[Entity("NAME", 0, 4), Entity("SURNAME", 5, 9)]]
    detector = Detector(FakeNames())
    assert detector.detect("Иван user@example.org") == [["NAME", 0, 4], ["EMAIL", 5, 21]]


def test_adjacent_names_remain_separate():
    text = "Иван Пётр Иванов"
    entities = [Entity("NAME", 0, 4), Entity("NAME", 5, 9), Entity("SURNAME", 10, 16)]
    tokens, tags = to_bio(text, entities)
    assert tags == ["B-NAME", "B-NAME", "B-SURNAME"]
    assert from_bio(tokens, tags) == entities


def test_missing_model_is_explicit():
    with pytest.raises(ValueError, match="обученная модель"):
        Detector()


def test_generation_and_validation_checks():
    assert valid_inn(make_inn("5001007322"))
    assert valid_card(make_card("411111111111111"))


def test_validators_reject_foreign_characters():
    assert not valid_inn("abc500100732259")
    assert not valid_card("4111x1111x1111x1111")
    assert not valid_snils("112x233x445x95")
    assert not valid_phone("+8 (912) 345-67-89")


def test_cli_reads_crlf_without_offset_shift(tmp_path, capsys):
    import json
    from pdn_lab.cli import main

    text = "🙂\r\nПишите: student@example.org."
    path = tmp_path / "input.txt"
    path.write_bytes(text.encode("utf-8"))
    assert main(["predict", "--rules-only", "--file", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == [["EMAIL", text.index("student"), text.index("student") + len("student@example.org")]]

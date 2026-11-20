import ipaddress
import re

import pytest

from pdn_lab.anonymization import anon, mask
from pdn_lab.course_rules import valid_ogrnip
from pdn_lab.detector import Detector
from pdn_lab.kladr import catalog, level
from pdn_lab.name_replacement import parses
from pdn_lab.validators import digits, make_card, make_inn, snils_check, valid_card, valid_inn, valid_snils


def spans(text, label, pattern):
    return [[label, *m.span()] for m in re.finditer(pattern, text)]


def test_morphology_and_consistency():
    text = "Саша пришла. К Саше вернулись. Женя уже пришёл. С Женей поговорили."
    result = anon(text, spans(text, "NAME", "Саша|Саше|Женя|Женей"), seed=143)
    female, dative, male, instrumental = re.findall(r"(?:^|К |вернулись\. |С )([А-ЯЁ][а-яё]+)", result)
    def lemmas(word, gender, case):
        return {p.normal_form for p in parses(word) if "Name" in p.tag and gender in p.tag and case in p.tag}
    assert lemmas(female, "femn", "nomn") & lemmas(dative, "femn", "datv")
    assert lemmas(male, "masc", "nomn") & lemmas(instrumental, "masc", "ablt")
    assert not any(w in result for w in ("Саша", "Саше", "Женя", "Женей"))


def test_compound_surnames_plural_register_and_initials():
    text = "САЛТЫКОВ-ЩЕДРИН выступил. К Салтыкову-Щедрину подошли. Ивановы уехали, у Ивановых остался ключ."
    new = anon(text, spans(text, "SURNAME", r"САЛТЫКОВ-ЩЕДРИН|Салтыкову-Щедрину|Ивановы[х]?"), seed=6)
    first, to, family, genitive = re.fullmatch(r"(.+) выступил\. К (.+) подошли\. (.+) уехали, у (.+) остался ключ\.", new).groups()
    assert first.isupper() and first != "САЛТЫКОВ-ЩЕДРИН" and len(first.split("-")) == 2
    for left, right in zip(first.split("-"), to.split("-")):
        assert {p.normal_form for p in parses(left) if "Surn" in p.tag} & {
            p.normal_form for p in parses(right) if "Surn" in p.tag and "datv" in p.tag}
    assert {p.normal_form for p in parses(family) if "plur" in p.tag} & {
        p.normal_form for p in parses(genitive) if "plur" in p.tag and "gent" in p.tag}
    assert re.fullmatch(r"[А-ЯЁ]\.", anon("И.", [["NAME", 0, 2]], seed=6))


def test_names_do_not_swap_with_originals_and_empty_input():
    text = "Иван и Пётр, затем Иван."
    found = spans(text, "NAME", "Иван|Пётр")
    for seed in range(24):
        new = anon(text, found[::-1], seed=seed)
        a, b, repeat = re.fullmatch(r"(\w+) и (\w+), затем (\w+)\.", new).groups()
        assert a == repeat and a != b and not {a, b} & {"Иван", "Пётр"}
    assert anon("", []) == ""
    assert anon("Обычный текст.\r\n", []) == "Обычный текст.\r\n"
    assert len({anon("Иван", ["NAME", 0, 4]) for _ in range(12)}) > 1
    assert mask(text, found) == "**** и ****, затем ****."


def test_all_structured_constraints_over_multiple_runs():
    card = make_card("400000123456789")
    formatted = " ".join(card[i:i+4] for i in range(0, len(card), 4))
    inn = make_inn("7743123456")
    snils = "123456789" + f"{snils_check('123456789'):02d}"
    ogr = "32377012345678"
    ogr += str(int(ogr) % 13 % 10)
    src = {"CARD": formatted, "INN": inn, "SNILS": snils, "OGRNIP": ogr, "PASSPORT": "45 10 № 654321",
           "BIRTH_CERT": "III–АБ № 123456", "EMAIL": card + "@Example.org", "PHONE": "+7 (999) 123-45-67",
           "IP": "2001:db8:1:2::3", "KLADR": "77000000000000100"}
    for label, value in src.items():
        outputs = set()
        for seed in range(20):
            new = anon(value, [label, 0, len(value)], seed=seed)
            outputs.add(new)
            assert new != value
            numbers = digits(new)
            if label == "CARD":
                assert valid_card(new) and numbers[:6] == card[:6] and numbers[-4:] == card[-4:]
                assert new[4::5] == formatted[4::5]
            if label == "INN":
                assert valid_inn(new) and new.startswith(inn[:4])
            if label == "SNILS":
                assert valid_snils(new)
            if label == "OGRNIP":
                assert valid_ogrnip(new) and new.startswith(ogr[:5])
            if label in {"PASSPORT", "BIRTH_CERT"}:
                assert new[:-6] == value[:-6] and new[-6:].isdigit()
            if label == "EMAIL":
                assert new.endswith("@Example.org")
            if label == "PHONE":
                assert numbers[:4] == digits(value)[:4] and re.sub(r"\d", "x", new) == re.sub(r"\d", "x", value)
            if label == "IP":
                assert ipaddress.ip_address(new) in ipaddress.ip_network("2001:db8:1:2::/64")
            if label == "KLADR":
                assert new in catalog()[0] and new[:2] == "77" and level(new) == "street"
        assert len(outputs) > 1


def test_card_lengths_and_formatted_duplicates():
    for n in (13, 15, 16, 19):
        card = make_card("400000" + "8" * (n - 7))
        printed = card[:4] + "-" + card[4:]
        text = card + "\n" + printed
        result = anon(text, [["CARD", 0, n], ["CARD", n+1, len(text)]], seed=n)
        a, b = result.splitlines()
        assert valid_card(a) and digits(b) == a and a[:6] == card[:6] and a[-4:] == card[-4:]
    for inn in (make_inn("770012345"), make_inn("7700123456")):
        assert valid_inn(anon(inn, ["INN", 0, len(inn)]))


def test_boundaries_validation():
    for invalid in ([["NAME", True, 4]], [["NAME", -1, 4]], [["X", 0, 4]],
                    [["NAME", 0, 4], ["SURNAME", 2, 4]], [["NAME", 1, 6]], ["NAME", "0", 4]):
        with pytest.raises(ValueError):
            anon("Иван", invalid)
    for kind, value in (("CARD", "123"), ("INN", "1234567890"), ("NAME", "123"),
                        ("PHONE", "abc9991234567"), ("BIRTH_CERT", "IIX-АБ 123456"), ("IP", "256.1.2.3")):
        with pytest.raises(ValueError):
            anon(value, [kind, 0, len(value)])


def test_competing_fields_are_resolved_for_the_selected_profile():
    inn = make_inn("770012345")
    card = make_card("400000123456789")
    ogr = "32377012345678"
    ogr += str(int(ogr) % 13 % 10)
    text = f"В паспорте указан номер {inn}. ИНН: {inn}. {card}@example.org. ОГРНИП: {ogr}. IP [2001:db8::5]:8443."
    expected = [("PASSPORT", inn), ("INN", inn), ("EMAIL", card+"@example.org"), ("OGRNIP", ogr), ("IP", "2001:db8::5")]
    result = Detector(rules_only=True, profile="course").detect(text)
    assert [(label, text[a:b]) for label, a, b in result] == expected
    wrong = ogr[:-1] + str((int(ogr[-1]) + 1) % 10)
    assert not Detector(rules_only=True, profile="course").detect(f"Подключение к 999.2.2.2, ОГРНИП: {wrong}.")
    assert all(e[0] not in {"PASSPORT", "IP", "OGRNIP"} for e in Detector(rules_only=True).detect(text))

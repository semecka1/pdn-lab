import pytest

from pdn_lab.rules import find_structured
from pdn_lab.name_context import refine_name_fields
from pdn_lab.schema import Entity


@pytest.mark.parametrize("caption", ["Карта", "Телефон", "ИНН", "СНИЛС"])
def test_complete_email_wins_over_caption_and_embedded_number(caption):
    text = caption + ": 4111111111111111@example.org."
    found = find_structured(text)
    assert [(e.label, text[e.start:e.end]) for e in found] == [("EMAIL", "4111111111111111@example.org")]


@pytest.mark.parametrize("text, expected", [
    ("'ivan@example.org'", "ivan@example.org"),
    ("E-mail=ivan@example.org ;", "ivan@example.org"),
    ("https://example.org/?email=ivan@example.org", "ivan@example.org"),
    ("https://example.org/?other=42&email=a=b@example.org", "a=b@example.org"),
    ("o'connor@example.org", "o'connor@example.org"),
    ("a=b@example.org", "a=b@example.org"),
    ('"first last"@example.org', '"first last"@example.org'),
    ("пользователь@пример.рф", "пользователь@пример.рф"),
])
def test_email_boundaries_and_valid_special_characters(text, expected):
    found = find_structured(text)
    assert [(e.label, text[e.start:e.end]) for e in found] == [("EMAIL", expected)]


def test_typographic_separators_and_adjacent_valid_numbers():
    text = "Карта: 4111\u202f1111\u202f1111\u202f1111; СНИЛС: 112–233–445 95."
    assert [e.label for e in find_structured(text)] == ["CARD", "SNILS"]
    text = "4111 1111 1111 1111 500100732259"
    assert [(e.label,text[e.start:e.end]) for e in find_structured(text)] == [("CARD","4111 1111 1111 1111"),("INN","500100732259")]


def test_explicit_order_identifier_is_not_personal_number():
    assert find_structured("Номер заказа 4111111111111111 относится к партии деталей.") == []
    assert find_structured("Идентификатор строки в тестовой базе: 500100732259.") == []


def test_name_caption_retypes_model_prediction_without_fabricating_value():
    text = "Отчество: Иванович. Имя: И."
    found = refine_name_fields(text, [Entity("NAME",10,18),Entity("SURNAME",25,26)])
    assert found == [Entity("PATRONYMIC",10,18),Entity("NAME",25,27)]
    assert refine_name_fields("Имя: отсутствует", []) == []


def test_table_header_does_not_retype_first_numeric_cell():
    text = "ИНН\tСНИЛС\r\n500100732259\t112-233-445 95"
    assert [e.label for e in find_structured(text)] == ['INN','SNILS']


def test_quantities_and_explicit_identifiers_are_not_names():
    text = '30 винтов и 101325 Па. Название переменной: Иванов.'
    # Позиции задаются по исходной строке, без токенизации предсказателем.
    candidates = [Entity(label,text.index(value),text.index(value)+len(value)) for label,value in [('SURNAME','винтов'),('NAME','Па'),('SURNAME','Иванов')]]
    assert refine_name_fields(text,candidates)==[]

"""Проверка смысла уточнений на намеренно ошибочных NER-кандидатах.

Модель и её словарь здесь не используются: правило должно исправлять именно
неверное предсказание, сохраняя человека в похожем по форме контексте.
"""
import pytest

from pdn_lab.name_context import refine_name_fields
from pdn_lab.schema import Entity


def candidate(text, label, surface, occurrence=0):
    start = -1
    for _ in range(occurrence + 1):
        start = text.index(surface, start + 1)
    return Entity(label, start, start + len(surface))


@pytest.mark.parametrize("text", [
    "Лев живёт в саванне.",
    "Лев — крупный хищник из семейства кошачьих.",
    "Лев — животное.",
    "Лев обитает в африканской саванне.",
    "Лев спит в вольере зоопарка.",
    "В саванне лев охотится на антилоп.",
])
def test_animal_subject_is_not_a_person(text):
    surface = "Лев" if "Лев" in text else "лев"
    assert refine_name_fields(text, [candidate(text, "NAME", surface)]) == []


@pytest.mark.parametrize("text, extra", [
    ("Имя: Лев. Занимается охраной саванны.", []),
    ("Лев работает в зоопарке.", []),
    ("Лев — хищник бизнеса.", []),
    ("Лев приехал в саванну на экскурсию.", []),
    ("Лев Толстой писал о природе.", [("SURNAME", "Толстой")]),
    ("Лев Иванович живёт в городе.", [("PATRONYMIC", "Иванович")]),
    ("Имя: Лев. Лев живёт в саванне.", []),
])
def test_person_lev_survives_similar_subject_matter(text, extra):
    entities = [candidate(text, "NAME", "Лев")]
    entities += [candidate(text, label, surface) for label, surface in extra]
    assert refine_name_fields(text, entities) == sorted(entities, key=lambda e: (e.start, e.end))


@pytest.mark.parametrize("text, surface", [
    ("Товарный знак «Гордеев» напечатан на упаковке.", "Гордеев"),
    ("В названии модели упомянут Петров.", "Петров"),
    ("На упаковке указан бренд «Белозеров».", "Белозеров"),
    ("Название проекта: «Голубев».", "Голубев"),
    ("Файл ПЕТРОВ-СИДОРОВ.txt сохранён.", "СИДОРОВ"),
])
def test_explicit_non_person_title_removes_erroneous_surname(text, surface):
    assert refine_name_fields(text, [candidate(text, "SURNAME", surface)]) == []


@pytest.mark.parametrize("text, surface", [
    ("Школа имени Петрова открыта утром.", "Петрова"),
    ("Школа им. Петрова открыта утром.", "Петрова"),
    ("Петров сказал: «Я согласен».", "Петров"),
    ("Редактор написал: «Петров подготовил отчёт».", "Петров"),
    ("В музее упомянули писателя Петрова.", "Петрова"),
])
def test_quotes_and_memorial_names_do_not_remove_people(text, surface):
    entities = [candidate(text, "SURNAME", surface)]
    assert refine_name_fields(text, entities) == entities


def test_title_scope_does_not_swallow_brand_owner_or_next_sentence():
    text = "Товарный знак «Петров» принадлежит Иванову. Петров сообщил об этом."
    brand = candidate(text, "SURNAME", "Петров")
    owner = candidate(text, "SURNAME", "Иванову")
    next_sentence_person = candidate(text, "SURNAME", "Петров", occurrence=1)
    assert refine_name_fields(text, [brand, owner, next_sentence_person]) == [owner, next_sentence_person]


@pytest.mark.parametrize("part", ["ПЕТРОВ", "СИДОРОВ"])
@pytest.mark.parametrize("text", [
    "ПЕТРОВ-СИДОРОВ ИВАН ИВАНОВИЧ ответил на письмо.",
    "Автор: ПЕТРОВ-СИДОРОВ.",
    "ПЕТРОВ-СИДОРОВ И. И. ответил на письмо.",
    "И. И. ПЕТРОВ-СИДОРОВ ответил на письмо.",
    "Фамилия: ПЕТРОВ-СИДОРОВ.",
])
def test_either_half_of_compound_expands_with_person_support(text, part):
    entities = [candidate(text, "SURNAME", part)]
    if "ИВАН ИВАНОВИЧ" in text:
        entities += [candidate(text, "NAME", "ИВАН"), candidate(text, "PATRONYMIC", "ИВАНОВИЧ")]
    expected = [candidate(text, "SURNAME", "ПЕТРОВ-СИДОРОВ")]
    expected += [e for e in entities if e.label != "SURNAME"]
    assert refine_name_fields(text, entities) == expected


@pytest.mark.parametrize("text, surface", [
    ("Ростов-на-Дону встретил гостей.", "Ростов"),
    ("Ростов-на-Дону встретил гостей.", "Дону"),
    ("премьер-министр Иванов выступил с докладом.", "министр"),
    ("Автор: Ростов-на-Дону.", "Ростов"),
    ("Автор: премьер-министр.", "министр"),
    ("ПРЕМЬЕР-МИНИСТР ИВАН выступил с докладом.", "МИНИСТР"),
    ("РОСТОВ-НА-ДОНУ ИВАН указал в анкете как место жительства.", "ДОНУ"),
])
def test_non_person_hyphenated_words_are_not_expanded(text, surface):
    entities = [candidate(text, "SURNAME", surface)]
    if "ИВАН" in text:
        entities.append(candidate(text, "NAME", "ИВАН"))
    assert refine_name_fields(text, entities) == entities


@pytest.mark.parametrize("part", ["ПЕТРОВ", "СИДОРОВ"])
@pytest.mark.parametrize("text", [
    "Товарный знак «ПЕТРОВ-СИДОРОВ» указан на упаковке.",
    "Бренд «Автор: ПЕТРОВ-СИДОРОВ» зарегистрирован.",
    "Название модели: «ПЕТРОВ-СИДОРОВ И. И.».",
    "Название модели: «Фамилия: ПЕТРОВ-СИДОРОВ».",
])
def test_title_semantics_remain_negative_after_compound_expansion(text, part):
    assert refine_name_fields(text, [candidate(text, "SURNAME", part)]) == []


@pytest.mark.parametrize("text", [
    "Лев Толстой написал роман.",
    "Имя: Лев; фамилия: ПЕТРОВ-СИДОРОВ.",
    "Автор: ПЕТРОВ-СИДОРОВ И. И.",
    "Товарный знак «Петров» указан на упаковке.",
])
def test_refinement_never_fabricates_people_without_model_evidence(text):
    assert refine_name_fields(text, []) == []

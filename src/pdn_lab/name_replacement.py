"""Inflection and context disambiguation of Russian personal names."""
from dataclasses import dataclass, replace
from functools import lru_cache
import re

from faker.providers.person.ru_RU import Provider
from pymorphy3 import MorphAnalyzer

WORD = re.compile(r"[^\W\d_]+")
MARK = {"NAME": "Name", "SURNAME": "Surn", "PATRONYMIC": "Patr"}
COMMON = ("Саша", "Женя", "Валя", "Шура", "Слава", "Женечка", "Сашенька", "Шурочка")
CASES = {
    "для": "gent", "без": "gent", "у": "gent", "от": "gent", "из": "gent",
    "к": "datv", "ко": "datv", "благодаря": "datv", "помог": "datv", "помогла": "datv",
    "передали": "datv", "передал": "datv", "передала": "datv", "позвонили": "datv",
    "про": "accs", "вижу": "accs", "встретил": "accs", "встретила": "accs",
    "с": "ablt", "со": "ablt", "перед": "ablt", "между": "ablt",
    "о": "loct", "об": "loct", "при": "loct",
}


@lru_cache(maxsize=1)
def analyzer():
    return MorphAnalyzer()


def parses(word):
    return analyzer().parse(word.lower())


@dataclass(frozen=True)
class Form:
    lemma: str
    gender: str | None
    number: str
    case: str
    initial: bool

    def key(self, label):
        return label, self.lemma, self.gender


def _agreement(word):
    for item in parses(word):
        if item.tag.POS in {"VERB", "ADJF", "PRTS", "ADJS"} and item.tag.gender in {"masc", "femn"}:
            return item.tag.gender
    return None


def context(text, entity, entities):
    """Use only contiguous name fields and their immediate grammatical context."""
    group = [entity]
    for other in reversed([e for e in entities if e.end <= entity.start]):
        if other.label not in MARK or text[other.end:group[0].start].strip(" \t"):
            break
        group.insert(0, other)
    for other in (e for e in entities if e.start >= entity.end):
        if other.label not in MARK or text[group[-1].end:other.start].strip(" \t"):
            break
        group.append(other)
    left = text[max(0, group[0].start - 60):group[0].start]
    right = text[group[-1].end:group[-1].end + 70]
    left_word = re.search(r"([^\W\d_]+)[ \t]+$", left)
    right_word = re.match(r"[ \t]+(?:(?:уже|вчера|сегодня|лично|сразу|не)[ \t]+){0,2}([^\W\d_]+)", right, re.I)
    case = CASES.get(left_word[1].lower()) if left_word else None
    gender = _agreement(right_word[1]) if right_word else None
    if not gender and left_word:
        gender = _agreement(left_word[1])
    for other in group:
        if other == entity:
            continue
        ps = [p for p in parses(text[other.start:other.end]) if MARK[other.label] in p.tag]
        genders = {p.tag.gender for p in ps}
        if len(genders) == 1 and genders <= {"masc", "femn"}:
            gender = genders.pop()
            break
    return gender, case


def analyze(word, label, gender_hint, case_hint):
    marker = MARK[label]
    options = parses(word)
    specific = [p for p in options if marker in p.tag]
    choices = specific or [p for p in options if p.tag.POS in {"NOUN", "ADJF"}] or options
    best = max(choices, key=lambda p: (bool(case_hint and p.tag.case == case_hint),
                                       bool(gender_hint and p.tag.gender == gender_hint), p.score))
    gender = best.tag.gender
    if gender not in {"masc", "femn"}:
        gender = gender_hint
    if best.tag.number == "plur":
        gender = None
    case = best.tag.case or case_hint or "nomn"
    return Form(best.normal_form.replace("ё", "е"), gender, best.tag.number or "sing", case, len(word) == 1)


def resolve_common(forms):
    """Associate a gender-neutral later mention with a single known context."""
    genders = {}
    for (label, _, _), form in forms.items():
        if form.gender and label == "NAME":
            genders.setdefault(form.lemma, set()).add(form.gender)
    for position, form in list(forms.items()):
        options = genders.get(form.lemma, set())
        if position[0] == "NAME" and not form.gender and form.number == "sing" and len(options) == 1:
            forms[position] = replace(form, gender=next(iter(options)))


def candidates(label, form):
    if form.initial:
        return tuple("АБВГДЕЖЗИКЛМНОПРСТУФХЭЮЯ")
    if label == "SURNAME":
        return Provider.last_names_male
    if label == "PATRONYMIC":
        return Provider.middle_names_female if form.gender == "femn" else Provider.middle_names_male
    if form.gender is None and form.number == "sing":
        return COMMON
    return Provider.first_names_female if form.gender == "femn" else Provider.first_names_male


def inflect(base, label, form):
    if form.initial:
        return base
    features = {form.case, form.number}
    if form.gender and form.number != "plur":
        features.add(form.gender)
    for p in parses(base):
        if MARK[label] in p.tag:
            word = p.inflect(features)
            if word:
                return word.word
    return None


def same_case(value, original):
    if original.isupper():
        return value.upper()
    if original.islower():
        return value.lower()
    return value.capitalize()

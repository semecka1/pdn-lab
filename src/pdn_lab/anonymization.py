"""Synonymous replacement on original character offsets, one mapping per call."""
from collections import defaultdict
import random

from . import name_replacement as names
from . import structured_replacement as fields
from .schema import ALL_ATTRIBUTES, Entity, validate_entities


def checked_spans(text, spans):
    if not isinstance(text, str) or not isinstance(spans, (list, tuple)):
        raise ValueError("Ожидаются строка и список сущностей")
    if len(spans) == 3 and isinstance(spans[0], str):
        spans = [spans]
    result = []
    for item in spans:
        if isinstance(item, Entity):
            item = item.as_list()
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError("Для каждой сущности нужны тип, начало и конец")
        label, start, end = item
        if not isinstance(label, str) or label not in ALL_ATTRIBUTES or type(start) is not int or type(end) is not int:
            raise ValueError("Неизвестный тип или нецелые границы сущности")
        result.append(Entity(label, start, end))
    validate_entities(text, result)
    return sorted(result, key=lambda e: e.start)


def _name_map(forms, rng):
    # Reserve the original lemmas and every spelling required by the document
    # before drawing anything. This also avoids mapping A to another source B.
    blocked, groups = defaultdict(set), defaultdict(set)
    for (label, _, _), form in forms.items():
        blocked[label].add(form.lemma)
        groups[form.key(label)].add(form)
    mapping = {}
    for key, required in groups.items():
        label = key[0]
        example = min(required, key=lambda f: (f.number, f.case, f.initial))
        pool = list(names.candidates(label, example))
        rng.shuffle(pool)
        for candidate in pool:
            canonical = names.analyze(candidate, label, example.gender, "nomn").lemma
            if canonical in blocked[label]:
                continue
            if all(names.inflect(candidate, label, f) for f in required):
                mapping[key] = candidate
                blocked[label].add(canonical)
                break
        else:
            raise ValueError(f"Исчерпаны совместимые замены для {label}")
    return mapping


def anon(text, spans, *, seed=None):
    """ЛР-2. Return text with the supplied non-overlapping source spans replaced."""
    entities = checked_spans(text, spans)
    rng = random.SystemRandom() if seed is None else random.Random(seed)
    forms, originals, forbidden = {}, {}, defaultdict(set)
    for e in entities:
        value = text[e.start:e.end]
        if e.label in names.MARK:
            matches = list(names.WORD.finditer(value))
            if not matches or any(not (c.isalpha() or c in " \t-‑–.’'") for c in value):
                raise ValueError(f"Некорректное значение {e.label}")
            gender, case = names.context(text, e, entities)
            for match in matches:
                pos = (e.label, e.start + match.start(), e.start + match.end())
                forms[pos] = names.analyze(match.group(), e.label, gender, case)
        else:
            key = fields.canonical(e.label, value)
            originals[e] = key
            forbidden[e.label].add(key)
    names.resolve_common(forms)
    name_map = _name_map(forms, rng)
    mapped = {}
    chunks, cursor = [], 0
    for e in entities:
        chunks.append(text[cursor:e.start])
        value = text[e.start:e.end]
        if e.label in names.MARK:
            def word(match):
                form = forms[e.label, e.start + match.start(), e.start + match.end()]
                new = names.inflect(name_map[form.key(e.label)], e.label, form)
                return names.same_case(new, match.group())
            replacement = names.WORD.sub(word, value)
        else:
            key = originals[e]
            map_key = e.label, key
            if map_key not in mapped:
                candidate = fields.generate(e.label, key, value, forbidden[e.label], rng)
                mapped[map_key] = candidate
                forbidden[e.label].add(candidate)
            replacement = fields.display(e.label, mapped[map_key], value)
        chunks.append(replacement)
        cursor = e.end
    return "".join(chunks) + text[cursor:]


def mask(text, spans):
    """Mask exactly the same source spans, one asterisk per character."""
    chars = list(text)
    for e in checked_spans(text, spans):
        chars[e.start:e.end] = "*" * (e.end - e.start)
    return "".join(chars)


def anonymize(text, detector=None, *, seed=None):
    """Coursework. Return transformed text and detections in the original text."""
    if detector is None:
        from .detector import Detector
        from .model import NameModel, default_model_directory
        detector = Detector(NameModel.load(default_model_directory()), profile="course")
    entities = detector.detect(text)
    return anon(text, entities, seed=seed), entities

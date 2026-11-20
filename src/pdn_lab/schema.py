from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

ATTRIBUTES = ("SURNAME", "NAME", "PATRONYMIC", "EMAIL", "INN", "PHONE", "CARD", "SNILS")
NAME_ATTRIBUTES = ATTRIBUTES[:3]
COURSE_ATTRIBUTES = ("SURNAME", "NAME", "INN", "PASSPORT", "CARD", "OGRNIP", "IP",
                     "EMAIL", "BIRTH_CERT", "KLADR")
ALL_ATTRIBUTES = tuple(dict.fromkeys(ATTRIBUTES + COURSE_ATTRIBUTES))
PROFILES = {"lab": ATTRIBUTES, "course": COURSE_ATTRIBUTES, "all": ALL_ATTRIBUTES}
BIO_LABELS = ("O",) + tuple(tag + "-" + label for label in ATTRIBUTES for tag in ("B", "I"))
MODEL_LABELS = ("O",) + tuple(tag + "-" + label for label in NAME_ATTRIBUTES for tag in ("B", "I"))
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Entity:
    label: str
    start: int
    end: int

    def as_list(self) -> list:
        return [self.label, self.start, self.end]


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


def tokenize(text: str) -> list[Token]:
    """Единая независимая от эталона токенизация для модели и оценки."""
    return [Token(m.group(), m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text)]


def validate_entities(text: str, entities: list[Entity]) -> None:
    previous_end = 0
    for entity in sorted(entities, key=lambda e: (e.start, e.end)):
        if entity.label not in ALL_ATTRIBUTES:
            raise ValueError(f"Неизвестный атрибут: {entity.label}")
        if not 0 <= entity.start < entity.end <= len(text):
            raise ValueError(f"Некорректные позиции: {entity}")
        if entity.start < previous_end:
            raise ValueError(f"Пересекающиеся сущности: {entity}")
        previous_end = entity.end


def to_bio(text: str, entities: list[Entity]) -> tuple[list[Token], list[str]]:
    validate_entities(text, entities)
    tokens = tokenize(text)
    labels = ["O"] * len(tokens)
    cursor = 0
    for entity in sorted(entities, key=lambda e: e.start):
        while cursor < len(tokens) and tokens[cursor].end <= entity.start:
            cursor += 1
        selected = []
        while cursor < len(tokens) and tokens[cursor].start < entity.end:
            selected.append(cursor)
            cursor += 1
        if not selected or tokens[selected[0]].start != entity.start or tokens[selected[-1]].end != entity.end:
            raise ValueError(f"Границы сущности не совпадают с токенами: {entity}, {text[entity.start:entity.end]!r}")
        for j, index in enumerate(selected):
            labels[index] = ("B-" if j == 0 else "I-") + entity.label
    return tokens, labels


def from_bio(tokens: list[Token], labels: list[str]) -> list[Entity]:
    if len(tokens) != len(labels):
        raise ValueError("Число токенов и меток различается")
    result: list[Entity] = []
    current: Entity | None = None
    for token, tag in zip(tokens, labels):
        if tag == "O":
            if current:
                result.append(current)
                current = None
            continue
        prefix, label = tag.split("-", 1)
        if prefix not in ("B", "I") or label not in ALL_ATTRIBUTES:
            raise ValueError(f"Некорректная BIO-метка: {tag}")
        if prefix == "I" and current is not None and current.label == label:
            current = Entity(label, current.start, token.end)
        else:
            if current:
                result.append(current)
            # Отдельная I после O/другого типа начинает новую сущность.
            current = Entity(label, token.start, token.end)
    if current:
        result.append(current)
    return result


def entities_of(record: dict) -> list[Entity]:
    return [Entity(*item) for item in record["entities"]]


def read_jsonl(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = set()
    for record in records:
        if record["id"] in ids:
            raise ValueError(f"Повторяющийся id: {record['id']}")
        ids.add(record["id"])
        to_bio(record["text"], entities_of(record))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

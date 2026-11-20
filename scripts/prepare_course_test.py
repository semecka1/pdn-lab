"""Build a fixed evaluation set. Gold offsets come from annotations and insertion."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import string

from faker import Faker

from pdn_lab.data import dataset_statistics, sha256
from pdn_lab.schema import COURSE_ATTRIBUTES, PROJECT_ROOT, read_jsonl, write_json, write_jsonl

SEED = 41172
# Literal address codes selected from the documented editions of the DBF files.
CODES = ("5000000200000", "5000000300000", "5000000000200", "5000000000300",
         "5000200000000", "5000300000000", "77000000000000100", "77000000000000200",
         "7700000000000020001", "7700000000000020002", "7700000000000", "7800000000000")
FIELDS = {
    "EMAIL": ("ответ направили на {EMAIL}", "контактный адрес {EMAIL}", "почта: {EMAIL}"),
    "INN": ("в графе ИНН указан {INN}", "налоговый номер {INN}", "ИНН: {INN}"),
    "CARD": ("оплата прошла по карте {CARD}", "карта: {CARD}", "в квитанции указан номер карты {CARD}"),
    "PASSPORT": ("в предъявленном паспорте указан номер {PASSPORT}", "паспорт: {PASSPORT}", "паспорт РФ № {PASSPORT}"),
    "OGRNIP": ("проверен ОГРНИП {OGRNIP}", "в выписке стоит {OGRNIP}", "ОГРНИП: {OGRNIP}"),
    "IP": ("подключение с адреса {IP}", "узел [{IP}]", "IP-адрес: {IP}"),
    "BIRTH_CERT": ("получено свидетельство о рождении {BIRTH_CERT}", "копия бланка {BIRTH_CERT}", "свидетельство о рождении № {BIRTH_CERT}"),
    "KLADR": ("территории соответствует код КЛАДР {KLADR}", "адресный код {KLADR}", "КЛАДР: {KLADR}"),
}
PREFIXES = ("В договор внесли сведения.", "Из заявления перенесли данные.", "При сверке документов выяснилось следующее.",
            "В журнале обслуживания появилась запись.", "Сотрудник подготовил выписку.", "Поступил запрос на уточнение реквизитов.")
NEGATIVES = (
    "Номер заказа {n}. Комплект прибыл на склад, отдел проверил опись.",
    "Сборка 192.0.2.999 завершилась. В журнале осталось {n} строк.",
    "Изделие «Лев» упаковано. Артикул товара {n}.",
    "Паспорт изделия проверен. Размеры коробки 12 × 30, партия {n}.",
    "Проверка отклонена: ошибочный e-mail a..b@, служебный индекс {n}.",
    "В инструкции упоминается ООО «Салтыков». Остаток на складе {n} единиц.",
)


def annotated(template, values, **metadata):
    parts, entities, cursor = [], [], 0
    for literal, field, _, _ in string.Formatter().parse(template):
        parts.append(literal)
        cursor += len(literal)
        if field:
            value = values[field]
            entities.append([field, cursor, cursor + len(value)])
            parts.append(value)
            cursor += len(value)
    return {**metadata, "text": "".join(parts), "entities": entities}


def field_values(i, rng, fake):
    def digits(n):
        return "".join(rng.choices("0123456789", k=n))
    card = fake.credit_card_number(card_type="visa")
    series, serial = digits(4), digits(6)
    passport = (series + " " + serial, series[:2] + " " + series[2:] + " " + serial, series + serial)[i % 3]
    ogr = "3" + f"{rng.randrange(4, 25):02d}" + rng.choice(("50", "77", "16", "67")) + digits(9)
    female = bool(i % 2)
    return {
        "SURNAME": fake.last_name_female() if female else fake.last_name_male(),
        "NAME": fake.first_name_female() if female else fake.first_name_male(),
        "EMAIL": (card if i % 4 == 0 else f"request.{i:03d}") + "@example.org",
        "INN": fake.businesses_inn() if i % 2 else fake.individuals_inn(),
        "CARD": card if i % 3 else " ".join(card[p:p + 4] for p in range(0, len(card), 4)),
        "PASSPORT": passport, "OGRNIP": ogr + str(int(ogr) % 13 % 10),
        "IP": (f"192.0.2.{i%253+1}" if i % 2 == 0 else f"2001:db8:{i:x}::a"),
        "BIRTH_CERT": rng.choice(("IX-АТ № ", "III–МЮ ", "IV АБ ", "II-ВГ")) + digits(6),
        "KLADR": rng.choice(CODES),
    }


def prepare(base_test, output, heldout=False):
    base = read_jsonl(base_test)
    eligible = [r for r in base if r["source"] == "factrueval" and 0 < len(r["text"]) <= 650]
    # Rank by id only; selection does not depend on NER predictions or scores.
    ranked = sorted(eligible, key=lambda r: hashlib.sha256(f"{SEED}/{r['id']}".encode()).digest())
    if len(ranked) < 120:
        raise ValueError("Требуется исходный test с корпусом factRuEval2016")
    result = [{**r, "entities": [e for e in r["entities"] if e[0] in COURSE_ATTRIBUTES]} for r in ranked[:120]]
    if heldout:
        from course_holdout import heldout_records
        previous_documents = {r["document"] for r in result}
        return save(heldout_records(eligible, previous_documents, field_values, annotated), base_test, output, 73091)
    rng, fake = random.Random(SEED), Faker("ru_RU")
    fake.seed_instance(SEED)
    for i in range(168):
        places = ["Москва", "Тула", "Казань", "Томск"]
        values = field_values(i, rng, fake)
        # Rotate fields for class coverage, shuffle prose order independently.
        labels = list(FIELDS)
        selected = [labels[(i + j) % len(labels)] for j in range(3 + i % 5)]
        rng.shuffle(selected)
        clauses = [rng.choice(FIELDS[label]) for label in selected]
        person = rng.choice(("{NAME} {SURNAME}", "{SURNAME} {NAME}"))
        opening = rng.choice(PREFIXES) + " "
        if i % 4 != 0:
            opening += "Заявитель " + person + ". "
        if i % 8 == 1:
            clauses.append("ответ получит {NAME}")
        template = opening + rng.choice(("; ", ". ", "\n")).join(clauses) + "."
        template += f" Место обработки — {rng.choice(places)}, заявка № {i + 101}."
        result.append(annotated(template, values, id=f"course/fields/{i:03d}", source="course_forms"))
    for i in range(24):
        result.append({"id": f"course/negative/{i:03d}", "source": "course_negative",
                       "text": NEGATIVES[i % len(NEGATIVES)].format(n=rng.randrange(1000, 9999)), "entities": []})
    rng.shuffle(result)
    return save(result, base_test, output, SEED)


def save(result, base_test, output, seed):
    if len({r["text"] for r in result}) != len(result):
        raise ValueError("Повторяющийся текст в тесте")
    write_jsonl(output, result)
    read_jsonl(output)
    info = {"seed": seed, "base_sha256": sha256(base_test), "test_sha256": sha256(output),
            "documents_factrueval": len({r["document"] for r in result if r["source"] == "factrueval"}),
            **dataset_statistics(result, COURSE_ATTRIBUTES)}
    write_json(output.with_suffix(".manifest.json"), info)
    return info


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-test", type=Path, default=PROJECT_ROOT / "data/processed/test.jsonl")
    p.add_argument("--output", type=Path)
    p.add_argument("--development", action="store_true", help="Воспроизвести первую диагностическую выборку")
    a = p.parse_args()
    name = "development" if a.development else "heldout"
    path = a.output or PROJECT_ROOT / f"data/course-v2/{name}.jsonl"
    print(json.dumps(prepare(a.base_test, path, heldout=not a.development), ensure_ascii=False, indent=2))

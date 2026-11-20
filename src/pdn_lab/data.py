from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import urllib.request
import zipfile

from faker import Faker

from .schema import ATTRIBUTES, Entity, entities_of, to_bio, tokenize, write_json, write_jsonl
from .validators import make_card, make_inn, snils_check, valid_snils

DATASET_REVISION = "dfe96237fce79fde13b2021d1938587e4ea3042e"
DATASET_SHA256 = "1c8f84b9831021fe2b4e4fc6e25ddbb1eb7c0972a029c09b8e5e8aab7fffc804"
BASE_MODEL = "DeepPavlov/rubert-base-cased"
MODEL_REVISION = "4036cab694767a299f2b9e6492909664d9414229"

# Семейства шаблонов разделены до подстановки значений, а не после генерации.
TEMPLATES = {
    "train": [
        "Клиент {SURNAME} {NAME} {PATRONYMIC} оставил e-mail {EMAIL} и телефон {PHONE}. ИНН: {INN}; СНИЛС: {SNILS}; карта: {CARD}.",
        "Фамилия: {SURNAME}\nИмя: {NAME}\nОтчество: {PATRONYMIC}\nТелефон: {PHONE}\nПочта: {EMAIL}\nИНН: {INN}\nСНИЛС: {SNILS}\nКарта: {CARD}",
        "Поступила заявка от {NAME} {PATRONYMIC} {SURNAME}. Для связи: {PHONE}, {EMAIL}. В анкете ИНН {INN}, СНИЛС {SNILS}, карта {CARD}.",
        "Сотрудник {SURNAME} {NAME} {PATRONYMIC} проверил ИНН {INN} и СНИЛС {SNILS}. Его контакты — {EMAIL}, {PHONE}; карта {CARD}.",
        "Владелец карты {CARD} — {NAME} {SURNAME}, отчество {PATRONYMIC}. Телефон {PHONE}; e-mail {EMAIL}; ИНН {INN}; СНИЛС {SNILS}.",
        "Запись участника: {NAME} {PATRONYMIC} {SURNAME}; почта {EMAIL}; мобильный {PHONE}; налоговый номер {INN}; страховой номер {SNILS}; карта {CARD}.",
    ],
    "validation": [
        "Уважаемый {NAME} {PATRONYMIC}! Фамилия в заявке: {SURNAME}. Подтвердите почту {EMAIL}, телефон {PHONE}, ИНН {INN}, карту {CARD} и СНИЛС {SNILS}.",
        "Договор оформлен: {SURNAME}, {NAME}, {PATRONYMIC}. СНИЛС={SNILS}; ИНН={INN}; карта={CARD}. Связь: {PHONE} / {EMAIL}.",
        "Абонент: {NAME} {SURNAME} ({PATRONYMIC}). Электронный адрес {EMAIL}. Телефон — {PHONE}. Реквизиты: ИНН {INN}; СНИЛС {SNILS}; карта {CARD}.",
    ],
    "test": [
        "Письмо подписал {NAME} {PATRONYMIC} {SURNAME}: «Ответ направьте на {EMAIL}, позвоните {PHONE}». В приложении ИНН {INN}, СНИЛС {SNILS} и карта {CARD}.",
        "На приёме посетитель {SURNAME} {NAME} {PATRONYMIC} продиктовал СНИЛС {SNILS} и ИНН {INN}. Для оплаты указана карта {CARD}, для уведомлений — {PHONE} и {EMAIL}.",
        "Из переписки: «Я {NAME}, моя фамилия {SURNAME}, отчество {PATRONYMIC}. Мой телефон {PHONE}; пишите на {EMAIL}. ИНН {INN}, СНИЛС {SNILS}. Карта {CARD}».",
        "Контакт в архиве — {NAME} {SURNAME}. Отчество — {PATRONYMIC}; e-mail — {EMAIL}; телефон — {PHONE}. СНИЛС: {SNILS}. Карта: {CARD}. ИНН: {INN}.",
        "Для {NAME} {PATRONYMIC} {SURNAME} подготовлено уведомление. Его доставят на {EMAIL}. Телефон {PHONE}. Сверьте карту {CARD}, ИНН {INN}, СНИЛС {SNILS}.",
    ],
}
PARTIAL = {
    "train": ["{NAME} {SURNAME} написал письмо на {EMAIL}.", "У {NAME} {PATRONYMIC} новый телефон {PHONE}.", "Проверьте ИНН {INN} и СНИЛС {SNILS}.", "Карта {CARD} указана рядом с почтой {EMAIL}."],
    "validation": ["Получатель — {SURNAME}, {NAME}; телефон {PHONE}.", "Для сверки прислали СНИЛС {SNILS}, ИНН {INN} и карту {CARD}.", "{NAME} {PATRONYMIC} просит ответить: {EMAIL}."],
    "test": ["В ответе значится {SURNAME} {NAME}, обратный адрес {EMAIL}.", "СНИЛС {SNILS} принадлежит {NAME} {PATRONYMIC} {SURNAME}.", "Платёжные данные: карта {CARD}; налоговый номер {INN}. Связаться можно по телефону {PHONE}.", "Позовите {NAME} {PATRONYMIC}; сообщение пришло с {EMAIL}."],
}
FIELD_NAMES = {"NAME": "Имя", "SURNAME": "Фамилия", "PATRONYMIC": "Отчество", "EMAIL": "E-mail", "PHONE": "Телефон", "INN": "ИНН", "SNILS": "СНИЛС", "CARD": "Карта"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch_factrueval(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "factrueval.zip"
    if not archive.exists():
        url = f"https://codeload.github.com/dialogue-evaluation/factRuEval-2016/zip/{DATASET_REVISION}"
        request = urllib.request.Request(url, headers={"User-Agent": "pdn-lab1/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            archive.with_suffix(".download").write_bytes(response.read())
        archive.with_suffix(".download").replace(archive)
    if sha256(archive) != DATASET_SHA256:
        raise ValueError("SHA-256 датасета не совпал. Сохранённый архив не используется.")
    write_json(raw_dir / "source.json", {"repository": "https://github.com/dialogue-evaluation/factRuEval-2016", "revision": DATASET_REVISION, "archive_sha256": DATASET_SHA256})
    return archive


def _values(fake: Faker, rng: random.Random) -> dict[str, str]:
    gender = rng.choice(("male", "female"))
    values = {
        "NAME": getattr(fake, f"first_name_{gender}")(),
        "SURNAME": getattr(fake, f"last_name_{gender}")(),
        "PATRONYMIC": getattr(fake, f"middle_name_{gender}")(),
    }
    # Модель видит также нижний регистр; исходные позиции не изменяются.
    if rng.random() < 0.1:
        values = {key: value.lower() for key, value in values.items()}
    elif rng.random() < 0.1:
        values = {key: value.upper() for key, value in values.items()}
    values["EMAIL"] = rng.choice(("user", "student", "contact", "name.surname", "mail_box", "info+lab")) + str(rng.randrange(10000, 999999)) + "@" + rng.choice(("example.org", "example.net", "example.com", "mail.example.org"))
    inn_base = str(rng.randrange(1000000000, 9999999999))
    values["INN"] = make_inn(inn_base)
    snils_base = str(rng.randrange(1002000, 999999999)).zfill(9)
    snils = snils_base + f"{snils_check(snils_base):02d}"
    values["SNILS"] = rng.choice((snils, f"{snils[:3]}-{snils[3:6]}-{snils[6:9]} {snils[9:]}", f"{snils[:3]} {snils[3:6]} {snils[6:9]} {snils[9:]}"))
    card = make_card("4" + "".join(str(rng.randrange(10)) for _ in range(14)))
    separator = rng.choice(("", " ", "-"))
    values["CARD"] = separator.join(card[i:i+4] for i in range(0, len(card), 4))
    while True:
        phone = "79" + "".join(str(rng.randrange(10)) for _ in range(9))
        if not valid_snils(phone):
            break
    values["PHONE"] = rng.choice(("+" + phone, phone, "8" + phone[1:], f"+7 ({phone[1:4]}) {phone[4:7]}-{phone[7:9]}-{phone[9:]}", f"8 {phone[1:4]} {phone[4:7]} {phone[7:9]} {phone[9:]}"))
    return values


def fill_template(template: str, values: dict[str, str]) -> tuple[str, list[list]]:
    parts, entities, cursor, position = [], [], 0, 0
    for match in re.finditer(r"\{([A-Z]+)\}", template):
        prefix = template[cursor:match.start()]
        parts.append(prefix)
        position += len(prefix)
        label = match.group(1)
        value = values[label]
        parts.append(value)
        entities.append([label, position, position + len(value)])
        position += len(value)
        cursor = match.end()
    parts.append(template[cursor:])
    return "".join(parts), entities


def generate_synthetic(split: str, count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    fake = Faker("ru_RU")
    fake.seed_instance(seed)
    records = []
    for index in range(count):
        values = _values(fake, rng)
        bucket = index % 10
        if bucket < 6:
            template_index = rng.randrange(len(TEMPLATES[split]))
            template = TEMPLATES[split][template_index]
            family = f"{split}/full/{template_index}"
        elif bucket < 8:
            template_index = rng.randrange(len(PARTIAL[split]))
            template = PARTIAL[split][template_index]
            family = f"{split}/partial/{template_index}"
        elif bucket == 8:
            label = ATTRIBUTES[(index // 10) % len(ATTRIBUTES)]
            prefix = {"train": "В анкете осталось одно поле", "validation": "Проверяется значение", "test": "Из карточки выписано"}[split]
            template = prefix + ": " + FIELD_NAMES[label] + " {" + label + "}."
            family = f"{split}/single/{label}"
        else:
            # Контрольная цифра изменена намеренно, эталон не содержит ПДн.
            bad_inn = values["INN"][:-1] + str((int(values["INN"][-1]) + 1) % 10)
            snils = re.sub(r"\D", "", values["SNILS"])
            bad_snils = snils[:-1] + str((int(snils[-1]) + 1) % 10)
            card = re.sub(r"\D", "", values["CARD"])
            bad_card = card[:-1] + str((int(card[-1]) + 1) % 10)
            prefix = {"train": "Невалидные реквизиты", "validation": "Ошибки при вводе", "test": "Записи с повреждённой контрольной цифрой"}[split]
            template = f"{prefix}: ИНН {bad_inn}; СНИЛС {bad_snils}; карта {bad_card}. Код заявки {rng.randrange(10000,99999)}. Адрес записан как user(at)example.org."
            family = f"{split}/negative"
        text, entities = fill_template(template, values)
        record = {"id": f"synthetic/{split}/{index:05d}", "source": "synthetic", "template_family": family, "text": text, "entities": entities}
        to_bio(text, entities_of(record))
        records.append(record)
    return records


def _chunk_document(text: str, entities: list[Entity], max_tokens: int = 110):
    tokens = tokenize(text)
    index = 0
    while index < len(tokens):
        last = min(index + max_tokens, len(tokens))
        # Сущность не должна пересекать границу фрагментов.
        if last < len(tokens):
            while last > index and any(e.start < tokens[last].start < e.end for e in entities):
                last -= 1
        if last == index:
            raise ValueError("Сущность длиннее размера фрагмента")
        start, end = tokens[index].start, tokens[last - 1].end
        selected = [Entity(e.label, e.start - start, e.end - start) for e in entities if start <= e.start and e.end <= end]
        yield text[start:end], selected, start
        index = last


def import_factrueval(archive: Path, seed: int) -> tuple[dict[str, list[dict]], list[dict]]:
    splits = {key: [] for key in ("train", "validation", "test")}
    exclusions = []
    with zipfile.ZipFile(archive) as files:
        root = files.namelist()[0].split("/")[0]
        dev = sorted(name for name in files.namelist() if "/devset/" in name and re.fullmatch(r"book_[0-9]+\.txt", Path(name).name))
        random.Random(seed).shuffle(dev)
        validation_docs = set(dev[:max(1, round(len(dev) * 0.2))])
        documents = sorted(name for name in files.namelist() if ("/devset/" in name or "/testset/" in name) and re.fullmatch(r"book_[0-9]+\.txt", Path(name).name))
        for name in documents:
            split = "test" if "/testset/" in name else ("validation" if name in validation_docs else "train")
            span_name = name[:-4] + ".spans"
            if span_name not in files.namelist():
                raise ValueError(f"Нет разметки: {span_name}")
            text = files.read(name).decode("utf-8-sig")
            mapping = {"name": "NAME", "surname": "SURNAME", "patronymic": "PATRONYMIC"}
            entities = set()
            for line in files.read(span_name).decode("utf-8-sig").splitlines():
                fields = line.split()
                if len(fields) >= 4 and fields[1] in mapping:
                    start, length = int(fields[2]), int(fields[3])
                    entities.add(Entity(mapping[fields[1]], start, start + length))
            entities = sorted(entities, key=lambda e: (e.start, e.end))
            try:
                to_bio(text, entities)
            except ValueError as error:
                exclusions.append({"document": name, "reason": str(error)})
                continue
            for chunk_index, (fragment, selected, start) in enumerate(_chunk_document(text, entities)):
                # В factRuEval нет полной эталонной разметки контактов/номеров.
                # Консервативно исключаем такие фрагменты, не размечая их своим же детектором.
                if re.search(r"@|[0-9][0-9 ()+\-]{8,}[0-9]", fragment):
                    exclusions.append({"document": name, "chunk": chunk_index, "reason": "Возможные неразмеченные структурированные ПДн"})
                    continue
                record = {"id": f"factrueval/{name.split('/')[-2]}/{Path(name).stem}/{chunk_index}", "source": "factrueval", "document": name[len(root)+1:], "document_start": start, "text": fragment, "entities": [e.as_list() for e in selected]}
                splits[split].append(record)
        # Лицензия и описание сохраняются рядом с подготовленными данными.
        for filename in ("LICENSE", "README.md"):
            source = f"{root}/{filename}"
            if source in files.namelist():
                (archive.parent / ("factrueval_" + filename)).write_bytes(files.read(source))
    return splits, exclusions


def dataset_statistics(records: list[dict], attributes=ATTRIBUTES) -> dict:
    texts, entities = Counter(), Counter()
    for record in records:
        entities.update(entity[0] for entity in record["entities"])
        texts.update({entity[0] for entity in record["entities"]})
    return {"texts": len(records), "texts_per_attribute": {a: texts[a] for a in attributes}, "entities_per_attribute": {a: entities[a] for a in attributes}, "sources": dict(Counter(r["source"] for r in records)), "negative_texts": sum(not r["entities"] for r in records)}


def prepare(data_dir: Path, seed: int = 20261022, train_count: int = 1800, validation_count: int = 320, test_count: int = 480, synthetic_only: bool = False) -> dict:
    processed = data_dir / "processed"
    if (processed / "manifest.json").exists():
        raise FileExistsError("Данные уже подготовлены. Используйте другую --data-dir, чтобы сохранить исходный итоговый тест.")
    if synthetic_only:
        corpus = {key: [] for key in TEMPLATES}
        exclusions = []
    else:
        archive = fetch_factrueval(data_dir / "raw")
        corpus, exclusions = import_factrueval(archive, seed)
    splits = {}
    seen_texts = set()
    removed_duplicates = Counter()
    # Приоритет train, затем validation, затем test: пересечения текста удаляются.
    for index, (split, count) in enumerate((("train", train_count), ("validation", validation_count), ("test", test_count))):
        candidates = corpus[split] + generate_synthetic(split, count, seed + index * 1009)
        records = []
        for record in candidates:
            digest = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
            if digest in seen_texts:
                removed_duplicates[split] += 1
                continue
            seen_texts.add(digest)
            records.append(record)
        random.Random(seed + index).shuffle(records)
        splits[split] = records
    statistics = {key: dataset_statistics(value) for key, value in splits.items()}
    if any(value <= 50 for value in statistics["test"]["texts_per_attribute"].values()):
        raise ValueError("В итоговом тесте требуется больше 50 текстов на каждый из 8 атрибутов. Увеличьте --test-count.")
    for split, records in splits.items():
        write_jsonl(processed / f"{split}.jsonl", records)
    manifest = {"schema_version": 1, "seed": seed, "dataset_revision": None if synthetic_only else DATASET_REVISION, "dataset_archive_sha256": None if synthetic_only else DATASET_SHA256, "base_model": BASE_MODEL, "model_revision": MODEL_REVISION, "synthetic_only": synthetic_only, "statistics": statistics, "removed_duplicate_texts": dict(removed_duplicates), "excluded_corpus_fragments": exclusions, "files": {split: sha256(processed / f"{split}.jsonl") for split in splits}, "limitations": ["Синтетический тест проверяет контролируемые сценарии и не доказывает качество на произвольных текстах.", "factRuEval содержит публичные тексты и исходные имена, а не синтетические ПДн.", "Структурированные атрибуты не добавляются в эталон factRuEval результатами детектора.", "Разделение factRuEval выполнено по документам; отдельные публичные персоны могут встречаться в разных документах."]}
    write_json(processed / "manifest.json", manifest)
    return manifest

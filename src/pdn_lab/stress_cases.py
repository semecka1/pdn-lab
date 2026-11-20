"""Независимые размеченные сценарии стресс-теста, без вызова детектора."""
from __future__ import annotations

import itertools
import math
import random
import re

from .schema import ATTRIBUTES, Entity, to_bio

SEED = 20261022
MARKER = re.compile(r"\[\[([A-Z]+)\|(.*?)\]\]", re.S)


def mark(label: str, value: str) -> str:
    return f"[[{label}|{value}]]"


def case(category: str, template: str, scope: str = "core", note: str = "") -> dict:
    pieces, entities, cursor, position = [], [], 0, 0
    for match in MARKER.finditer(template):
        prefix = template[cursor:match.start()]
        pieces.append(prefix)
        position += len(prefix)
        label, value = match.groups()
        assert label in ATTRIBUTES
        entities.append(Entity(label, position, position + len(value)))
        pieces.append(value)
        position += len(value)
        cursor = match.end()
    pieces.append(template[cursor:])
    text = "".join(pieces)
    to_bio(text, entities)
    return {"category": category, "scope": scope, "source": "stress", "note": note, "text": text, "entities": [e.as_list() for e in entities]}


VALUES = {
    "SURNAME": "Иванов", "NAME": "Иван", "PATRONYMIC": "Иванович",
    "EMAIL": "4111111111111111@example.org", "PHONE": "+7 (912) 345-67-89",
    "INN": "500100732259", "CARD": "4111 1111 1111 1111", "SNILS": "112-233-445 95",
}
CAPTIONS = {"SURNAME": "Фамилия", "NAME": "Имя", "PATRONYMIC": "Отчество", "EMAIL": "Почта", "PHONE": "Телефон", "INN": "ИНН", "CARD": "Карта", "SNILS": "СНИЛС"}
NOISE = [
    "Заказ № 42, дата 22.10.2026, сумма 1234.56 руб.",
    "Версия 1.2.3; IP стенда 192.0.2.1; порт 8080.",
    "Склад Б-17: 12 коробок, 30 винтов, 4 шайбы.",
    "UUID: 550e8400-e29b-41d4-a716-446655440000.",
    "Температура 21.5 °C, давление 101325 Па.",
    "План выполнен на 85%; размер 1920×1080.",
    "В отчёте перечислены товары и общие сведения о процессе.",
    "У меня есть надежда на мир, вера в успех и любовь к природе.",
]


def _card_with_checksum(base: str) -> str:
    # Независимый оракул: сумма через цифровой корень удвоенных цифр.
    # Не импортирует make_card/valid_card из программы.
    contributions = []
    for index, char in enumerate(reversed(base)):
        digit = int(char)
        contributions.append(sum(map(int, str(digit * 2))) if index % 2 == 0 else digit)
    return base + str((-sum(contributions)) % 10)


def generate_cases(max_permutations: int = math.factorial(8), seed: int = SEED) -> list[dict]:
    if not 0 <= max_permutations <= math.factorial(8):
        raise ValueError("Число перестановок должно быть от 0 до 40320")
    rng = random.Random(seed)
    records = []

    def add(category, text, scope="core", note=""):
        records.append(case(category, text, scope, note))

    fields = [CAPTIONS[label] + ": " + mark(label, VALUES[label]) for label in ATTRIBUTES]
    for order in itertools.islice(itertools.permutations(fields), max_permutations):
        add("all_8_permutations", "; ".join(order) + ".")

    # Поля без подписей, вмешанные числа, разные разделители, все порядки ФИО.
    separators = ["; ", ", ", " | ", "\n", "\t", " — ", " / ", ". "]
    profiles = [
        ("Иванов", "Иван", "Иванович"),
        ("Петрова", "Анна", "Сергеевна"),
        ("Сидоров", "Пётр", "Алексеевич"),
        ("Смирнова", "Надежда", "Викторовна"),
        ("Кузнецов", "Илья", "Олегович"),
        ("Орлова", "Любовь", "Ильинична"),
    ]
    for profile in profiles:
        names = [mark(label, value) for label, value in zip(ATTRIBUTES[:3], profile)]
        for order in itertools.permutations(names):
            for repetition in range(12):
                items = list(order) + [mark(label, VALUES[label]) for label in ATTRIBUTES[3:]]
                rng.shuffle(items)
                for _ in range(repetition % 4):
                    items.insert(rng.randrange(len(items) + 1), rng.choice(NOISE))
                add("unlabelled_interleaving", rng.choice(separators).join(items))

    # Полные e-mail с локальной частью, похожей на различные номера.
    number_prefixes = ["4111111111111111", "5555555555554444", "378282246310005", "500100732259", "7707083893", "79123456789", "11223344595"]
    email_patterns = ["{}@example.org", "{}.user@example.org", "{}+test@example.org", "{}_user@example.org", "{}-test@example.org", "a{}@example.org", "user@{}.example.org"]
    for prefix, pattern in itertools.product(number_prefixes, email_patterns):
        address = pattern.format(prefix)
        for surrounding in ["Почта: {}.", "Напишите на <{}>.", 'Адрес: "{}".', "Связь: ({}).", "{}; карта " + mark("CARD", VALUES["CARD"]), "До: " + rng.choice(NOISE) + "\n{}\nПосле: запись закончена."]:
            add("email_numeric_prefix", surrounding.format(mark("EMAIL", address)))

    # Неверная подпись не должна превращать часть валидного e-mail в номер.
    for caption, prefix in itertools.product(["Карта", "Телефон", "ИНН", "СНИЛС", "Налоговый номер", "Страховой номер"], number_prefixes):
        for suffix in ["", ".user", "+test", "_user", "-backup"]:
            add("email_with_misleading_caption", f"{caption}: " + mark("EMAIL", prefix + suffix + "@example.org") + ".", note="Самостоятельная сущность — полный e-mail; вложенные цифровые совпадения не размечаются.")

    addresses = ["ivan@example.org", "4111111111111111@example.org", "79123456789@example.net", "anna+test@mail.example.org", "user_name@example.com"]
    for address in addresses:
        for context in ["'{}'", '"{}"', "«{}»", "({})", "[{}]", "<{}>", "{}.", "{},", "{}!", "{}?", "E-mail={} ;", "mailto:{}", "https://example.org/?email={}"]:
            add("email_surrounding_punctuation", context.format(mark("EMAIL", address)), note="Внешние кавычки и URL-параметр не входят в e-mail.")

    # Числа на границах форматов, корректные и повреждённые контрольные цифры.
    cards = ["4111111111111111", "5555555555554444", "378282246310005", "4222222222222", "30569309025904", "6011111111111117"]
    for length in (17, 18, 19):
        cards.append(_card_with_checksum("4" + "".join(str(rng.randrange(10)) for _ in range(length - 2))))
    for number in cards:
        for separator in ["", " ", "-"]:
            formatted = separator.join(number[i:i+4] for i in range(0, len(number), 4))
            for prefix in ["Карта: ", "Оплата — ", "", "Данные клиента: "]:
                add("card_lengths_and_formats", prefix + mark("CARD", formatted) + ".")
        for last in "0123456789":
            if last != number[-1]:
                add("invalid_checksums", "Карта: " + number[:-1] + last + ".")
    for label, numbers in [("INN", ["500100732259", "7707083893", "7736207543"]), ("SNILS", ["11223344595", "99999999901", "90000005900", "90000006800"])]:
        for number in numbers:
            add("valid_checksum_boundaries", CAPTIONS[label] + ": " + mark(label, number) + ".")
            for last in "0123456789":
                if last != number[-1]:
                    add("invalid_checksums", CAPTIONS[label] + ": " + number[:-1] + last + ".")
    for number in ["0000000000", "000000000000", "111111111111", "0000000000000000", "00000000000", "1111111111111111"]:
        add("degenerate_numbers", "Пустые реквизиты: " + number + ".")

    phone_formats = ["+7 (912) 345-67-89", "8 (912) 345-67-89", "+79123456789", "79123456789", "89123456789", "8 912 345 67 89", "+7-912-345-67-89", "8-495-123-45-67", "+7 (495)1234567"]
    for phone in phone_formats:
        for template in ["Телефон: {}.", "Позвонить {}!", 'Тел.: "{}"', "({}); доб. 123", "Контакт [{}] — отвечают вечером."]:
            add("phone_formats", template.format(mark("PHONE", phone)))

    # Отдельно размеченные неподдержанные типографские варианты.
    for separator in ["\u00a0", "\u202f", "\t", "\n", "–", "‑"]:
        add("unicode_number_separators", "Карта: " + mark("CARD", separator.join(["4111", "1111", "1111", "1111"])), "extended", "Типографские/многострочные разделители шире исходных ASCII-форматов.")
        add("unicode_number_separators", "СНИЛС: " + mark("SNILS", separator.join(["112", "233", "445", "95"])), "extended")
        add("unicode_number_separators", "Телефон: " + mark("PHONE", separator.join(["+7", "912", "345", "67", "89"])), "extended")
    for phone in ["+1 202 555 0123", "+44 20 7946 0958", "9123456789", "4951234567"]:
        add("foreign_and_short_phones", "Телефон: " + mark("PHONE", phone), "extended", "Программа заявляет только российские 11-значные номера.")
    for email in ["пользователь@пример.рф", "user@пример.рф", '"first last"@example.org']:
        add("extended_email_syntax", "Почта: " + mark("EMAIL", email), "extended", "Unicode/IDN и quoted local-part не заявлены как поддерживаемые.")

    # Чистый шум, похожие на ПДн последовательности и явный контекст не-ПДн.
    for text in ["", " ", "\n\t\r\n", "🙂🚀✅", *NOISE,
                 "Имена файлов: report.txt, data.csv, README.md; контрольная сумма abcdef0123456789.",
                 "Лев живёт в саванне. вера в победу и надежда на успех помогают команде.",
                 "почта: user(at)example.org; example.org; ivan@; @example.org; ivan @ example.org; a..b@localhost",
                 "Даты: 01.01.2000, 22/10/2026; время 12:34:56; дроби 1/2 и 3.14159."]:
        add("negative_noise", text)
    for number in ["4111111111111111", "500100732259", "79123456789", "11223344595"]:
        for template in ["Артикул товара: {}.", "Номер заказа {} относится к партии деталей.", "Идентификатор строки в тестовой базе: {}."]:
            add("non_pdn_context_valid_number", template.format(number), "semantic", "Номер явно назван заказом/артикулом/идентификатором; желаемый результат — O. По одному виду числа это семантически неоднозначно.")
        for template in ["abc{}def", "id_{}_tmp", "v{}beta", "sha256:abc{}def"]:
            add("numbers_inside_identifiers", template.format(number))
    for text in ["ООО «Иванов» поставляет детали.", "Название переменной: Иванов.", "Файл Иванов.txt находится в каталоге.", "В названии модели упомянут Петров."]:
        add("names_in_non_person_context", text, "semantic", "Именоподобный фрагмент используется как название/идентификатор, а не упоминание человека.")

    # Свободная речь, падежи, омонимы, регистр, дефисы, инициалы.
    prose = [
        "Заявление подал [[SURNAME|Иванов]] [[NAME|Иван]] [[PATRONYMIC|Иванович]].",
        "Документ получен от [[SURNAME|Иванова]] [[NAME|Ивана]] [[PATRONYMIC|Ивановича]].",
        "Письмо передали [[NAME|Ивану]] [[PATRONYMIC|Ивановичу]] [[SURNAME|Иванову]].",
        "Мы поговорили с [[NAME|Анной]] [[PATRONYMIC|Сергеевной]] [[SURNAME|Петровой]].",
        "В отчёте говорится об [[NAME|Илье]] [[PATRONYMIC|Олеговиче]] [[SURNAME|Кузнецове]].",
        "[[NAME|Надежда]] [[SURNAME|Орлова]] сказала, что надежда на успех остаётся.",
        "[[NAME|Любовь]] [[SURNAME|Петрова]] описывает любовь к природе.",
        "[[NAME|Лев]] [[SURNAME|Иванов]] увидел, как лев подошёл к воде.",
        "[[NAME|Вера]] [[PATRONYMIC|Ильинична]] [[SURNAME|Смирнова]] — вера в успех ей помогает.",
        "Отвечает [[SURNAME|САЛТЫКОВ-ЩЕДРИН]] [[NAME|МИХАИЛ]] [[PATRONYMIC|ЕВГРАФОВИЧ]].",
        "Запись: [[SURNAME|Салтыков-Щедрин]] [[NAME|Михаил]] [[PATRONYMIC|Евграфович]].",
        "Поступило сообщение от [[NAME|Анны-Марии]] [[SURNAME|Ивановой]].",
        "Имя: [[NAME|И.]]; фамилия: [[SURNAME|Иванов]]; отчество: [[PATRONYMIC|И.]].",
    ]
    for index, text in enumerate(prose):
        for noise in ["", NOISE[0] + " ", "🙂\r\n", NOISE[7] + "\n"]:
            add("names_in_prose", noise + text, note="Инициалы размечены вместе с точкой." if index == len(prose)-1 else "")
    for surname, name, patronymic in profiles:
        for transform in [str.lower, str.upper, str.title]:
            for order in itertools.permutations([(label, transform(value)) for label, value in zip(ATTRIBUTES[:3], (surname, name, patronymic))]):
                add("name_case_and_order", "; ".join(CAPTIONS[label] + ": " + mark(label, value) for label, value in order))

    # Форматы сообщений; разметка задаётся заранее, а не результатом правил.
    for repetition in range(32):
        order = list(ATTRIBUTES)
        rng.shuffle(order)
        selected = [(label, VALUES[label]) for label in order]
        add("json_csv_xml", "{" + ", ".join('"' + label + '": "' + mark(label, value) + '"' for label, value in selected) + "}")
        add("json_csv_xml", "\t".join(CAPTIONS[label] for label, _ in selected) + "\r\n" + "\t".join(mark(label, value) for label, value in selected))
        add("json_csv_xml", "<record>" + "".join("<" + label.lower() + ">" + mark(label, value) + "</" + label.lower() + ">" for label, value in selected) + "</record>")

    for _ in range(80):
        people = rng.sample(profiles, 3)
        persons = []
        for surname, name, patronymic in people:
            persons.append(" ".join([mark("SURNAME", surname), mark("NAME", name), mark("PATRONYMIC", patronymic)]))
        add("multiple_people", "Сотрудники: " + "; ".join(persons) + ". Общая почта " + mark("EMAIL", rng.choice(addresses)) + ". " + rng.choice(NOISE))

    for left, right in itertools.permutations(["CARD", "PHONE", "INN", "SNILS"], 2):
        for separator in ["; ", ", ", " | ", "\n", " "]:
            add("adjacent_numbers", mark(left, VALUES[left]) + separator + mark(right, VALUES[right]), "semantic" if separator == " " else "core", "Без подписей и с одним пробелом граница цифровых сущностей может быть неоднозначной." if separator == " " else "")
    for separator in [" ", "\n", "\r\n", "\t", "\u00a0", "\u202f", "🙂"]:
        add("unicode_offsets_between_fields", separator.join(fields))
    for repeat in [10, 50, 150, 500]:
        filler = "В этом разделе приведены общие сведения о порядке работы системы. "
        add("long_text", filler * repeat + "\n" + "; ".join(fields))
        add("long_text", "; ".join(fields) + "\n" + filler * repeat + "\n" + "; ".join(reversed(fields)))

    # Десятки различных числовых префиксов e-mail и подписи в случайном порядке.
    for index in range(256):
        card = _card_with_checksum("4" + "".join(str(rng.randrange(10)) for _ in range(14)))
        email = card + rng.choice(["", ".user", "+backup", "_office"]) + "@" + rng.choice(["example.org", "example.net", "mail.example.org"])
        items = ["Почта " + mark("EMAIL", email), "Карта " + mark("CARD", card), "Телефон " + mark("PHONE", VALUES["PHONE"]), rng.choice(NOISE)]
        rng.shuffle(items)
        add("random_card_email_collisions", rng.choice(separators).join(items))

    for index, record in enumerate(records):
        record["id"] = f"stress/{record['category']}/{index:06d}"
    return records

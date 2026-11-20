"""Independent coursework templates and news documents, fixed before evaluation."""
import hashlib
import random

from faker import Faker

from pdn_lab.schema import COURSE_ATTRIBUTES


def heldout_records(eligible, excluded_documents, field_values, annotated):
    seed = 73091
    rng, fake = random.Random(seed), Faker("ru_RU")
    fake.seed_instance(seed)
    news = sorted((r for r in eligible if r["document"] not in excluded_documents),
                  key=lambda r: hashlib.sha256(f"held/{seed}/{r['id']}".encode()).digest())
    if len(news) < 84:
        raise ValueError("Недостаточно отдельных новостных документов для итогового теста")
    rows = [{**r, "entities": [e for e in r["entities"] if e[0] in COURSE_ATTRIBUTES]} for r in news[:84]]
    fields = {
        "PASSPORT": ("предъявлен паспорт гражданина РФ № {PASSPORT}", "в паспорте записан номер {PASSPORT}", "паспортные данные: {PASSPORT}"),
        "EMAIL": ("письмо вернулось с адреса {EMAIL}", "пишите на {EMAIL}", "адрес для копии — {EMAIL}"),
        "IP": ("соединение с {IP} завершилось", "источник пакета — [{IP}]", "клиент подключился с узла {IP}"),
        "CARD": ("списание с карты {CARD}", "для возврата использована карта № {CARD}", "в чеке напечатано {CARD}"),
        "KLADR": ("заполните поле КЛАДР: {KLADR}", "шифр территории по КЛАДР — {KLADR}", "объект найден под кодом {KLADR}"),
        "INN": ("сведения по ИНН {INN} сверены", "ИНН заявителя — {INN}", "из реестра перенесён ИНН: {INN}"),
        "OGRNIP": ("регистрационная запись ИП {OGRNIP}", "по ОГРНИП {OGRNIP} найдено обращение", "предприниматель сообщил ОГРНИП № {OGRNIP}"),
        "BIRTH_CERT": ("вложена копия свидетельства {BIRTH_CERT}", "запись о рождении сверили с бланком {BIRTH_CERT}", "реквизиты метрики — {BIRTH_CERT}"),
    }
    codes = ("2200000101400", "22000001000000600", "2200001300000", "2200000100000650003",
             "3602100100000", "3600000100000150002", "3601200000000", "5200000000000",
             "52000001000001200", "5200000100000190001", "78000000000001100", "7800000300000")
    for i in range(144):
        values = field_values(i + 211, rng, fake)
        values["KLADR"] = rng.choice(codes)
        if i % 4:
            values["EMAIL"] = f"letter-{i}.archive@service.example.org"
        if i % 2:
            values["IP"] = f"2001:db8:{i:x}:12::f"
        keys = list(fields)
        subset = [keys[(i+j) % len(keys)] for j in range(3 + (i//3) % 5)]
        rng.shuffle(subset)
        clauses = [rng.choice(fields[key]) for key in subset]
        opening = rng.choice(("Ответственный специалист {NAME} {SURNAME} внёс исправления.",
                              "Приём ведёт {SURNAME} {NAME}.", "Запрос поступил в отдел доставки.",
                              "Копию заявления получил сотрудник {NAME} {SURNAME}."))
        template = opening + "\n" + rng.choice(("; ", ". ", "\n")).join(clauses) + "."
        rows.append(annotated(template, values, id=f"held/forms/{i:03d}", source="course_forms"))
    families = (("Римский-Корсаков", "Римскому-Корсакову"), ("Петрова-Соколова", "Петровой-Соколовой"),
                ("Мамин-Сибиряк", "Мамину-Сибиряку"), ("Смирнова-Волкова", "Смирновой-Волковой"))
    for i in range(32):
        family, dative = families[i % len(families)]
        values = {"NAME": ("Саша", "Женя", "Валя", "Ира")[i % 4], "SURNAME": family}
        if i % 3 == 0:
            template = ("Сотрудница {SURNAME} передала бумаги. Подпись поставила {SURNAME}." if i % 2
                        else "Сотрудник {SURNAME} передал бумаги. Подпись поставил {SURNAME}.")
        elif i % 3 == 1:
            template = "{NAME} отправила письмо. Материалы доставили к {SURNAME}."
            values["SURNAME"] = dative
        else:
            template = "В журнале есть фамилия: {SURNAME}. {NAME} лично проверил опись."
            values["NAME"] = ("Саша", "Женя", "Валя", "Слава")[i % 4]
        template += f" Выпуск ведомости {i+19}."
        rows.append(annotated(template, values, id=f"held/context/{i:03d}", source="course_context"))
    negatives = (
        "Технический паспорт изделия выдан на партию {n}. Формат упаковки не изменился.",
        "Узел 260.1.2.3 не прошёл проверку. Код операции {n}.",
        "Имя файла «Иванов.txt» перенесли в архив, строка {n}.",
        "Оформлена серия книг «Петров». Тираж {n} экземпляров.",
        "В форме пропущен домен: строка editor@. Ошибка номер {n}.",
    )
    for i in range(20):
        rows.append({"id": f"held/negative/{i:03d}", "source": "course_negative",
                     "text": negatives[i % 5].format(n=i + 2471), "entities": []})
    rng.shuffle(rows)
    return rows

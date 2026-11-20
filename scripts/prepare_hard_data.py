"""Новые сложные train/validation и заранее фиксируемый контрольный набор."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil

from faker import Faker

from pdn_lab.data import _values, dataset_statistics, sha256
from pdn_lab.schema import ATTRIBUTES, PROJECT_ROOT, entities_of, read_jsonl, to_bio, write_json, write_jsonl

if __package__:
    from .legacy_stress_cases import legacy_cases
else:
    from legacy_stress_cases import legacy_cases


class Text:
    def __init__(self):
        self.parts = []
        self.entities = []
        self.length = 0

    def add(self, value, label=None):
        if label:
            self.entities.append([label, self.length, self.length + len(value)])
        self.parts.append(value)
        self.length += len(value)

    def record(self, split, index, family):
        record = {"id": f"hard-v2/{split}/{index:05d}", "source": "hard_synthetic", "category": family, "scope": "core", "note": "Новый экземпляр, не скопированный из исследованного стресс-теста.", "text": "".join(self.parts), "entities": self.entities}
        to_bio(record["text"], entities_of(record))
        return record


NOISE = {
    "train": ["За смену изготовлено 73 детали.", "Сумма 725.40 руб.; дата 14.09.2025.", "Контроль процесса завершён, версия 3.7.1.", "На столе лежат пять тетрадей и линейка."],
    "validation": ["За месяц выпущено 94 изделия.", "В отчёте сумма 816.25 руб. и дата 03.02.2024.", "Система обновлена до версии 4.8.2."],
    "holdout": ["В коробке находилось 57 креплений.", "Поставка на 913.70 руб. принята 19.07.2023.", "Номер сборки 5.9.3 указан в журнале."],
}
PREFIX = {"train": "Из регистрационной записи: ", "validation": "Проверяем поступившие сведения: ", "holdout": "При разборе архивного сообщения обнаружено: "}
CAPTIONS = {"SURNAME": "Фамилия", "NAME": "Имя", "PATRONYMIC": "Отчество", "EMAIL": "Почта", "PHONE": "Телефон", "INN": "ИНН", "CARD": "Карта", "SNILS": "СНИЛС"}


def sample(split, index, rng, fake):
    values = _values(fake, rng)
    mode = index % 14
    text = Text()
    separator = {"train": "; ", "validation": " | ", "holdout": " / "}[split]
    order = list(ATTRIBUTES)
    rng.shuffle(order)
    if mode in (0, 1, 2, 3):
        family = ("fields", "json", "xml", "table")[mode]
        keys = {a: a for a in ATTRIBUTES}
        if split == "validation":
            keys.update(SURNAME="last_name", NAME="first_name", PATRONYMIC="middle_name")
        if split == "holdout":
            keys.update(SURNAME="surname", NAME="name", PATRONYMIC="patronymic")
        if mode == 1: text.add("{")
        if mode == 2: text.add("<person>")
        if mode == 3:
            text.add("\t".join(CAPTIONS[a] if split != "validation" else keys[a] for a in order) + "\r\n")
        for i, label in enumerate(order):
            if mode == 0:
                text.add(CAPTIONS[label] + (": " if split == "train" else " = "))
            elif mode == 1: text.add('"' + keys[label] + '": "')
            elif mode == 2: text.add("<" + keys[label] + ">")
            text.add(values[label], label)
            if mode == 0 and i < len(order)-1: text.add(separator)
            elif mode == 1: text.add('"' + (", " if i < len(order)-1 else ""))
            elif mode == 2: text.add("</" + keys[label] + ">")
            elif mode == 3 and i < len(order)-1: text.add("\t")
        if mode == 1: text.add("}")
        if mode == 2: text.add("</person>")
    elif mode in (4, 5, 6):
        family = ("unlabelled_mixed", "unlabelled_names", "isolated_patronymic")[mode-4]
        text.add(PREFIX[split])
        labels = order if mode == 4 else ([a for a in order if a in ATTRIBUTES[:3]] if mode == 5 else ["PATRONYMIC"])
        for i, label in enumerate(labels):
            text.add(values[label], label)
            if i < len(labels)-1:
                text.add(rng.choice([separator, ", ", "\n", " — ", ". "]))
                if rng.random() < .25:
                    text.add(rng.choice(NOISE[split]) + " ")
        text.add(". " + rng.choice(NOISE[split]))
    elif mode == 7:
        family = "declined_prose"
        profiles = {
            "train": [("Соколова", "Андрея", "Петровича"), ("Морозовой", "Еленой", "Дмитриевной"), ("Власову", "Денису", "Вадимовичу")],
            "validation": [("Зайцева", "Алексея", "Константиновича"), ("Беляевой", "Ольгой", "Михайловной")],
            "holdout": [("Гусева", "Николая", "Станиславовича"), ("Крыловой", "Татьяной", "Анатольевной")],
        }
        names = rng.choice(profiles[split])
        text.add(PREFIX[split] + rng.choice(["упоминается обращение ", "приведены данные ", "указаны сведения о "]))
        for i, (label,value) in enumerate(zip(ATTRIBUTES[:3], names)):
            if i: text.add(" ")
            text.add(value,label)
        text.add(". " + rng.choice(NOISE[split]))
    elif mode == 8:
        family = "compound_names"
        text.add(PREFIX[split])
        values["SURNAME"] = fake.last_name_male() + "-" + fake.last_name_male()
        values["NAME"] = rng.choice(["Анна-Мария", "Жан-Поль", "Анна-Луиза", "Мария-Елена"])
        for i,label in enumerate(rng.sample(list(ATTRIBUTES[:3]),3)):
            if i: text.add("; ")
            text.add(CAPTIONS[label] + ": ")
            text.add(values[label],label)
        text.add(". " + rng.choice(NOISE[split]))
    elif mode == 9:
        family = "negative_context"
        patterns = {
            "train": ["Лев лежал на тёплом камне в зоопарке.", "У человека есть вера в успех и надежда на лучшее.", "ООО «{surname}» выпустило каталог товаров.", "Открыть файл {surname}.txt в редакторе.", "Переменная {surname} хранит количество строк."],
            "validation": ["Лев охотится на антилопу в африканской саванне.", "В статье обсуждаются любовь к искусству и вера в добро.", "Каталог компании «{surname}» состоит из пяти страниц.", "Имя файла: {surname}.txt; размер 20 КБ."],
            "holdout": ["Лев — крупный хищник из семейства кошачьих.", "Надежда на результат и любовь к труду поддерживают коллектив.", "Товарный знак «{surname}» напечатан на упаковке.", "В справочнике записан идентификатор {surname}, а не имя человека."],
        }
        text.add(rng.choice(patterns[split]).format(surname=fake.last_name_male()) + " " + rng.choice(NOISE[split]))
    elif mode == 10:
        family = "initials"
        for i,label in enumerate(rng.sample(list(ATTRIBUTES[:3]),3)):
            if i: text.add(separator)
            text.add(CAPTIONS[label] + ": ")
            value = values[label] if label == "SURNAME" else values[label][0].upper() + "."
            text.add(value,label)
    elif mode == 11:
        family = "email_conflicts"
        prefix = "".join(char for char in values["CARD"] if char.isdigit())
        text.add(rng.choice(["Карта", "Телефон", "ИНН", "СНИЛС"]) + ": ")
        text.add(prefix + rng.choice(["", "+lab", ".office"]) + "@example.net", "EMAIL")
        text.add("; " + rng.choice(NOISE[split]) + " Контактное лицо ")
        text.add(values["NAME"],"NAME")
        text.add(" ")
        text.add(values["PATRONYMIC"],"PATRONYMIC")
    elif mode == 12:
        family = "several_people"
        text.add(PREFIX[split])
        for person in range(2):
            if person: text.add("; второй участник — ")
            current = _values(fake,rng)
            for i,label in enumerate(ATTRIBUTES[:3]):
                if i: text.add(" ")
                text.add(current[label],label)
        text.add(". " + rng.choice(NOISE[split]))
    else:
        family = "positive_homonyms"
        text.add(PREFIX[split])
        text.add(rng.choice(["Вера", "Надежда", "Любовь", "Лев"]),"NAME")
        text.add(" ")
        text.add(fake.last_name(),"SURNAME")
        text.add(" — участник обсуждения. " + rng.choice(NOISE[split]))
    return text.record(split,index,family)


def prepare(data_dir: Path, output_dir: Path) -> dict:
    target=output_dir/'processed'
    if target.exists():raise FileExistsError('Новый набор уже подготовлен')
    original=json.loads((data_dir/'processed/manifest.json').read_text(encoding='utf-8'))
    original_splits={split:read_jsonl(data_dir/'processed'/f'{split}.jsonl') for split in ('train','validation','test')}
    fingerprint=lambda text:hashlib.sha256(text.encode('utf-8')).hexdigest()
    blocked={fingerprint(r['text']) for records in original_splits.values() for r in records}
    blocked.update(fingerprint(record['text']) for record in legacy_cases())
    generated={}
    for offset,(split,count) in enumerate([('train',4500),('validation',900),('holdout',1000)]):
        rng=random.Random(2026102300+offset)
        fake=Faker('ru_RU');fake.seed_instance(2026102300+offset)
        records=[];index=0
        while len(records)<count:
            record=sample(split,index,rng,fake);index+=1
            digest=fingerprint(record['text'])
            if digest not in blocked:
                blocked.add(digest);records.append(record)
        generated[split]=records
    target.mkdir(parents=True)
    combined={
        'train':original_splits['train']+generated['train'],
        'validation':original_splits['validation']+generated['validation'],
        'test':original_splits['test'],
    }
    for split,records in combined.items():
        random.Random(20261023).shuffle(records) if split!='test' else None
        if split=='test':shutil.copyfile(data_dir/'processed/test.jsonl',target/'test.jsonl')
        else:write_jsonl(target/f'{split}.jsonl',records)
    write_jsonl(target/'holdout.jsonl',generated['holdout'])
    manifest={**original,'augmentation_version':2,'parent_manifest_sha256':sha256(data_dir/'processed/manifest.json'),'seed':2026102300,'statistics':{k:dataset_statistics(v) for k,v in combined.items()},'augmentation_counts':{k:len(v) for k,v in generated.items()},'files':{split:sha256(target/f'{split}.jsonl') for split in combined},'holdout_sha256':sha256(target/'holdout.jsonl'),'leakage_check':'Нет полных совпадений новых текстов между train/validation/holdout, исходным тестом и старым стресс-набором. Типовые схемы форматов могут повторяться.'}
    write_json(target/'manifest.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=PROJECT_ROOT/'data',help='Корень исходных данных с processed/manifest.json')
    parser.add_argument('--output-dir',type=Path,default=PROJECT_ROOT/'data/hard-v2',help='Корень новых данных; внутри создаётся processed/')
    args=parser.parse_args()
    manifest=prepare(args.data_dir,args.output_dir)
    print(json.dumps({'statistics':manifest['statistics'],'augmentation':manifest['augmentation_counts'],'holdout_sha256':manifest['holdout_sha256']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

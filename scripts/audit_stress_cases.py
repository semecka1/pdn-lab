"""Исправляет ошибку эталона v1, сохраняя тексты и добавляя валидные карты."""
from __future__ import annotations

import argparse
from pathlib import Path

from pdn_lab.schema import PROJECT_ROOT, write_jsonl, write_json
from pdn_lab.stress_cases import case, mark

if __package__:
    from .legacy_stress_cases import legacy_cases
else:
    from legacy_stress_cases import legacy_cases


def luhn(number):
    digits=[int(char) for char in number if char.isascii() and char.isdigit()]
    return sum(sum(map(int,str(digit*(2 if index%2 else 1)))) for index,digit in enumerate(reversed(digits)))%10==0


def prepare(target: Path, legacy_path: Path | None = None) -> dict:
    if target.exists():
        raise FileExistsError(target)
    records=legacy_cases(legacy_path)
    changes=[]
    for record in records:
        if record['category']=='unicode_number_separators' and record['entities'][0][0]=='CARD':
            label,start,end=record['entities'][0]
            if luhn(record['text'][start:end]):
                raise ValueError('Ожидалась историческая карта с ошибочной контрольной суммой')
            changes.append({'id':record['id'],'text':record['text'],'old_entities':record['entities'],'new_entities':[],'reason':'4111 4111 4111 4111 не проходит алгоритм Луна; исходный эталон ошибочно требовал CARD.'})
            record['entities']=[]
            record['category']='unicode_invalid_card_checksum'
            record['scope']='core'
            record['note']='Отрицательный пример: исправлена ошибочная положительная разметка карты в v1.'
    if len(changes)!=6:
        raise ValueError(f'Ожидались шесть исправлений эталона, получено {len(changes)}')
    for index,separator in enumerate(['\u00a0','\u202f','\t','\n','–','‑']):
        value=separator.join(['4111','1111','1111','1111'])
        assert luhn(value)
        record=case('unicode_number_separators','Карта: '+mark('CARD',value),'extended','Дополнительная корректная карта с Unicode-разделителем.')
        record['id']=f'stress/valid_unicode_card_added/{index}'
        records.append(record)

    target.mkdir(parents=True)
    write_jsonl(target/'cases.jsonl',records)
    audit={'corrected_annotations':changes,'added_valid_cases':6,'texts':len(records),'reason':'Корректность эталона проверена отдельно от распознавателя. Исходные протоколы v1 сохранены.'}
    write_json(target/'audit.json',audit)
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=PROJECT_ROOT/'data/stress-audited-v2')
    parser.add_argument('--legacy-cases',type=Path,help='Необязательный исходный cases.jsonl v1; по умолчанию восстанавливается из исходников')
    args=parser.parse_args()
    audit=prepare(args.output_dir,args.legacy_cases)
    print('Audited cases:',audit['texts'],'corrected annotations:',len(audit['corrected_annotations']),'added valid cards:',audit['added_valid_cases'])


if __name__=='__main__':
    main()

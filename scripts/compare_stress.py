from __future__ import annotations

import json
from pathlib import Path

from pdn_lab.schema import PROJECT_ROOT, write_json


def comparison(before, after):
    old=json.loads((before/'report.json').read_text(encoding='utf-8'))
    new=json.loads((after/'report.json').read_text(encoding='utf-8'))
    assert old['cases_sha256']==new['cases_sha256'],'До и после использованы разные тексты'
    assert old['overall']['texts']==new['overall']['texts']
    categories={}
    for name,old_category in old['categories'].items():
        fresh=new['categories'][name]
        assert old_category['texts']==fresh['texts']
        categories[name]={'texts':fresh['texts'],'passed_before':old_category['exact_passed'],'passed_after':fresh['exact_passed'],'errors_before':old_category['exact_failed'],'errors_after':fresh['exact_failed']}
    corrected=[];regressed=[];still_bad=[]
    for raw,old_line,new_line in zip((before/'cases.jsonl').open(encoding='utf-8'),(before/'predictions.jsonl').open(encoding='utf-8'),(after/'predictions.jsonl').open(encoding='utf-8'),strict=True):
        record,past,fresh=map(json.loads,(raw,old_line,new_line))
        assert record['id']==past['id']==fresh['id']
        gold=set(map(tuple,record['entities']))
        old_ok=gold==set(map(tuple,past['entities'])) and not past['error']
        new_ok=gold==set(map(tuple,fresh['entities'])) and not fresh['error']
        entry={'id':record['id'],'text':record['text'],'expected':record['entities'],'before':past['entities'],'after':fresh['entities']}
        if not old_ok and new_ok:corrected.append(entry)
        if old_ok and not new_ok:regressed.append(entry)
        if not old_ok and not new_ok:still_bad.append(entry)
    result={'before_model_sha256':old['model_sha256'],'after_model_sha256':new['model_sha256'],'cases_sha256':new['cases_sha256'],'before':old['overall'],'after':new['overall'],'slices_before':old['slices'],'slices_after':new['slices'],'categories':categories,'corrected':len(corrected),'regressed':len(regressed),'still_bad':len(still_bad),'corrected_examples':corrected[:12],'regressed_examples':regressed[:12],'remaining_examples':still_bad[:20]}
    return result


def main():
    root=PROJECT_ROOT
    known=comparison(root/'reports/stress-audited-v2-before',root/'reports/stress-audited-v2-after')
    fresh=comparison(root/'reports/holdout-v2-before',root/'reports/holdout-v2-final')
    intermediate=json.loads((root/'reports/holdout-v2-final-new-rules-old-model/report.json').read_text(encoding='utf-8'))
    assert intermediate['model_sha256']==fresh['before_model_sha256']
    assert intermediate['cases_sha256']==fresh['cases_sha256']
    original_before=json.loads((root/'reports/test.json').read_text(encoding='utf-8'))
    original_after=json.loads((root/'reports/test-v2-final.json').read_text(encoding='utf-8'))
    assert original_before['evaluation_sha256']==original_after['evaluation_sha256']
    target=root/'reports/comparison-v2'
    if target.exists():raise FileExistsError('Сравнение уже существует')
    target.mkdir(parents=True)
    audit=json.loads((root/'data/stress-audited-v2/audit.json').read_text(encoding='utf-8'))
    write_json(target/'comparison.json',{'known_regression_suite':known,'fresh_holdout':fresh,'new_rules_old_model_holdout':intermediate['overall'],'original_test_before':original_before['scores'],'original_test_after':original_after['scores'],'stress_annotation_audit':audit})
    lines=['# Сравнение первой и исправленной версии','', 'Одинаковые входные тексты и эталоны до и после подтверждены SHA-256. Известный стресс-набор рассматривается как регрессионный; отдельные 1000 текстов отложены до обучения и не использовались для выбора эпохи. Типовые форматы этих новых экземпляров могут совпадать с обучением.','', '| Набор | Текстов | Полностью верно до | Полностью верно после | Ошибок до | Ошибок после |','|---|---:|---:|---:|---:|---:|']
    for title,result in [('Известный стресс-набор',known),('Отложенные новые тексты',fresh)]:
        old,new=result['before'],result['after']
        lines.append(f"| {title} | {old['texts']} | {old['exact_passed']} ({old['exact_pass_rate']:.2%}) | {new['exact_passed']} ({new['exact_pass_rate']:.2%}) | {old['exact_failed']} | {new['exact_failed']} |")
    old=known['slices_before']['without_exhaustive_permutations']
    new=known['slices_after']['without_exhaustive_permutations']
    lines.append(f"| Стресс-набор без 40320 перестановок | {old['texts']} | {old['exact_passed']} ({old['exact_pass_rate']:.2%}) | {new['exact_passed']} ({new['exact_pass_rate']:.2%}) | {old['exact_failed']} | {new['exact_failed']} |")
    lines.extend(['','## Исправление эталона стресс-теста','','В исходном наборе шесть положительных примеров карты содержали `4111 4111 4111 4111`, которая не проходит алгоритм Луна. Их исходные тексты сохранены, разметка исправлена на отрицательную; добавлены шесть корректных карт `4111 1111 1111 1111` с такими же разделителями. Поэтому итоговый общий набор содержит 42377 текстов. Обе версии заново запущены на одном уточнённом эталоне. Первоначальные протоколы из 42371 текста не изменены. Подробный аудит: `data/stress-audited-v2/audit.json`.','','## Отдельный вклад дообучения','','| Версия на новых 1000 текстах | Полностью верно | Строгая weighted F1 |','|---|---:|---:|'])
    for title,item in [('Исходные правила и модель',fresh['before']),('Новые правила со старой моделью',intermediate['overall']),('Новые правила и дообученная модель',fresh['after'])]:
        lines.append(f"| {title} | {item['exact_passed']} / {item['texts']} | {item['scores']['strict_entity_weighted_f1']:.4f} |")
    lines.extend(['','## Исходный лабораторный тест','','| Метрика | До | После |','|---|---:|---:|'])
    for title,key in [('Weighted BIO F1 без O','token_weighted_f1_without_o'),('Строгая weighted F1 сущностей','strict_entity_weighted_f1')]:
        lines.append(f"| {title} | {original_before['scores'][key]:.4f} | {original_after['scores'][key]:.4f} |")
    lines.extend(['','Исходный тест сохранён побайтно. Небольшое снижение строгой F1 на нём означает, что улучшение сложных синтетических сценариев не гарантирует улучшения каждого показателя естественного корпуса.','', '## E-mail и отчество на новых контрольных текстах','','| Атрибут | F1 до | F1 после | FP до → после | FN до → после |','|---|---:|---:|---:|---:|'])
    for label in ('EMAIL','PATRONYMIC'):
        old=fresh['before']['scores']['strict_entities_by_attribute'][label]
        new=fresh['after']['scores']['strict_entities_by_attribute'][label]
        lines.append(f"| {label} | {old['f1']:.4f} | {new['f1']:.4f} | {old['fp']} → {new['fp']} | {old['fn']} → {new['fn']} |")
    lines.extend(['','## Категории исходного стресс-набора','','| Категория | Всего | Верно до | Верно после |','|---|---:|---:|---:|'])
    for name,item in known['categories'].items():lines.append(f"| {name} | {item['texts']} | {item['passed_before']} | {item['passed_after']} |")
    for title,result in [('Известный стресс-набор',known),('Отложенные тексты',fresh)]:
        old,new=result['before']['scores'],result['after']['scores']
        lines.extend(['',f'## Метрики {title}','',f"Weighted BIO F1 без O: {old['token_weighted_f1_without_o']:.6f} → {new['token_weighted_f1_without_o']:.6f}.",f"Строгая weighted F1 по сущностям: {old['strict_entity_weighted_f1']:.6f} → {new['strict_entity_weighted_f1']:.6f}.",f"Исправлено текстов: {result['corrected']}; новых неудачных текстов: {result['regressed']}; остались неудачными: {result['still_bad']}.",'','### Оставшиеся ошибки',''])
        for example in result['remaining_examples'][:5]+result['regressed_examples'][:3]:
            short={**example,'text':example['text'][:1500]}
            lines.extend(['```json',json.dumps(short,ensure_ascii=False,indent=2),'```',''])
    (target/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:{'before':v['before']['exact_passed'],'after':v['after']['exact_passed'],'total':v['after']['texts'],'corrected':v['corrected'],'regressed':v['regressed'],'still_bad':v['still_bad']} for k,v in [('known',known),('fresh',fresh)]},indent=2))


if __name__=='__main__':main()

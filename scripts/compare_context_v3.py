"""Сравнение v2/v3 и проверка неизменности исходного лабораторного теста."""
from datetime import datetime, timezone
import argparse
import json

from compare_stress import comparison
from pdn_lab.data import sha256
from pdn_lab.schema import PROJECT_ROOT, write_json


def read(relative):
    return json.loads((PROJECT_ROOT / relative).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Проверить артефакты без записи отчётов')
    parser.add_argument('--output-dir', default='reports/comparison-v3')
    args = parser.parse_args()
    root = PROJECT_ROOT
    suites = {
        'stress': ('reports/stress-audited-v2-after', 'reports/stress-v3-final'),
        'control_1000': ('reports/holdout-v2-final', 'reports/holdout-v3-final'),
        'context_176': ('reports/context-v3-before', 'reports/context-v3-final'),
    }
    results = {name: comparison(root / before, root / after) for name, (before, after) in suites.items()}
    snapshot = read('runs/baseline-v2/manifest.json')
    before = read('reports/test-v3-before.json')
    after = read('reports/test-v3-final.json')
    initial = read('reports/test.json')
    data = read('data/hard-v2/processed/manifest.json')
    assert sha256(root / 'data/hard-v2/processed/test.jsonl') == before['evaluation_sha256'] == after['evaluation_sha256'] == initial['evaluation_sha256']
    assert sha256(root / 'data/processed/test.jsonl') == after['evaluation_sha256']
    for split, digest in data['files'].items():
        assert sha256(root / 'data/hard-v2/processed' / f'{split}.jsonl') == digest
    for relative, digest in snapshot['program_sha256'].items():
        assert sha256(root / 'runs/baseline-v2' / relative) == digest
    model_hash = sha256(root / 'models/rubert-pdn-v2/model.safetensors')
    assert model_hash == snapshot['model_sha256']
    for name, (old_path, new_path) in suites.items():
        old, new = read(old_path + '/report.json'), read(new_path + '/report.json')
        assert old['model_sha256'] == new['model_sha256'] == model_hash
        assert old['program_sha256'] == snapshot['program_sha256']
        for relative, digest in new['program_sha256'].items():
            assert sha256(root / relative) == digest
        assert not old['inference_errors'] and not new['inference_errors']
        assert not new['overall']['bio_alignment_errors']
        assert results[name]['regressed'] == 0, f'Новый неудачный текст в {name}'
    assert results['context_176']['cases_sha256'] == read('data/context-v3/manifest.json')['cases_sha256'] == sha256(root / 'data/context-v3/cases.jsonl')
    identical = read('reports/test-v3-before_predictions.json') == read('reports/test-v3-final_predictions.json')
    assert identical, 'Изменились ответы на исходном тесте'
    keys = ('token_weighted_f1_without_o', 'strict_entity_weighted_f1')
    assert all(after['scores'][key] >= before['scores'][key] for key in keys)
    verification = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'all_checks_passed': True,
        'model_weights_unchanged_sha256': model_hash,
        'original_test_unchanged_sha256': after['evaluation_sha256'],
        'original_test_all_1095_predictions_identical': identical,
        'both_original_test_f1_not_decreased_vs_v2': True,
        'original_test_strict_f1_still_below_v1': after['scores']['strict_entity_weighted_f1'] < initial['scores']['strict_entity_weighted_f1'],
        'no_new_fully_incorrect_texts_in_three_suites': True,
        'checked_current_source_sha256': read('reports/stress-v3-final/report.json')['program_sha256'],
    }
    if args.verify_only:
        print(json.dumps(verification, ensure_ascii=False, indent=2))
        return
    target = root / args.output_dir
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)
    write_json(target / 'verification.json', verification)
    write_json(target / 'comparison.json', {
        'suites': results, 'original_test_v1': initial['scores'],
        'original_test_v2': before['scores'], 'original_test_v3': after['scores'],
        'verification': verification,
    })
    titles = {'stress': 'Полный стресс-набор', 'control_1000': 'Прежние 1000 контрольных текстов', 'context_176': 'Новый ручной контекстный набор'}
    lines = ['# Исправление контекста ФИО: v2 → v3', '',
        'Веса дообученной модели v2 сохранены. Изменён только модуль уточнения ФИО. Эталон и тексты каждого набора совпадают до/после; это проверено по SHA-256.', '',
        '| Проверка | Текстов | Полностью верно до | Полностью верно после | Ошибок до → после |',
        '|---|---:|---:|---:|---:|']
    for name, result in results.items():
        old, new = result['before'], result['after']
        lines.append(f"| {titles[name]} | {old['texts']} | {old['exact_passed']} ({old['exact_pass_rate']:.2%}) | {new['exact_passed']} ({new['exact_pass_rate']:.2%}) | {old['exact_failed']} → {new['exact_failed']} |")
    lines.extend(['', 'Ни в одном из трёх наборов нет ранее полностью правильного текста, ставшего неправильным. Набор из 176 текстов проверяет ручную разметку контекста и защитные контрпримеры; он изучался при разработке правил и считается регрессионным, а не независимым подтверждением обобщения.', '',
        '## Исходный лабораторный тест', '',
        '**Все 1095 списков предсказанных сущностей совпадают с v2**, включая типы и позиции. Сам test сохранён побайтно.', '',
        '| Метрика | v1 | До исправлений: v2 | После: v3 |', '|---|---:|---:|---:|'])
    for title, key in [('Weighted BIO F1 без O', keys[0]), ('Строгая weighted F1 сущностей', keys[1])]:
        lines.append(f"| {title} | {initial['scores'][key]:.9f} | {before['scores'][key]:.9f} | {after['scores'][key]:.9f} |")
    lines.extend(['', 'Относительно текущей v2 качество не ухудшилось. Небольшое снижение строгой F1, возникшее при переходе v1 → v2, остаётся; v3 его не увеличивает и не устраняет.', '', '## Исправленные исходные четыре сценария', ''])
    for example in results['stress']['corrected_examples']:
        lines.extend(['```json', json.dumps(example, ensure_ascii=False, indent=2), '```', ''])
    for name in ('control_1000', 'context_176'):
        result = results[name]
        lines.extend([f"## Оставшиеся ошибки: {titles[name]}", '', f"Осталось {result['after']['exact_failed']} текстов с ошибками. Полные предсказания и примеры находятся в `{suites[name][1]}`.", ''])
        for example in result['remaining_examples'][:8]:
            lines.extend(['```json', json.dumps(example, ensure_ascii=False, indent=2), '```', ''])
    (target / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({name: {'before': result['before']['exact_passed'], 'after': result['after']['exact_passed'], 'total': result['after']['texts'], 'corrected': result['corrected'], 'regressed': result['regressed']} for name, result in results.items()}, ensure_ascii=False, indent=2))
    print('Original-test non-regression and source/model/data hashes verified.')


if __name__ == '__main__':
    main()

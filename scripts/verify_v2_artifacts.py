"""Проверка происхождения артефактов v2 после обучения и сравнения."""
import json

from pdn_lab.data import sha256
from pdn_lab.schema import PROJECT_ROOT, read_jsonl, write_json


def read(path):
    return json.loads((PROJECT_ROOT / path).read_text(encoding='utf-8'))


def main():
    root = PROJECT_ROOT
    baseline = read('runs/baseline-v1/manifest.json')
    original = read('data/processed/manifest.json')
    data = read('data/hard-v2/processed/manifest.json')
    trained = read('models/rubert-pdn-v2/training_manifest.json')
    before = read('reports/stress-audited-v2-before/report.json')
    after = read('reports/stress-audited-v2-after/report.json')
    comparison = read('reports/comparison-v2/comparison.json')
    assert sha256(root / 'models/rubert-pdn/model.safetensors') == baseline['model_sha256'] == before['model_sha256']
    assert before['program_sha256'] == baseline['program_sha256']
    for relative, digest in baseline['program_sha256'].items():
        assert sha256(root / 'runs/baseline-v1' / relative) == digest
    # После v3 прежняя программа доступна в сохранённом снимке v2.
    source_root = root / 'runs/baseline-v2' if (root / 'runs/baseline-v2/manifest.json').exists() else root
    for relative, digest in after['program_sha256'].items():
        assert sha256(source_root / relative) == digest
    assert sha256(root / 'reports/stress-20261022-full/cases.jsonl') == baseline['stress_cases_sha256']
    assert sha256(root / 'models/rubert-pdn-v2/model.safetensors') == after['model_sha256'] != baseline['model_sha256']
    assert data['parent_manifest_sha256'] == sha256(root / 'data/processed/manifest.json')
    for split, digest in original['files'].items():
        assert sha256(root / 'data/processed' / f'{split}.jsonl') == digest
    for split, digest in data['files'].items():
        assert sha256(root / 'data/hard-v2/processed' / f'{split}.jsonl') == digest
    assert data['files']['test'] == original['files']['test'] == read('reports/test-v2-final.json')['evaluation_sha256']
    assert trained['training_completed']
    assert all(trained[f'{split}_sha256'] == data['files'][split] for split in ('train', 'validation'))
    assert sha256(root / 'data/hard-v2/processed/holdout.jsonl') == data['holdout_sha256']
    splits = {split: read_jsonl(root / 'data/hard-v2/processed' / f'{split}.jsonl') for split in ('train', 'validation', 'test', 'holdout')}
    texts = {split: {record['text'] for record in records} for split, records in splits.items()}
    for split in ('train', 'validation', 'test'):
        assert not texts['holdout'] & texts[split]
    assert not texts['train'] & texts['validation']
    old_stress = {record['text'] for record in read_jsonl(root / 'reports/stress-20261022-full/cases.jsonl')}
    for split in ('train', 'validation', 'holdout'):
        additions = {record['text'] for record in splits[split] if record['id'].startswith('hard-v2/')}
        assert not additions & old_stress
    cases_hash = sha256(root / 'data/stress-audited-v2/cases.jsonl')
    assert before['cases_sha256'] == after['cases_sha256'] == cases_hash
    assert not before['inference_errors'] and not after['inference_errors']
    assert not before['overall']['bio_alignment_errors'] and not after['overall']['bio_alignment_errors']
    result = {
        'checks_passed': True,
        'original_model_unchanged': True,
        'original_data_unchanged': True,
        'test_byte_identical': True,
        'no_holdout_text_duplicates_in_training_validation_test': True,
        'no_new_texts_copied_from_old_stress': True,
        'same_stress_cases_and_gold_before_after': True,
        'final_program_hashes_match_tested_program': True,
        'verified_program_source_root': str(source_root.relative_to(root)),
        'texts_per_split': {split: len(records) for split, records in splits.items()},
        'original_model_sha256': baseline['model_sha256'],
        'finetuned_model_sha256': after['model_sha256'],
        'stress_cases_sha256': cases_hash,
        'holdout_sha256': data['holdout_sha256'],
        'holdout_corrected': comparison['fresh_holdout']['corrected'],
        'holdout_regressed': comparison['fresh_holdout']['regressed'],
    }
    write_json(root / 'reports/comparison-v2/verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

"""python scripts/stress_test.py --help"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from pdn_lab.data import sha256
from pdn_lab.detector import Detector
from pdn_lab.metrics import evaluate_predictions
from pdn_lab.model import NameModel
from pdn_lab.schema import ATTRIBUTES, PROJECT_ROOT, entities_of, to_bio, write_json, write_jsonl, read_jsonl
from pdn_lab.stress_cases import generate_cases


def describe(text, entities):
    return [{"label": e.label, "start": e.start, "end": e.end, "text": text[e.start:e.end]} for e in entities]


def summarize(records, predictions, errors):
    alignable = []
    invalid = []
    for i, (record, predicted) in enumerate(zip(records, predictions)):
        try:
            to_bio(record["text"], predicted)
            alignable.append(i)
        except ValueError as error:
            invalid.append({"id": record["id"], "error": str(error)})
    scores = evaluate_predictions([records[i] for i in alignable], [predictions[i] for i in alignable]) if alignable else None
    exact = sum(set(entities_of(record)) == set(predicted) and record["id"] not in errors for record, predicted in zip(records, predictions))
    return {"texts": len(records), "exact_passed": exact, "exact_failed": len(records)-exact, "exact_pass_rate": exact/len(records), "inference_errors": sum(r["id"] in errors for r in records), "bio_alignment_errors": invalid, "token_metrics_texts": len(alignable), "scores": scores}


def write_report(report, examples, directory):
    overall = report["overall"]
    permutations = report["categories"].get("all_8_permutations", {}).get("texts", 0)
    note = f"В наборе {permutations} перестановок одних базовых значений; их вес не должен скрывать другие категории." if permutations else "Это отдельный набор новых экземпляров; типовые схемы форматов могут повторяться в обучении."
    lines = ["# Стресс-тест распознавания ПДн", "", f"Проверено текстов: **{overall['texts']}**. Полное совпадение всех типов и границ: **{overall['exact_passed']}**, ошибки: **{overall['exact_failed']}**.", "", "Это набор специально составленных сценариев, а не независимая оценка качества на произвольных документах. " + note, "", "Эталон задан вручную в шаблонах до запуска детектора. Модель и правила при этом прогоне не менялись.", "", "| Категория | Область | Текстов | Полностью верно | Ошибок | Строгая weighted F1 |", "|---|---|---:|---:|---:|---:|"]
    for category, item in report["categories"].items():
        score = item["scores"]["strict_entity_weighted_f1"] if item["scores"] else 0.0
        lines.append(f"| {category} | {item['scope']} | {item['texts']} | {item['exact_passed']} | {item['exact_failed']} | {score:.4f} |")
    lines.extend(["", "## Области проверки", "", "core — поддержанные форматы и обычные варианты контекста; extended — форматы шире заявленных ограничений; semantic — спорные случаи, для которых явно задано желаемое семантическое поведение.", "", "В отрицательных категориях без эталонных сущностей weighted F1 по сущностям равна 0 по соглашению даже при отсутствии ошибок; для них смотрите число полностью верных текстов и FP."])
    for name, item in report["slices"].items():
        scores = item["scores"]
        if scores:
            lines.extend(["", f"## {name}", "", f"Текстов: {item['texts']}; полное совпадение: {item['exact_pass_rate']:.2%}; weighted F1 BIO без O: {scores['token_weighted_f1_without_o']:.4f}; строгая weighted F1 сущностей: {scores['strict_entity_weighted_f1']:.4f}."])
    lines.extend(["", "## Примеры ошибок", ""])
    for category, items in examples.items():
        lines.extend([f"### {category}", ""])
        for example in items[:3]:
            lines.extend([f"Сценарий `{example['id']}`. {example['note']}", "", "```text", example["text"], "```", "", "Пропущено:", "```json", json.dumps(example["missed"],ensure_ascii=False,indent=2), "```", "Лишние или с неверными границами:", "```json", json.dumps(example["false_positives"],ensure_ascii=False,indent=2), "```", ""])
    (directory/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description="Систематический стресс-тест сохранённой модели без обучения")
    parser.add_argument("--max-permutations",type=int,default=40320)
    parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--chunk-size",type=int,default=256)
    parser.add_argument("--device",choices=("auto","cuda","cpu"),default="auto")
    parser.add_argument("--model-dir",type=Path,default=PROJECT_ROOT/"models"/"rubert-pdn")
    parser.add_argument("--cases",type=Path,help="Использовать зафиксированный JSONL вместо генерации")
    parser.add_argument("--output-dir",type=Path,default=PROJECT_ROOT/"reports"/"stress-20261022")
    args=parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Папка результата уже существует; укажите другой --output-dir")
    if args.chunk_size<1 or args.batch_size<1:
        raise ValueError("Размеры пакетов должны быть положительными")
    started=time.monotonic()
    records=read_jsonl(args.cases) if args.cases else generate_cases(args.max_permutations)
    if not records:
        raise ValueError("Набор проверки пуст")
    args.output_dir.mkdir(parents=True)
    write_jsonl(args.output_dir/"cases.jsonl",records)
    print(json.dumps({"event":"generated","texts":len(records),"categories":dict(Counter(r['category'] for r in records))}),flush=True)
    model_dir=args.model_dir
    weight_hash=sha256(model_dir/"model.safetensors")
    program_files={str(p.relative_to(PROJECT_ROOT)):sha256(p) for p in sorted((PROJECT_ROOT/"src"/"pdn_lab").glob("*.py"))}
    detector=Detector(NameModel.load(model_dir,args.device))
    predictions=[]
    errors={}
    with (args.output_dir/"predictions.jsonl").open("w",encoding="utf-8") as output:
        for first in range(0,len(records),args.chunk_size):
            chunk=records[first:first+args.chunk_size]
            try:
                found=detector.detect_many([r["text"] for r in chunk],args.batch_size)
            except Exception:
                found=[]
                for record in chunk:
                    try: found.append(detector.detect_many([record["text"]],1)[0])
                    except Exception as error:
                        found.append([])
                        errors[record["id"]]=repr(error)
            predictions.extend(found)
            for record, entities in zip(chunk,found):
                output.write(json.dumps({"id":record["id"],"entities":[e.as_list() for e in entities],"error":errors.get(record["id"])},ensure_ascii=False)+"\n")
            if first%2048==0 or first+len(chunk)==len(records):
                print(json.dumps({"event":"inference","completed":first+len(chunk),"total":len(records),"seconds":round(time.monotonic()-started)}),flush=True)
    if sha256(model_dir/"model.safetensors")!=weight_hash:
        raise RuntimeError("Веса модели изменились во время проверки")
    examples=defaultdict(list)
    by_category=defaultdict(list)
    for index,(record,found) in enumerate(zip(records,predictions)):
        by_category[record["category"]].append(index)
        gold_set,pred_set=set(entities_of(record)),set(found)
        if (gold_set!=pred_set or record["id"] in errors) and len(examples[record["category"]])<8:
            examples[record["category"]].append({"id":record["id"],"note":record["note"],"text":record["text"],"expected":describe(record["text"],sorted(gold_set,key=lambda e:e.start)),"actual":describe(record["text"],found),"missed":describe(record["text"],sorted(gold_set-pred_set,key=lambda e:e.start)),"false_positives":describe(record["text"],sorted(pred_set-gold_set,key=lambda e:e.start)),"error":errors.get(record["id"])})
    categories={}
    for category,indices in by_category.items():
        selected=[records[i] for i in indices]
        categories[category]=summarize(selected,[predictions[i] for i in indices],errors)
        categories[category]["scope"]=sorted({r["scope"] for r in selected})
        print(json.dumps({"event":"category","category":category,"texts":len(indices),"failed":categories[category]["exact_failed"]}),flush=True)
    slices={}
    conditions={"supported_core":lambda r:r['scope']=='core',"without_exhaustive_permutations":lambda r:r['category']!='all_8_permutations',"core_without_exhaustive_permutations":lambda r:r['scope']=='core' and r['category']!='all_8_permutations',"extended_formats":lambda r:r['scope']=='extended',"semantic_context":lambda r:r['scope']=='semantic'}
    for name,condition in conditions.items():
        indices=[i for i,r in enumerate(records) if condition(r)]
        if indices:slices[name]=summarize([records[i] for i in indices],[predictions[i] for i in indices],errors)
    report={"created_at_utc":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":round(time.monotonic()-started),"model_sha256":weight_hash,"program_sha256":program_files,"cases_sha256":sha256(args.output_dir/"cases.jsonl"),"requested_permutations":args.max_permutations,"model_training_manifest":json.loads((model_dir/"training_manifest.json").read_text(encoding="utf-8")),"overall":summarize(records,predictions,errors),"categories":categories,"slices":slices,"inference_errors":errors}
    write_json(args.output_dir/"report.json",report)
    write_json(args.output_dir/"examples.json",dict(examples))
    write_report(report,examples,args.output_dir)
    print(json.dumps({"event":"finished","texts":report['overall']['texts'],"exact_passed":report['overall']['exact_passed'],"exact_failed":report['overall']['exact_failed'],"elapsed_seconds":report['elapsed_seconds'],"report":str(args.output_dir/'report.md')}),flush=True)


if __name__=="__main__":
    main()

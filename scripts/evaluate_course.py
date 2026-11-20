"""End-to-end evaluation of recognition and both methods on the same texts."""
import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from pdn_lab.anonymization import anon, mask
from pdn_lab.data import BASE_MODEL, MODEL_REVISION, sha256
from pdn_lab.detector import Detector
from pdn_lab.metrics import evaluate_predictions
from pdn_lab.model import NameModel, configure_runtime, default_model_directory
from pdn_lab.schema import COURSE_ATTRIBUTES, PROJECT_ROOT, entities_of, read_jsonl, write_json


def semantic_scores(references, candidates, sources, device, batch_size):
    from bert_score import BERTScorer
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    configure_runtime()
    required = ["config.json", "pytorch_model.bin", "vocab.txt", "tokenizer_config.json", "special_tokens_map.json"]
    options = dict(repo_id=BASE_MODEL, revision=MODEL_REVISION, allow_patterns=required)
    try:
        directory = Path(snapshot_download(**options, local_files_only=True))
        if not all((directory / name).is_file() for name in required):
            raise FileNotFoundError
    except (FileNotFoundError, OSError):
        directory = Path(snapshot_download(**options))
    tokenizer = AutoTokenizer.from_pretrained(directory, use_fast=False)
    max_positions = AutoConfig.from_pretrained(directory).max_position_embeddings
    unique = set(references).union(*candidates.values())
    max_observed = max(len(tokenizer.encode(t.strip(), truncation=False)) for t in unique)
    if max_observed > max_positions:
        raise ValueError(f"Текст содержит {max_observed} токенов при пределе {max_positions}")
    scorer = BERTScorer(model_type=str(directory), num_layers=9, device=device, idf=False,
                        rescale_with_baseline=False, use_fast_tokenizer=False, batch_size=batch_size)
    arrays = {name: np.column_stack([tensor.cpu().numpy() for tensor in
                                     scorer.score(texts, references, batch_size=batch_size)]).astype(float)
              for name, texts in candidates.items()}
    changed = np.array([a != b for a, b in zip(references, candidates["mask"])])
    groups = {"all": np.ones(len(sources), dtype=bool), "changed": changed}
    groups.update({source: np.array(sources) == source for source in sorted(set(sources))})
    result = {}
    for group, selection in groups.items():
        if not selection.any():
            continue
        scores = {name: dict(zip(("P", "R", "F1"), a[selection].mean(axis=0))) for name, a in arrays.items()}
        delta = (arrays["synonym"][:, 2] - arrays["mask"][:, 2])[selection]
        result[group] = {"texts": int(selection.sum()), **scores, "delta_f1": float(delta.mean()),
                         "wins": int((delta > 1e-6).sum()), "ties": int((np.abs(delta) <= 1e-6).sum()),
                         "losses": int((delta < -1e-6).sum())}
    return {"encoder": BASE_MODEL, "revision": MODEL_REVISION, "layer": 9, "idf": False,
            "rescale_with_baseline": False, "fast_tokenizer": False, "bert_score": version("bert-score"),
            "transformers": version("transformers"), "max_tokens_including_special": max_observed,
            "max_positions": max_positions, "scores": result}


def evaluate(args):
    manifest = json.loads(args.test.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    if sha256(args.test) != manifest["test_sha256"]:
        raise ValueError("Изменена зафиксированная тестовая выборка")
    records = read_jsonl(args.test)
    texts = [r["text"] for r in records]
    name_model = NameModel.load(args.model_dir, args.device)
    device = name_model.device
    detector = Detector(name_model, profile="course")
    predicted = detector.detect_many(texts, args.batch_size)
    ner = evaluate_predictions(records, predicted, COURSE_ATTRIBUTES)
    observed = [v["f1-score"] for tag, v in ner["token_bio_report"].items()
                if tag.startswith(("B-", "I-")) and v["support"] > 0]
    ner["macro_bio_with_reference_support"] = sum(observed) / len(observed)
    sources = [r["source"] for r in records]
    breakdown = {}
    for source in sorted(set(sources)):
        subset = [i for i, value in enumerate(sources) if value == source]
        scores = evaluate_predictions([records[i] for i in subset], [predicted[i] for i in subset], COURSE_ATTRIBUTES)
        breakdown[source] = {"texts": len(subset), "weighted_bio": scores["token_weighted_f1_without_o"],
                             "strict": scores["strict_entity_micro"]}
    candidates = {"synonym": [], "mask": []}
    coverage = Counter()
    for record, spans in zip(records, predicted):
        text = record["text"]
        try:
            new_text = anon(text, spans, seed=f"{args.seed}/{record['id']}")
        except ValueError as error:
            raise ValueError(f"Ошибка обезличивания {record['id']}: {error}") from error
        candidates["synonym"].append(new_text)
        candidates["mask"].append(mask(text, spans))
        for gold in entities_of(record):
            covered = sum(max(0, min(gold.end, p.end) - max(gold.start, p.start)) for p in spans)
            coverage["full" if covered == gold.end - gold.start else "partial" if covered else "missed"] += 1
    del detector, name_model
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()
    report = {"test": manifest, "model_sha256": sha256(args.model_dir / "model.safetensors"),
              "seed": args.seed, "device": device, "recognition": ner, "by_source": breakdown,
              "span_coverage": dict(coverage)}
    report["semantic"] = semantic_scores(texts, candidates, sources, device, args.batch_size)
    report["acceptance"] = {
        "dataset": len(records) >= 100 and min(ner["dataset"]["entities_per_attribute"].values()) >= 50,
        "weighted_bio_f1": ner["token_weighted_f1_without_o"] >= .85,
        "macro_attribute_f1": ner["token_attribute_report"]["macro avg"]["f1-score"] >= .8,
    }
    write_json(args.output, report)
    print(json.dumps({"acceptance": report["acceptance"], "weighted_bio": ner["token_weighted_f1_without_o"],
                      "macro_attribute_f1": ner["token_attribute_report"]["macro avg"]["f1-score"],
                      "semantic": report["semantic"]["scores"], "output": str(args.output)}, ensure_ascii=False, indent=2))
    return all(report["acceptance"].values())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test", type=Path, default=PROJECT_ROOT / "data/course-v2/heldout.jsonl")
    p.add_argument("--model-dir", type=Path, default=default_model_directory())
    p.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/course-v2.json")
    p.add_argument("--seed", type=int, default=8063)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--batch-size", type=int, default=8)
    raise SystemExit(0 if evaluate(p.parse_args()) else 2)

from __future__ import annotations

from collections import Counter

from sklearn.metrics import classification_report, f1_score

from .data import dataset_statistics
from .schema import ATTRIBUTES, Entity, entities_of, to_bio


def _score(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn, "tp": tp, "fp": fp, "fn": fn}


def evaluate_predictions(records: list[dict], predictions: list[list[Entity]], attributes=ATTRIBUTES) -> dict:
    if len(records) != len(predictions) or not records:
        raise ValueError("Оценка требует непустые тексты и по одному предсказанию на каждый текст")
    true_tags, predicted_tags = [], []
    counts = {label: Counter() for label in attributes}
    error_examples = []
    for record, predicted in zip(records, predictions):
        gold = entities_of(record)
        _, expected = to_bio(record["text"], gold)
        _, actual = to_bio(record["text"], predicted)
        true_tags.extend(expected)
        predicted_tags.extend(actual)
        gold_set, predicted_set = set(gold), set(predicted)
        for label in attributes:
            expected_set = {e for e in gold_set if e.label == label}
            actual_set = {e for e in predicted_set if e.label == label}
            counts[label].update(tp=len(expected_set & actual_set), fp=len(actual_set - expected_set), fn=len(expected_set - actual_set))
        if gold_set != predicted_set and len(error_examples) < 30:
            error_examples.append({"id": record["id"], "text": record["text"], "missed": [e.as_list() for e in sorted(gold_set - predicted_set, key=lambda e:e.start)], "false_positives": [e.as_list() for e in sorted(predicted_set - gold_set, key=lambda e:e.start)]})
    entity_scores = {label: _score(**{key: count[key] for key in ("tp", "fp", "fn")}) for label, count in counts.items()}
    total_support = sum(score["support"] for score in entity_scores.values())
    totals = {key: sum(count[key] for count in counts.values()) for key in ("tp", "fp", "fn")}
    entity_weighted = sum(score["f1"] * score["support"] for score in entity_scores.values()) / total_support if total_support else 0.0
    # O исключён только из списка классов усреднения. Все исходные токены сохранены,
    # поэтому предсказанная ПДн на месте O увеличивает FP соответствующего класса.
    labels = [f"{prefix}-{label}" for label in attributes for prefix in ("B", "I")]
    result = {
        "token_weighted_f1_without_o": float(f1_score(true_tags, predicted_tags, labels=labels, average="weighted", zero_division=0)),
        "token_weighted_f1_with_o": float(f1_score(true_tags, predicted_tags, labels=["O", *labels], average="weighted", zero_division=0)),
        "token_bio_report": classification_report(true_tags, predicted_tags, labels=labels, output_dict=True, zero_division=0),
        "strict_entity_micro": _score(**totals),
        "strict_entity_weighted_f1": entity_weighted,
        "strict_entities_by_attribute": entity_scores,
        "dataset": dataset_statistics(records, attributes),
        "error_examples": error_examples,
    }
    true_types = [tag.split("-",1)[-1] for tag in true_tags]
    predicted_types = [tag.split("-",1)[-1] for tag in predicted_tags]
    result["token_attribute_report"] = classification_report(true_types, predicted_types, labels=list(attributes), output_dict=True, zero_division=0)
    return result


def markdown_report(report: dict) -> str:
    scores = report["scores"]
    lines = ["# Проверка качества распознавания ПДн", "", f"Режим: `{report['mode']}`. Выборка: `{report['split']}`.", "", f"Weighted F1 по BIO без O: **{scores['token_weighted_f1_without_o']:.4f}**.", f"Weighted F1 по BIO с O: **{scores['token_weighted_f1_with_o']:.4f}**.", f"Строгая F1 по сущностям, micro: **{scores['strict_entity_micro']['f1']:.4f}**.", f"Строгая F1 по сущностям, weighted: **{scores['strict_entity_weighted_f1']:.4f}**.", "", "Строгое совпадение требует одинакового типа, начала и конца сущности.", "", "| Атрибут | Текстов | Precision | Recall | F1 | Сущностей |", "|---|---:|---:|---:|---:|---:|"]
    for label, score in scores["strict_entities_by_attribute"].items():
        lines.append(f"| {label} | {scores['dataset']['texts_per_attribute'][label]} | {score['precision']:.4f} | {score['recall']:.4f} | {score['f1']:.4f} | {score['support']} |")
    lines.extend(["", "## Условия лабораторной", ""])
    for key, value in report["requirements"].items():
        lines.append(f"- {key}: {value}")
    for source, subset in report.get("by_source", {}).items():
        lines.extend(["", f"## Источник {source}", "", f"Текстов: {subset['dataset']['texts']}; weighted F1 без O: {subset['token_weighted_f1_without_o']:.4f}; строгая weighted F1: {subset['strict_entity_weighted_f1']:.4f}."])
    lines.extend(["", "## Ограничения", "", "Результат относится только к указанной выборке. Синтетические тексты и публичный новостной корпус не доказывают качество на любых документах.", "Оценка без модели является диагностикой правил и не подтверждает выполнение всей лабораторной.", "Веса классов — число их появлений в эталонной разметке; O не участвует в основной средней, но ошибки на обычных словах учитываются.", "", "Примеры ошибок приведены в соседнем JSON-файле."])
    return "\n".join(lines) + "\n"

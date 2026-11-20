from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import time

from .data import BASE_MODEL, MODEL_REVISION, sha256
from .detector import Detector
from .metrics import evaluate_predictions
from .model import NameModel, configure_runtime
from .schema import MODEL_LABELS, NAME_ATTRIBUTES, entities_of, read_jsonl, to_bio, write_json


def train(data_dir: Path, output_dir: Path, *, epochs: int = 4, batch_size: int = 4, accumulation: int = 2, learning_rate: float = 3e-5, max_length: int = 256, seed: int = 20261022, device: str = "auto", resume_from: Path | None = None) -> dict:
    configure_runtime()
    import numpy as np
    import torch
    import transformers
    from torch.utils.data import DataLoader
    from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, get_linear_schedule_with_warmup

    if epochs < 1 or batch_size < 1 or accumulation < 1 or max_length < 32 or max_length > 512:
        raise ValueError("Некорректные параметры обучения")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Папка модели уже содержит файлы. Укажите новый --output-dir, чтобы сохранить предыдущую модель.")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна")
    train_records = read_jsonl(data_dir / "processed" / "train.jsonl")
    validation_records = read_jsonl(data_dir / "processed" / "validation.jsonl")
    if not train_records or not validation_records:
        raise ValueError("Обучающая и валидационная выборки должны быть непустыми")
    # test.jsonl здесь намеренно не открывается.
    source = str(resume_from) if resume_from else BASE_MODEL
    revision = None if resume_from else MODEL_REVISION
    if resume_from:
        previous = json.loads((resume_from / "training_manifest.json").read_text(encoding="utf-8"))
        if not previous.get("trained"):
            raise ValueError("Продолжение обучения требует ранее обученную модель проекта")
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision, use_fast=True)
    label2id = {label: index for index, label in enumerate(MODEL_LABELS)}
    model = AutoModelForTokenClassification.from_pretrained(source, revision=revision, use_safetensors=True if resume_from else False, num_labels=len(MODEL_LABELS), id2label=dict(enumerate(MODEL_LABELS)), label2id=label2id).to(device)
    features = []
    for record in train_records:
        tokens, tags = to_bio(record["text"], entities_of(record))
        tags = [tag if tag == "O" or tag.split("-", 1)[1] in NAME_ATTRIBUTES else "O" for tag in tags]
        if not tokens:
            continue
        encoded = tokenizer([token.text for token in tokens], is_split_into_words=True, truncation=True, max_length=max_length, return_overflowing_tokens=True, return_offsets_mapping=True)
        for window in range(len(encoded["input_ids"])):
            item = {key: encoded[key][window] for key in tokenizer.model_input_names if key in encoded}
            item["labels"] = [label2id[tags[word_index]] if word_index is not None and offset[0] == 0 else -100 for word_index, offset in zip(encoded.word_ids(batch_index=window), encoded["offset_mapping"][window])]
            if any(label != -100 for label in item["labels"]):
                features.append(item)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer, pad_to_multiple_of=8)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(features, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=0, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    steps = math.ceil(len(loader) / accumulation) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, round(steps * 0.1)), num_training_steps=steps)
    mixed = device == "cuda"
    dtype = torch.bfloat16 if mixed and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=mixed and dtype == torch.float16)
    tagger = NameModel(model, tokenizer, device, max_length)
    detector = Detector(tagger)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = -1.0
    started = time.monotonic()
    metadata = {"trained": True, "base_model": BASE_MODEL, "model_revision": MODEL_REVISION, "continued_from": str(resume_from) if resume_from else None, "seed": seed, "epochs_requested": epochs, "batch_size": batch_size, "gradient_accumulation": accumulation, "learning_rate": learning_rate, "max_length": max_length, "device": device, "precision": str(dtype) if mixed else "float32", "torch": torch.__version__, "transformers": transformers.__version__, "train_sha256": sha256(data_dir / "processed" / "train.jsonl"), "validation_sha256": sha256(data_dir / "processed" / "validation.jsonl"), "selection_metric": "validation_strict_entity_weighted_f1_names", "training_windows": len(features)}
    print(json.dumps({"event": "training_started", **metadata}, ensure_ascii=False), flush=True)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for batch_index, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            group_start = (batch_index // accumulation) * accumulation
            group_size = min(accumulation, len(loader) - group_start)
            with torch.autocast(device_type="cuda" if mixed else "cpu", dtype=dtype, enabled=mixed):
                loss = model(**batch).loss
                scaled_loss = loss / group_size
            if not torch.isfinite(loss):
                raise RuntimeError("Обучение остановлено: нечисловой loss")
            scaler.scale(scaled_loss).backward()
            total_loss += loss.item()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (batch_index + 1) % 50 == 0:
                print(json.dumps({"event": "progress", "epoch": epoch + 1, "batch": batch_index + 1, "batches": len(loader), "loss": round(total_loss / (batch_index + 1), 5), "elapsed_seconds": round(time.monotonic() - started)}), flush=True)
        predictions = detector.detect_many([record["text"] for record in validation_records], batch_size=batch_size * 2)
        scores = evaluate_predictions(validation_records, predictions)
        name_scores = [scores["strict_entities_by_attribute"][label] for label in NAME_ATTRIBUTES]
        support = sum(score["support"] for score in name_scores)
        selection_score = sum(score["f1"] * score["support"] for score in name_scores) / support if support else 0.0
        epoch_result = {"epoch": epoch + 1, "loss": total_loss / len(loader), "validation_names_f1": selection_score, "validation_hybrid_f1": scores["token_weighted_f1_without_o"], "validation_strict_entity_weighted_f1": scores["strict_entity_weighted_f1"], "elapsed_seconds": round(time.monotonic() - started)}
        history.append(epoch_result)
        print(json.dumps({"event": "epoch_finished", **epoch_result}), flush=True)
        if selection_score > best:
            best = selection_score
            model.save_pretrained(output_dir, safe_serialization=True)
            tokenizer.save_pretrained(output_dir)
            metadata.update(best_epoch=epoch + 1, best_validation_names_f1=best)
            write_json(output_dir / "training_manifest.json", metadata)
            write_json(output_dir / "validation_scores.json", scores)
        write_json(output_dir / "training_history.json", {"epochs": history})
    metadata["training_completed"] = True
    metadata["elapsed_seconds"] = round(time.monotonic() - started)
    write_json(output_dir / "training_manifest.json", metadata)
    return metadata

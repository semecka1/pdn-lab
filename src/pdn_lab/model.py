from __future__ import annotations

import json
import os
from pathlib import Path
import re

from .schema import MODEL_LABELS, PROJECT_ROOT, from_bio, tokenize


def _active_path(key: str, fallback: str) -> Path:
    pointer = PROJECT_ROOT / "models" / "active.json"
    value = json.loads(pointer.read_text(encoding="utf-8")).get(key, fallback) if pointer.exists() else fallback
    path = (PROJECT_ROOT / value).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError("Активный путь должен находиться внутри проекта")
    return path


def default_model_directory() -> Path:
    return _active_path("model_dir", "models/rubert-pdn")


def default_data_directory() -> Path:
    return _active_path("data_dir", "data")


def configure_runtime() -> None:
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class NameModel:
    def __init__(self, model, tokenizer, device: str, max_length: int = 256):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.max_length = max_length
        self.labels = {int(key): value for key, value in model.config.id2label.items()}
        if set(self.labels.values()) != set(MODEL_LABELS):
            raise ValueError("Модель должна иметь отдельные BIO-метки NAME, SURNAME и PATRONYMIC")
        if not tokenizer.is_fast:
            raise ValueError("Нужен быстрый токенизатор с word_ids и offset_mapping")

    @classmethod
    def load(cls, directory: Path, device: str = "auto"):
        configure_runtime()
        manifest = directory / "training_manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(f"Нет обученной модели в {directory}. Сначала выполните команду train. Диагностика правил доступна с --rules-only.")
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if not metadata.get("trained"):
            raise ValueError("Контрольная точка не помечена как обученная")
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        torch.set_num_threads(min(4, os.cpu_count() or 1))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA недоступна. Установите сборку PyTorch с CUDA или выберите --device cpu")
        model = AutoModelForTokenClassification.from_pretrained(directory, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(directory, use_fast=True, local_files_only=True)
        return cls(model.to(device), tokenizer, device, metadata.get("max_length", 256))

    def predict_many(self, texts: list[str], batch_size: int = 16):
        import torch

        token_lists = [tokenize(text) for text in texts]
        active = [index for index, tokens in enumerate(token_lists) if tokens]
        if not active:
            return [[] for _ in texts]
        encoding = self.tokenizer(
            [[token.text for token in token_lists[index]] for index in active],
            is_split_into_words=True, truncation=True, max_length=self.max_length,
            return_overflowing_tokens=True, stride=min(64, self.max_length // 4),
            return_offsets_mapping=True,
        )
        chosen = [{index: (-1, "O") for index in range(len(tokens))} for tokens in token_lists]
        self.model.eval()
        with torch.inference_mode():
            for first in range(0, len(encoding["input_ids"]), batch_size):
                last = min(first + batch_size, len(encoding["input_ids"]))
                features = [{key: encoding[key][index] for key in self.tokenizer.model_input_names if key in encoding} for index in range(first, last)]
                padded = self.tokenizer.pad(features, padding=True, return_tensors="pt")
                logits = self.model(**{key: value.to(self.device) for key, value in padded.items()}).logits
                predictions = logits.argmax(dim=-1).cpu().tolist()
                for local_index, window in enumerate(range(first, last)):
                    document = active[encoding["overflow_to_sample_mapping"][window]]
                    word_ids = encoding.word_ids(batch_index=window)
                    offsets = encoding["offset_mapping"][window]
                    length = len(word_ids)
                    for position, word_index in enumerate(word_ids):
                        if word_index is None or offsets[position][0] != 0:
                            continue
                        centrality = min(position, length - 1 - position)
                        if centrality > chosen[document][word_index][0]:
                            chosen[document][word_index] = (centrality, self.labels[predictions[local_index][position]])
        result = []
        for text, tokens, words in zip(texts, token_lists, chosen):
            if any(value[0] < 0 for value in words.values()):
                raise RuntimeError("Часть слов потерялась при обработке окон модели")
            entities = from_bio(tokens, [words[index][1] for index in range(len(tokens))])
            # Цифры и фрагменты адресов не могут быть компонентами ФИО.
            entities = [entity for entity in entities if re.fullmatch(r"[^\W\d_]+(?:[ \t’'.-][^\W\d_]+)*\.?", text[entity.start:entity.end])]
            result.append(entities)
        return result

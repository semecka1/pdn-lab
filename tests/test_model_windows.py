from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from pdn_lab.model import NameModel
from pdn_lab.schema import MODEL_LABELS


def test_overflow_and_subwords_preserve_original_long_text(tmp_path):
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "Иван", "Мар", "##ия"]
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text("\n".join(vocabulary), encoding="utf-8")
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=False)

    class ConstantNameModel(torch.nn.Module):
        config = SimpleNamespace(id2label=dict(enumerate(MODEL_LABELS)))

        def forward(self, input_ids, **kwargs):
            logits = torch.zeros((*input_ids.shape, len(MODEL_LABELS)), device=input_ids.device)
            logits[..., MODEL_LABELS.index("B-NAME")] = 10
            return SimpleNamespace(logits=logits)

    tagger = NameModel(ConstantNameModel(), tokenizer, "cpu", max_length=32)
    text = "Иван " * 600 + "🙂\r\nМария"
    found, empty = tagger.predict_many([text, ""], batch_size=8)
    assert empty == []
    assert len(found) == 601
    assert text[found[-1].start:found[-1].end] == "Мария"
    assert found[-1].end == len(text)

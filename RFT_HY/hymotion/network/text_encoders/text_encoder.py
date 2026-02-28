import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPTextModel,
    CLIPTokenizer,
)

from ...utils.type_converter import get_module_device
from .model_constants import PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION

USE_HF_MODELS = os.environ.get("USE_HF_MODELS", "0") == "1"

if USE_HF_MODELS:
    QWEN_PATH = "Qwen/Qwen3-8B"
    CLIP_PATH = "openai/clip-vit-large-patch14"
else:
    QWEN_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'pretrain', 'Qwen3-8B')
    CLIP_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'pretrain', 'clip-vit-large-patch14')

LLM_ENCODER_LAYOUT = {
    "qwen3": {
        "module_path": QWEN_PATH,
        "template": [
            {"role": "system", "content": f"{PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION}"},
            {"role": "user", "content": "{}"},
        ],
        "crop_start": 0,
        "tokenizer_class": AutoTokenizer,
        "text_encoder_class": AutoModelForCausalLM,
    },
}

SENTENCE_EMB_LAYOUT = {
    "clipl": {
        "module_path": CLIP_PATH,
        "tokenizer_class": CLIPTokenizer,
        "text_encoder_class": CLIPTextModel,
        "pooling_mode": "pooler_output",
        "max_length": 77,
    },
}


class HYTextModel(nn.Module):
    def __init__(
        self,
        llm_type: Optional[str] = "qwen3",
        max_length_llm: int = 512,
        sentence_emb_type: Optional[str] = "clipl",
        max_length_sentence_emb: int = 77,
        enable_llm_padding: bool = True,
    ) -> None:
        super().__init__()
        self.text_encoder_type = "hy_text_model"

        self.sentence_emb_type = sentence_emb_type
        self.sentence_emb_text_encoder = None
        self.sentence_emb_tokenizer = None
        self.vtxt_dim = 0
        if sentence_emb_type is not None:
            assert sentence_emb_type in SENTENCE_EMB_LAYOUT, f"Unsupported sentence embedding type: {sentence_emb_type}"
            self.max_length_sentence_emb = max_length_sentence_emb or SENTENCE_EMB_LAYOUT[sentence_emb_type].get(
                "max_length", 77
            )
            self._sentence_emb_pooling_mode = SENTENCE_EMB_LAYOUT[sentence_emb_type].get(
                "pooling_mode", "pooler_output"
            )
            tokenizer_kwargs = SENTENCE_EMB_LAYOUT[sentence_emb_type].get("tokenizer_kwargs", {})

            self.sentence_emb_tokenizer = SENTENCE_EMB_LAYOUT[sentence_emb_type]["tokenizer_class"].from_pretrained(
                SENTENCE_EMB_LAYOUT[sentence_emb_type]["module_path"],
                max_length=self.max_length_sentence_emb,
                **tokenizer_kwargs,
            )
            self.sentence_emb_text_encoder = SENTENCE_EMB_LAYOUT[sentence_emb_type][
                "text_encoder_class"
            ].from_pretrained(SENTENCE_EMB_LAYOUT[sentence_emb_type]["module_path"])
            self.sentence_emb_text_encoder = self.sentence_emb_text_encoder.eval().requires_grad_(False)
            self.vtxt_dim = self.sentence_emb_text_encoder.config.hidden_size

        self.llm_type = llm_type
        self.llm_text_encoder = None
        self.llm_tokenizer = None
        self.ctxt_dim = 0
        self.crop_start = 0
        self.max_length_llm = max_length_llm
        if llm_type is not None:
            assert llm_type in LLM_ENCODER_LAYOUT, f"Unsupported LLM type: {llm_type}"
            self._orig_max_length_llm = max_length_llm
            self.enable_llm_padding = enable_llm_padding
            self.llm_tokenizer = LLM_ENCODER_LAYOUT[llm_type]["tokenizer_class"].from_pretrained(
                LLM_ENCODER_LAYOUT[llm_type]["module_path"],
                padding_side="right",
            )
            self.llm_text_encoder = LLM_ENCODER_LAYOUT[llm_type]["text_encoder_class"].from_pretrained(
                LLM_ENCODER_LAYOUT[llm_type]["module_path"],
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
            )
            self.llm_text_encoder = self.llm_text_encoder.eval().requires_grad_(False)
            self.ctxt_dim = self.llm_text_encoder.config.hidden_size

            self.crop_start = self._compute_crop_start()
            self.max_length_llm = self._orig_max_length_llm + self.crop_start

    @torch.no_grad()
    def encode_llm(self, text: List[str]) -> Tuple[Tensor, Tensor]:
        if self.llm_type is None or self.llm_text_encoder is None or self.llm_tokenizer is None:
            raise ValueError("LLM model not initialized")

        device = get_module_device(self)
        llm_text = [
            (
                self.llm_tokenizer.apply_chat_template(
                    self.apply_text_to_template(one_text, LLM_ENCODER_LAYOUT[self.llm_type]["template"]),
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                if self.llm_type == "qwen3"
                else self.apply_text_to_template(one_text, LLM_ENCODER_LAYOUT[self.llm_type]["template"])
            )
            for one_text in text
        ]
        padding_mode = "max_length" if self.enable_llm_padding else False
        llm_batch_encoding = self.llm_tokenizer(
            llm_text,
            return_length=False,
            return_overflowing_tokens=False,
            truncation=True,
            return_attention_mask=True,
            max_length=self.max_length_llm,  # = crop_start + _orig_max_length_llm
            padding=padding_mode,
            return_tensors="pt",
        )
        llm_outputs = (
            self.llm_text_encoder(
                input_ids=llm_batch_encoding["input_ids"].to(device),
                attention_mask=llm_batch_encoding["attention_mask"].to(device),
                output_hidden_states=True,
            )
            if self.llm_type == "qwen3"
            else self.llm_text_encoder(
                input_ids=llm_batch_encoding["input_ids"].to(device),
                attention_mask=llm_batch_encoding["attention_mask"].to(device),
            )
        )
        if self.llm_type == "qwen3":
            ctxt_raw = llm_outputs.hidden_states[-1].clone()
        else:
            ctxt_raw = llm_outputs.last_hidden_state.clone()

        start = self.crop_start
        end = start + self._orig_max_length_llm
        ctxt_raw = ctxt_raw[:, start:end].contiguous()  # [bs, _orig_max_length_llm, hidden]
        ctxt_length = (llm_batch_encoding["attention_mask"].sum(dim=-1).to(device) - start).clamp(
            min=0, max=self._orig_max_length_llm
        )
        return ctxt_raw, ctxt_length

    @torch.no_grad()
    def encode_sentence_emb(self, text: List[str]) -> Tensor:
        if (
            self.sentence_emb_type is None
            or self.sentence_emb_text_encoder is None
            or self.sentence_emb_tokenizer is None
        ):
            raise ValueError("Sentence embedding model not initialized")

        device = get_module_device(self)
        enc = self.sentence_emb_tokenizer(
            text,
            return_length=False,
            return_overflowing_tokens=False,
            truncation=True,
            return_attention_mask=True,
            max_length=self.max_length_sentence_emb,
            padding=True,
            return_tensors="pt",
        )
        out = self.sentence_emb_text_encoder(
            input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)
        )
        if self._sentence_emb_pooling_mode == "pooler_output":
            # Pooler output pooling (clip-vit-large-patch14 等)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                vtxt_raw = out.pooler_output.unsqueeze(1)
            else:
                vtxt_raw = self._encode_pooling(enc["attention_mask"].to(device), out.last_hidden_state)
        elif self._sentence_emb_pooling_mode == "mean":
            vtxt_raw = self._encode_pooling(enc["attention_mask"].to(device), out.last_hidden_state)
        elif self._sentence_emb_pooling_mode == "last_token":
            vtxt_raw = self._last_token_pool(out.last_hidden_state, enc["attention_mask"].to(device))
        else:
            raise ValueError(f"Unknown pooling mode: {self._sentence_emb_pooling_mode}")

        return vtxt_raw

    def encode(self, text: List[str]) -> Tuple[Tensor, Tensor, Tensor]:
        ctxt_raw, ctxt_length = self.encode_llm(text=text)
        vtxt_raw = self.encode_sentence_emb(text=text)
        return vtxt_raw, ctxt_raw, ctxt_length

    @staticmethod
    def apply_text_to_template(text: str, template: Union[str, list]) -> Union[str, list]:
        if isinstance(template, str):
            return template.format(text)
        elif isinstance(template, list):
            return [
                {"role": "system", "content": f"{template[0]['content']}"},
                {"role": "user", "content": f"{text}"},
            ]
        else:
            raise TypeError(f"Unsupported template type: {type(template)}")

    def _compute_crop_start(self) -> int:
        if self.llm_type is None or self.llm_text_encoder is None or self.llm_tokenizer is None:
            raise ValueError("LLM model not initialized")

        def _find_subseq(a: str, b: str) -> int:
            for i in range(0, len(a) - len(b) + 1):
                if a[i : i + len(b)] == b:
                    return i
            return -1

        marker = "<BOC>"
        if self.llm_type == "qwen3":
            msgs = self.apply_text_to_template(marker, LLM_ENCODER_LAYOUT[self.llm_type]["template"])
            s = self.llm_tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False
            )
        else:
            s = self.apply_text_to_template(marker, LLM_ENCODER_LAYOUT[self.llm_type]["template"])
        full_ids = self.llm_tokenizer(s, return_tensors="pt", add_special_tokens=True)["input_ids"][0].tolist()
        marker_ids = self.llm_tokenizer(marker, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        pos = _find_subseq(full_ids, marker_ids)
        if pos >= 0:
            return pos
        else:
            return max(0, len(full_ids) - 1)

    def _pad_or_truncate_tensor(self, tensor: Tensor, target_length: int, dim: int = 0) -> Tensor:
        current_length = tensor.shape[dim]
        if current_length > target_length:
            return tensor.narrow(dim, 0, target_length)
        elif current_length < target_length:
            pad_shape = list(tensor.shape)
            pad_shape[dim] = target_length - current_length
            padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device) + tensor.narrow(dim, -1, 1)
            return torch.cat([tensor, padding], dim=dim)
        return tensor

    def _encode_pooling(self, attention_mask: Tensor, token_embeddings: Tensor) -> Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sentence_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        vtxt_raw = nn.functional.normalize(sentence_embeddings, p=2, dim=1).unsqueeze(1)  # shape of [bs, 1, D]
        return vtxt_raw

    def _last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            vtxt_raw = last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            vtxt_raw = last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]
        vtxt_raw = nn.functional.normalize(vtxt_raw, p=2, dim=-1).unsqueeze(1)  # shape of [bs, 1, D]
        return vtxt_raw


if __name__ == "__main__":
    # python -m hymotion.network.text_encoders.text_encoder
    text_encoder = HYTextModel(llm_type="qwen3", max_length_llm=5)
    vtxt_raw, ctxt_raw, ctxt_length = text_encoder.encode(["Hello, world!"])
    print(vtxt_raw.shape, ctxt_raw.shape, ctxt_length)

    crop_start = text_encoder._compute_crop_start()
    print(f"crop_start: {crop_start} when using {text_encoder.llm_type}")

    assert (
        vtxt_raw.shape[1:] == (1, text_encoder.vtxt_dim)
        and ctxt_raw.shape[1:] == (text_encoder._orig_max_length_llm, text_encoder.ctxt_dim)
        and torch.all((ctxt_length >= 0) & (ctxt_length <= text_encoder._orig_max_length_llm))
    ), f"Got unexpected output shape: {vtxt_raw.shape}, {ctxt_raw.shape}, {ctxt_length}"

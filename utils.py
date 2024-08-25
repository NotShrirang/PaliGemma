from core.gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os


def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right", "Tokenizer padding side must be right"

    safetensors_file = glob.glob(os.path.join(model_path, "*.safetensors"))

    tensors = {}

    for safetensors_file in safetensors_file:
        with safe_open(safetensors_file, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config_file = json.load(f)
        config = PaliGemmaConfig(**model_config_file)

    model = PaliGemmaForConditionalGeneration(config).to(device)

    model.load_state_dict(tensors)

    model.tie_weights()

    return (model, tokenizer)
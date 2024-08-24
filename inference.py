from PIL import Image
import torch
import fire

from core.processor import PaliGemmaProcessor
from core.gemma import PaliGemmaForConditionalGeneration, KVCache
from utils import load_hf_model


def move_inputs_to_device(model_inputs: dict, device: str):
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs


def _sample_top_p(logits: torch.Tensor, p: float):
    logit_sort, logit_idx = torch.sort(logits, dim=-1, descending=True)
    logit_sum = torch.cumsum(logit_sort, dim=-1)
    mask = logit_sum - logit_sort > p
    logit_sort[mask] = 0.0
    logit_sort.div_(logit_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(logit_sort, num_samples=1)
    next_token = torch.gather(logit_idx, -1, next_token)
    return next_token


def get_model_inputs(
    processor: PaliGemmaProcessor,
    prompt: str,
    image_file_path: str,
    device: str,
) -> dict:
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs


def test_inference(
    model: PaliGemmaForConditionalGeneration,
    processsor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    model_inputs = get_model_inputs(processsor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    image_features = model_inputs["image_features"]

    kv_cache = KVCache()

    stop_token = processsor.tokenizer.eos_token_id
    generated_tokens = []

    for _ in range(max_tokens_to_generate):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_features=image_features,
            kv_cache=kv_cache,
        )
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]
        if do_sample:
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = _sample_top_p(next_token_logits, top_p)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        assert next_token.size() == (1, 1)
        next_token = next_token.squeeze(0)
        generated_tokens.append(next_token)
        if next_token.item() == stop_token:
            break
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=-1
        )
    
    generated_tokens = torch.cat(generated_tokens, dim=-1)
    decoded = processsor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(prompt + decoded)

def main(
    model_path: str = None,
    prompt: str = None,
    image_path: str = None,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
    only_cpu: bool = False,
):
    device = "cpu"

    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"

    print("Device in use:", device)

    print("Loading model...")

    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()

    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    print("Running inference...")

    with torch.no_grad():
        test_inference(
            model,
            processor,
            device,
            prompt,
            image_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
        )

    print("Done!")


if __name__ == '__main__':
    fire.Fire(main)
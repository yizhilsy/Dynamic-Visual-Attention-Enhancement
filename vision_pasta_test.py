from vision_pastalib.pasta_mm import PASTAMM
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from transformers import AutoTokenizer

# Initialize pre-trained LlaVA MLLM
model_path = "/s/llava-series/llava-v1.5-7b"
device = "cuda:1"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    use_flash_attn=False,
    device="cuda",
    device_map=device,
)

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

# Select the attention heads to be steered, 
# following the format of {'layer_id': [head_ids]}: 
head_config = {
    "3": [17, 7, 6, 12, 18], "8": [28, 21, 24], "5": [24, 4], 
    "0": [17], "4": [3], "6": [14], "7": [13], "11": [16], 
}

pastamm = PASTAMM(
    model=model,
    tokenizer=tokenizer,
    image_processor=image_processor,
    head_config=head_config,
    alpha=0.01, # scaling coefficient
    scale_position="exclude", # downweighting text tokens
)


from datasets import load_dataset

dataset = load_dataset(
    "parquet",
    data_files="/s/datasets/MME/data/*.parquet",
    split="train"
)

prompts = [dataset[0]['question'], dataset[1]['question']]
images = [dataset[0]['image'], dataset[1]['image']]

pastamm.inputs_from_batch(
    prompts,
    images,
    tokenizer,
    image_processor,
    device,
)


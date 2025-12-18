import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math

from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer

from typing import Tuple
from peft import PeftModel

def load_lora_finetuned_model(base_model_name_or_path: str, lora_model_name_or_path:str, device: str, torch_dtype: torch.dtype) -> tuple[LlavaForConditionalGeneration, AutoProcessor, AutoTokenizer]:
    model = LlavaForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=base_model_name_or_path,
        device_map=device,
        torch_dtype=torch_dtype
    )

    if lora_model_name_or_path is not None:
        # Loading LoRA weights
        model = PeftModel.from_pretrained(model, lora_model_name_or_path)
        # Merging LoRA weights
        model = model.merge_and_unload()
    
    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=base_model_name_or_path)
    tokenizer = processor.tokenizer
    image_processor = processor.image_processor
    return model, processor, tokenizer, image_processor

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    lora_path = os.path.expanduser(args.lora_path) if args.lora_path else None
    device = args.device
    
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    model, processor, tokenizer, image_processor = load_lora_finetuned_model(model_path, lora_path, device, torch.bfloat16)

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        question = line['conversations'][0]
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs

        if 'image' in line:
            image_file = line["image"]
            images = Image.open(os.path.join(args.image_folder, image_file))
            # image_tensor = process_images([image], image_processor, model.config)[0]
            # images = image_tensor.unsqueeze(0).half().cuda()
            # image_sizes = [image.size]
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            images = None
            image_sizes = None

        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly without explanation and punctuation."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly without explanation and punctuation."

        # conv = conv_templates[args.conv_mode].copy()
        # conv.append_message(conv.roles[0], qs)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": qs},
        ]
        prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        input_ids = None
        if images is not None:
            input_ids = processor(images=images, text=prompt, return_tensors="pt")
        else:
            input_ids = tokenizer([prompt], return_tensors="pt").to(model.device)
        
        for temp_key in input_ids.keys():
            input_ids[temp_key] = input_ids[temp_key].to(device)

        # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        with torch.inference_mode():
            if 'pixel_values' in input_ids:  # 如果包含图片信息调用整个llava模型进行推理
                output_ids = model.generate(
                    **input_ids,
                    # images=images,
                    # image_sizes=image_sizes,
                    # do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=1024,
                    use_cache=True,
                )
            else:   # 如果不包含图片信息调用llava模型中的文本大语言模型进行推理
                output_ids = model.language_model.generate(
                    **input_ids,
                    temperature=args.temperature,
                    max_new_tokens=1024,
                    use_cache=True,
                )

        # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        outputs = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        extra_answer = outputs.split("assistant\n")[-1].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": extra_answer,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    eval_model(args)

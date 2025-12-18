"""
    Batch Inference for VQA type tasks using a huggingface multimodal model.
    Author: Shiyu Lu
"""
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
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer, my_LlavaForConditionalGeneration
from peft import PeftModel
from typing import Optional, List
from itertools import islice

def load_lora_finetuned_model(base_model_name_or_path: str, device: str, torch_dtype: torch.dtype, model_type: str,
                              lora_model_name_or_path: Optional[str] = None) -> tuple[LlavaForConditionalGeneration, AutoProcessor, AutoTokenizer]:
    model = None
    if model_type == "default":
        model = LlavaForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=base_model_name_or_path,
            device_map=device,
            torch_dtype=torch_dtype
        )
    elif model_type == "3layers":
        model = my_LlavaForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=base_model_name_or_path,
            device_map=device,
            torch_dtype=torch_dtype
        )
        
    """
        NOTE: append more layers situation
    """

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


def preprocess_multimodal(q_text: str):
    if DEFAULT_IMAGE_TOKEN in q_text:
        q_text = q_text.replace(DEFAULT_IMAGE_TOKEN, '').strip()
        q_text = DEFAULT_IMAGE_TOKEN + '\n' + q_text
        q_text = q_text.strip()
    return q_text


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, processor, model_config, chat_template_type="qwen"):
        self.questions = questions
        self.image_folder = image_folder
        self.processor = processor
        self.model_config = model_config
        self.chat_template_type = chat_template_type

    """
        NOTE: 返回 conversation->prompt && image 原始数据
    """
    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]

        if getattr(self.model_config, 'mm_use_im_start_end', False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        # conv = conv_templates[args.conv_mode].copy()
        # conv.append_message(conv.roles[0], qs)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()

        prompt = None
        if self.chat_template_type == "qwen":   # 使用Qwen作为语言模型的chat_template格式
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": qs},
            ]
            prompt = self.processor.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
        elif self.chat_template_type == "llama":    # 原生hf-llava所用语言模型的chat_template格式
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": line["text"]},
                    ],
                },
            ]
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
        
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        return prompt, image

    def __len__(self):
        return len(self.questions)

class BatchInferenceCollator:
    def __init__(self, processor: AutoProcessor):
        self.processor = processor

    def __call__(self, batch: List) -> dict:
        prompts, images = zip(*batch) # prompts: [], images: []
        inputs = self.processor(images=images, text=prompts, padding=True, return_tensors="pt") # 构造一个batch的tokenizer后的数据        
        return inputs

# DataLoader
def create_data_loader(questions, image_folder, processor, model_config, batch_size=1, num_workers=4, chat_template_type="qwen"):
    # assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, processor, model_config, chat_template_type)
    batchinferencecollator = BatchInferenceCollator(processor)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, 
                             shuffle=False, collate_fn=batchinferencecollator, pin_memory=True)
    return data_loader

def batch_iterable(iterable, batch_size):
    """把list分成以batch为一组的小块列表"""
    it = iter(iterable)
    while batch := list(islice(it, batch_size)):
        yield batch

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    lora_path = os.path.expanduser(args.lora_path) if args.lora_path is not None else None

    model, processor, tokenizer, image_processor = load_lora_finetuned_model(base_model_name_or_path=model_path, device=args.device, 
                                                                             torch_dtype=torch.bfloat16, model_type=args.model_type,
                                                                             lora_model_name_or_path=lora_path)

    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    answers_file = os.path.expanduser(args.answers_file)
    has_inference_question = {} # 记录断点续推时已经完成推理的问题主键
    if os.path.exists(answers_file):
        # 记录已经完成推理的问题序号
        has_inference_question = {json.loads(a)['question_id'] for a in open(answers_file, "r")}
        print(f"Answers file {answers_file} already exists. Continue the inference progress from the breakdown: {len(has_inference_question)}.")
    else:
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    # 根据 has_inference_question 过滤掉已经完成推理的问题
    questions = [q for q in questions if q["question_id"] not in has_inference_question]
    ans_file = open(answers_file, "a")

    # if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
    #     args.conv_mode = args.conv_mode + '_mmtag'
    #     print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, processor, model.config, args.batch_size, chat_template_type=args.chat_template_type)

    for inputs, lines in tqdm(zip(data_loader, batch_iterable(questions, args.batch_size)), total=len(data_loader)):
        # idx = line["question_id"]
        # cur_prompt = line["text"]

        for temp_key in inputs.keys():
            inputs[temp_key] = inputs[temp_key].to(args.device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                # images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                # image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True
            )
        # 将推理得到的token_id反向量化为字符串
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for i, output in enumerate(outputs):
            output = output.strip()
            extra_answer = output.split("assistant\n")[-1].strip()
            idx = lines[i]["question_id"]
            cur_prompt = lines[i]["text"]

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
    # parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    # parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--model_type", type=str, default="default")
    parser.add_argument("--chat_template_type", type=str, default="qwen", help="The chat template type to use for generating prompt.")
    args = parser.parse_args()

    eval_model(args)

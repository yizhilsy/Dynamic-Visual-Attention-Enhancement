"""
    PASTAMM MultiModal Implementation
    Arthor: Shiyu Lu
    Date: 19 December 2025
"""
import torch
import abc, json
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, cast, overload, Tuple
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle

import transformers 
from vision_pastalib.utils import tokenizer_utils
from vision_pastalib.utils.typing_mm import (
    Model,
    Dataset,
    Device,
    ModelInput,
    ModelOutput,
    StrSequence,
    Tokenizer,
    TokenizerOffsetMapping,
) 

from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
import PIL
from transformers import CLIPImageProcessor

class PASTAMM(abc.ABC):
    ATTN_MODULE_NAME = {
        "llava": "model.model.layers.{}.self_attn",
    }
    ATTENTION_MASK_ARGIDX = {}

    def __init__(
        self,
        model,
        tokenizer,
        image_processor,
        head_config: dict|list|None = None, 
        alpha: float = 0.01,
        scale_position: str = "exclude",
    ):
        # 初始化模型
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.setup_model(model)

        # 设置 PASTA 相关参数
        self.alpha = alpha
        self.scale_position = scale_position
        self.setup_head_config(head_config)
        
        assert self.scale_position in ['include', 'exclude', 'generation']
        assert self.alpha > 0
        self.scale_constant = None
    
    def setup_model(self, model):
        """
            Obtain the model type and complete the configuration.
            获取模型类型并获取模型每层注意力头的数量
        """
        if isinstance(model, transformers.LlamaForCausalLM):
            self.model_name = "llama"
            self.num_attn_head = model.config.num_attention_heads
        elif isinstance(model, transformers.GPTJForCausalLM):
            self.model_name = "gptj"
            self.num_attn_head = model.config.n_head
        elif isinstance(model, transformers.MistralForCausalLM):
            self.model_name = "mistral"
            self.num_attn_head = model.config.num_attention_heads
        elif isinstance(model, transformers.GemmaForCausalLM):
            self.model_name = "gemma"
            self.num_attn_head = model.config.num_attention_heads
        elif model.__class__.__name__ == "Phi3ForCausalLM":
            self.model_name = "phi3mini"
            self.num_attn_head = model.config.num_attention_heads

        # 多模态领域中的模型
        elif isinstance(model, LlavaLlamaForCausalLM):
            self.model_name = "llava"
            self.num_attn_head = model.config.num_attention_heads
        else:
            raise ValueError("Unimplemented Model Type.")
    
    def setup_head_config(self, head_config):
        """
        Config the attention heads to be steered.

        If `head_config` is `list` of layer index, PASTA will steer the entire layers. 
        """
        if isinstance(head_config, dict):
            self.head_config = {int(k):v for k,v in head_config.items()} 
            self.all_layers_idx = [int(key) for key in head_config]
        elif isinstance(head_config, list):
            self.all_layers_idx = [int(v) for v in head_config]
            self.head_config = {
                idx:list(range(self.num_attn_head)) for idx in self.all_layers_idx
            }
        else:
            raise ValueError(f"Incorrect head config: {head_config}")
    
    def _maybe_batch(self, text: str | StrSequence) -> StrSequence:
        """Batch the text if it is not already batched."""
        if isinstance(text, str):
            return [text]
        return text
    

    # edit the vision tokens' attention scores
    def edit_vision_attention_mask(
        self,
        module: torch.nn.Module,
        input_args: tuple,
        input_kwargs: dict,
        head_idx: list[int],
        image_position_mask: torch.Tensor,
    ):
        """
        The hook function registerred pre-forward for attention models. 

        Args: 
            module ([`torch.nn.Module`]): The registerred attention modules. 
            input_args (`tuple`): The positional arguments of forward function. 
            input_kwargs (`dict`): The keyword arguments of forward function. 
            head_idx (`list[int]`): The index of heads to be steered. 
            image_position_mask (`torch.Tensor`): A mask tensor indicating the position of image token(highlight token)
            input_len (`int`): The length L of inputs.

        Returns: 
            tuple, dict: return the modified `attention_mask`,
                while not changing other input arguments. 
        """
        if "attention_mask" in input_kwargs:
            attention_mask = input_kwargs['attention_mask'].clone()
        elif input_args is not None:
            arg_idx = self.ATTENTION_MASK_ARGIDX[self.model_name]
            attention_mask = input_args[arg_idx].clone()
        else:
            raise ValueError(f"Not found attention masks in {str(module)}")
        
        bsz, head_dim, tgt_len, src_len = attention_mask.size()
        dtype, device = attention_mask.dtype, attention_mask.device
        if head_dim != self.num_attn_head:  # 初始的 attention_mask 可能是不同注意力头共享的
            attention_mask = attention_mask.expand(
                bsz, self.num_attn_head, tgt_len, src_len
            ).clone()
        if not self.scale_constant:
            self.scale_constant = torch.Tensor([self.alpha]).to(dtype).to(device).log()
        
        # 基于 image_position_mask 进行视觉token的注意力增强
        for bi, row_image_position_mask in enumerate(image_position_mask):
            if self.scale_position == "include":
                attention_mask[bi, head_idx, :, row_image_position_mask] += self.scale_constant
            elif self.scale_position == "exclude":
                attention_mask[bi, head_idx, :, ~row_image_position_mask] += self.scale_constant
            else:
                raise ValueError(f"Unexcepted {self.scale_position}.")
        if self.scale_position == "include":
            attention_mask[:, head_idx, :, :] -= self.scale_constant

        if self.model_name in ["llava", "Qwen"]:
            attention_mask.old_size = attention_mask.size
            attention_mask.size = lambda:(bsz, 1, tgt_len, src_len)

        if "attention_mask" in input_kwargs:
            input_kwargs['attention_mask'] = attention_mask 
            return input_args, input_kwargs
        else:
            return (input_args[:arg_idx], attention_mask, *input_args[arg_idx+1:]), input_kwargs




    # def vision_token_ranges_from_batch(
    #     self,
    #     strings: str | StrSequence,
    #     substrings: str | StrSequence,
    #     offsets_mapping: Sequence[TokenizerOffsetMapping],
    #     occurrence: int = 0,
    # ) -> torch.Tensor:
    #     """Return shape (batch_size, 2) tensor of token ranges for (str, substr) pairs."""
    #     strings = self._maybe_batch(strings)
    #     substrings = self._maybe_batch(substrings)
    #     if len(strings) != len(substrings): # check batch size consistency
    #         raise ValueError(
    #             f"got {len(strings)} strings but only {len(substrings)} substrings"
    #         )
    #     return torch.tensor(
    #         [
    #             tokenizer_utils.find_token_range(
    #                 string, substring, offset_mapping=offset_mapping, occurrence=occurrence
    #             )
    #             for string, substring, offset_mapping in zip(
    #                 strings, substrings, offsets_mapping
    #             )
    #         ]
    #     )
    
    @contextmanager
    def apply_mm_steering(
        self,
        model: Model,
        multimodal_input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        image_position_mask: torch.Tensor
    ):
        """
            The function of context manager to register the pre-forward hook on `model`. 
            Args:
                model ([`transformers.PreTrainedModel`]): The transformer model to be steered. 
                multimodal_input_embeds (`torch.Tensor`): The multimodal concated embeddings cantaining the image embeddings.
                ... 
        """
        registered_hooks = []
        for layer_idx in self.all_layers_idx:
            name = self.ATTN_MODULE_NAME[self.model_name].format(layer_idx)
            module = model.get_submodule(name) 
            # Prepare the hook function with partial arguments being fixed. 
            # Pass the head_idx, image_position_mask for each attention module in advance. 
            hook_func = partial(
                self.edit_vision_attention_mask,
                head_idx = self.head_config[layer_idx],
                image_position_mask = image_position_mask
            )
            # hook函数挂载到llm模型指定的层上
            registered_hook = module.register_forward_pre_hook(hook_func, with_kwargs=True)
            registered_hooks.append(registered_hook)

        try:
            yield model
        except Exception as error:
            raise error
        finally:
            for registered_hook in registered_hooks:
                registered_hook.remove()


    def inputs_from_batch(
        self, 
        prompts: str | StrSequence,
        images: list[PIL.Image],
        tokenizer: Tokenizer|None = None,
        image_processor: CLIPImageProcessor|None = None,
        device: Optional[Device] = None,
    ) -> tuple[ModelInput, Sequence[TokenizerOffsetMapping]]:
        """
        Precompute model's multimodal inputs.
        
        Args:
            prompts: str or list of text prompts.
            images: list of PIL images.
            tokenizer: tokenizer to use. If None, use self.tokenizer.
            image_processor: image processor to use. If None, use self.image_processor.
            device: device to move the inputs to. If None, do not move.
        """

        if tokenizer is None:
            tokenizer = self.tokenizer
        if image_processor is None:
            image_processor = self.image_processor
        
        # 以对话模板形式将原始prompt嵌入
        conv_prompts = []
        for prompt in prompts:
            qs = prompt.replace('<image>', '').strip()
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            conv = conv_templates['vicuna_v1'].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            conv_prompts.append(conv.get_prompt())

        # with tokenizer_utils.set_padding_side(tokenizer, padding_side="left"):
        #     inputs = tokenizer(
        #         prompts,
        #         return_tensors="pt",
        #         truncation=False,
        #         padding="longest",
        #         return_offsets_mapping=True,    # 返回字符级到 token 的偏移映射
        #     )
        #     offset_mapping = inputs.pop("offset_mapping")
        #     input_ids = inputs.pop("input_ids")
        #     attention_mask = inputs.pop("attention_mask")
        
        # 序列化对话形式的prompt并进行右对齐
        conv_input_ids_list = []
        for conv_prompt in conv_prompts:
            conv_input_ids = tokenizer_image_token(conv_prompt, tokenizer, IMAGE_TOKEN_INDEX, 
                                                   return_tensors='pt').unsqueeze(0)
            conv_input_ids_list.append(conv_input_ids)
        
        max_len = max(conv_input_ids.size(1) for conv_input_ids in conv_input_ids_list)
        padded_conv_input_ids_list = []
        attention_mask = []
        for conv_input_ids in conv_input_ids_list:
            seq_len = conv_input_ids.size(1)
            pad_len = max_len - seq_len
            if pad_len > 0:
                pad_tensor = torch.full(
                    (1, pad_len),
                    tokenizer.pad_token_id,
                    dtype=conv_input_ids.dtype,
                    device=conv_input_ids.device
                )
                padded_conv_input_ids = torch.cat([pad_tensor, conv_input_ids], dim=1)  # 左 pad
                attn_mask = torch.cat(
                    [torch.zeros(1, pad_len, device=conv_input_ids.device), 
                     torch.ones(1, seq_len, device=conv_input_ids.device)],
                    dim=1
                )
            else:
                padded_conv_input_ids = conv_input_ids
                attn_mask = torch.ones_like(padded_conv_input_ids)
            padded_conv_input_ids_list.append(padded_conv_input_ids)
            attention_mask.append(attn_mask)
        input_ids = torch.cat(padded_conv_input_ids_list, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)


        images_tensor = process_images(images, image_processor, self.model.config)

        # images_tensor = image_processor.preprocess(
        #     images,
        #     return_tensors="pt",
        # )["pixel_values"].to(dtype=self.model.dtype)

        
        if device is not None:
            input_ids = input_ids.to(device=device)
            attention_mask = attention_mask.to(device=device)
            images_tensor = images_tensor.to(device=device, dtype=self.model.dtype)
        
        # use llava model's function prepare_inputs_labels_for_multimodal to generate multimodal inputs
        _, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, image_position_mask = self.model.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=attention_mask,
            past_key_values=None,
            labels=None,
            images=images_tensor,
        )

        # pasta_mm
        # return multimodal concated embedding tensor and corresponding attention_mask and corresponding image_position_mask
        return new_input_embeds, attention_mask, image_position_mask, input_ids, images_tensor
    


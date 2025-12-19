"""PASTA MultiModal Implementation"""
import torch
import abc, json
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, cast, overload, Tuple

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

        with tokenizer_utils.set_padding_side(tokenizer, padding_side="left"):
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                truncation=False,
                padding="longest",
                return_offsets_mapping=True,    # 返回字符级到 token 的偏移映射
            )
            offset_mapping = inputs.pop("offset_mapping")

        images_tensor = image_processor.preprocess(
            images,
            return_tensors="pt",
        )["pixel_values"].to(dtdtype=self.model.dtype)




        if device is not None:
            inputs = inputs.to(device)
            images_tensor = images_tensor.to(device)
        return inputs, images_tensor, offset_mapping
    


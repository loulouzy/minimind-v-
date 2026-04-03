import os
import warnings
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import Siglip2ImageProcessor, Siglip2VisionModel
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from .model_minimind import *

warnings.filterwarnings("ignore")

# This file extends the original MiniMind-V implementation while preserving
# the original replace-based fusion path. The added Q-Former and text
# cross-attention modules are incremental experimental branches built on top
# of the upstream project rather than a replacement for the original design.


def get_vlm_arch_suffix(config) -> str:
    parts = []
    if getattr(config, "vision_fusion_type", "replace") == "qformer_cross_attn":
        parts.append("qformer")
    if getattr(config, "use_moe", False):
        parts.append("moe")
    return f"_{'_'.join(parts)}" if parts else ""


class VLMConfig(MiniMindConfig):
    model_type = "minimind-v"

    def __init__(self, image_special_token="<|image_pad|>", image_ids=[12], **kwargs):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        self.image_hidden_size = kwargs.get("image_hidden_size", 768)
        self.image_token_len = kwargs.get("image_token_len", 64)
        self.vision_fusion_type = kwargs.get("vision_fusion_type", "replace")
        self.qformer_num_queries = kwargs.get("qformer_num_queries", 32)
        self.qformer_num_layers = kwargs.get("qformer_num_layers", 2)
        self.text_cross_attn_every_n_layers = kwargs.get("text_cross_attn_every_n_layers", 1)
        super().__init__(**kwargs)


class MMVisionProjector(nn.Module):
    def __init__(self, in_dim, out_dim, source_tokens=256, target_tokens=64):
        super().__init__()
        self.target_tokens = target_tokens
        self.merge = source_tokens // target_tokens
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * self.merge, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        b, _, d = x.shape
        x = x.reshape(b, self.target_tokens, d * self.merge)
        return self.mlp(x)


class CrossAttentionAdapter(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0, eps=1e-6):
        super().__init__()
        self.query_norm = RMSNorm(hidden_size, eps=eps)
        self.key_value_norm = RMSNorm(hidden_size, eps=eps)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, query_states, key_value_states):
        q = self.query_norm(query_states)
        kv = self.key_value_norm(key_value_states)
        attn_output, _ = self.cross_attn(q, kv, kv, need_weights=False)
        return query_states + self.resid_dropout(attn_output)


class QFormerBlock(nn.Module):
    def __init__(self, config: VLMConfig):
        super().__init__()
        self.self_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.self_dropout = nn.Dropout(config.dropout)
        self.cross_attn = CrossAttentionAdapter(
            config.hidden_size,
            config.num_attention_heads,
            dropout=config.dropout,
            eps=config.rms_norm_eps,
        )
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = FeedForward(config)

    def forward(self, query_states, vision_states):
        q = self.self_norm(query_states)
        self_attn_output, _ = self.self_attn(q, q, q, need_weights=False)
        query_states = query_states + self.self_dropout(self_attn_output)
        query_states = self.cross_attn(query_states, vision_states)
        query_states = query_states + self.ffn(self.ffn_norm(query_states))
        return query_states


class MiniMindQFormer(nn.Module):
    def __init__(self, config: VLMConfig):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, config.qformer_num_queries, config.hidden_size) * 0.02)
        self.vision_to_llm_proj = nn.Linear(config.image_hidden_size, config.hidden_size)
        self.layers = nn.ModuleList([QFormerBlock(config) for _ in range(config.qformer_num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, vision_states):
        query_states = self.query_tokens.expand(vision_states.size(0), -1, -1)
        vision_states = self.vision_to_llm_proj(vision_states)
        for layer in self.layers:
            query_states = layer(query_states, vision_states)
        return self.norm(query_states)


class MiniMindVLM(MiniMindForCausalLM):
    config_class = VLMConfig

    def __init__(self, config: VLMConfig = None, vision_model_path="./model/siglip2-base-p16-ve"):
        self.config = config or VLMConfig()
        super().__init__(self.config)
        self.vision_encoder, self.processor = self.__class__.get_vision_model(vision_model_path)
        self.vision_proj = MMVisionProjector(
            self.config.image_hidden_size,
            self.config.hidden_size,
            target_tokens=self.config.image_token_len,
        )
        self.qformer = None
        self.text_cross_attn_layers = None
        if self.config.vision_fusion_type == "qformer_cross_attn":
            self.qformer = MiniMindQFormer(self.config)
            every_n = max(1, int(self.config.text_cross_attn_every_n_layers))
            self.text_cross_attn_layers = nn.ModuleList(
                [
                    CrossAttentionAdapter(
                        self.config.hidden_size,
                        self.config.num_attention_heads,
                        dropout=self.config.dropout,
                        eps=self.config.rms_norm_eps,
                    )
                    if layer_idx % every_n == 0
                    else nn.Identity()
                    for layer_idx in range(len(self.model.layers))
                ]
            )

    @staticmethod
    def get_vision_model(model_path: str):
        from transformers import logging as hf_logging

        hf_logging.set_verbosity_error()
        if not os.path.exists(model_path):
            return None, None
        model = Siglip2VisionModel.from_pretrained(model_path)
        processor = Siglip2ImageProcessor.from_pretrained(model_path)
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor):
        if image.mode in ["RGBA", "LA"]:
            image = image.convert("RGB")
        return processor(images=image, return_tensors="pt")

    @staticmethod
    def get_image_embeddings(image_inputs, vision_model):
        if hasattr(image_inputs, "keys"):
            image_inputs = {
                k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v
                for k, v in image_inputs.items()
            }
        with torch.no_grad():
            outputs = vision_model(**image_inputs)
        return outputs.last_hidden_state

    @torch.compiler.disable
    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        if vision_tensors is None or not self.config.image_ids:
            return h
        marker, vf = self.config.image_ids[0], vision_tensors
        if vf.dim() == 3:
            vf = vf.unsqueeze(1)
        out = []
        for b in range(h.size(0)):
            hb, seq, k, i = h[b], tokens[b].tolist(), 0, 0
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if k < vf.size(1):
                        hb = torch.cat((hb[:start], vf[b][k][: i - start], hb[i:]), dim=0)[:seqlen]
                        k += 1
                else:
                    i += 1
            out.append(hb)
        return torch.stack(out)

    def encode_images(self, pixel_values):
        if pixel_values is None:
            return None, None
        if hasattr(pixel_values, "keys"):
            img_emb = MiniMindVLM.get_image_embeddings(pixel_values, self.vision_encoder)
            replace_states = self.vision_proj(img_emb)
            qformer_states = self.qformer(img_emb) if self.qformer is not None else None
            return replace_states, qformer_states

        if len(pixel_values.shape) == 6:
            pixel_values = pixel_values.squeeze(2)
        bs, num, _, _, _ = pixel_values.shape
        stack_dim = 1 if bs > 1 else 0
        replace_states = []
        qformer_states = []
        for image_idx in range(num):
            img_emb = MiniMindVLM.get_image_embeddings(pixel_values[:, image_idx, :, :, :], self.vision_encoder)
            replace_states.append(self.vision_proj(img_emb))
            if self.qformer is not None:
                qformer_states.append(self.qformer(img_emb))
        replace_states = torch.stack(replace_states, dim=stack_dim)
        if qformer_states:
            qformer_states = torch.stack(qformer_states, dim=stack_dim)
        else:
            qformer_states = None
        return replace_states, qformer_states

    def merge_visual_context(self, qformer_states):
        if qformer_states is None:
            return None
        if qformer_states.dim() == 4:
            batch_size, num_images, num_queries, hidden_size = qformer_states.shape
            return qformer_states.reshape(batch_size, num_images * num_queries, hidden_size)
        return qformer_states

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        **args,
    ):
        _, seq_length = input_ids.shape
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))
        visual_replace_states, qformer_states = (None, None)
        if pixel_values is not None:
            visual_replace_states, qformer_states = self.encode_images(pixel_values)

        if (
            self.config.vision_fusion_type == "replace"
            and visual_replace_states is not None
            and start_pos == 0
        ):
            hidden_states = self.count_vision_proj(
                tokens=input_ids,
                h=hidden_states,
                vision_tensors=visual_replace_states,
                seqlen=input_ids.shape[1],
            )

        visual_context = self.merge_visual_context(qformer_states)
        position_embeddings = (
            self.model.freqs_cos[start_pos : start_pos + seq_length],
            self.model.freqs_sin[start_pos : start_pos + seq_length],
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            if (
                self.config.vision_fusion_type == "qformer_cross_attn"
                and visual_context is not None
                and self.text_cross_attn_layers is not None
                and not isinstance(self.text_cross_attn_layers[layer_idx], nn.Identity)
            ):
                hidden_states = self.text_cross_attn_layers[layer_idx](hidden_states, visual_context)
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)
        aux_loss = sum(
            [l.mlp.aux_loss for l in self.model.layers if isinstance(l.mlp, MOEFeedForward)],
            hidden_states.new_zeros(1).squeeze(),
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=presents,
            hidden_states=hidden_states,
        )

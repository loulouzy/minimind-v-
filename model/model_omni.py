import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import Siglip2ImageProcessor, Siglip2VisionModel, WhisperModel, WhisperProcessor
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from .model_minimind import *

warnings.filterwarnings("ignore")


def get_omni_arch_suffix(config) -> str:
    parts = []
    if getattr(config, "fusion_type", "replace") == "qformer_cross_attn":
        parts.append("qformer")
    if getattr(config, "use_moe", False):
        parts.append("moe")
    return f"_{'_'.join(parts)}" if parts else ""


class OmniConfig(MiniMindConfig):
    model_type = "minimind-omni"

    def __init__(self, image_special_token="<|image_pad|>", image_ids=[12], audio_special_token="<|audio_pad|>", audio_ids=[16], **kwargs):
        self.image_special_token = image_special_token
        self.image_ids = image_ids
        self.audio_special_token = audio_special_token
        self.audio_ids = audio_ids
        self.image_hidden_size = kwargs.get("image_hidden_size", 768)
        self.audio_hidden_size = kwargs.get("audio_hidden_size", 512)
        self.image_token_len = kwargs.get("image_token_len", 64)
        self.audio_token_len = kwargs.get("audio_token_len", 64)
        self.audio_sample_rate = kwargs.get("audio_sample_rate", 16000)
        self.audio_max_seconds = kwargs.get("audio_max_seconds", 30)
        self.fusion_type = kwargs.get("fusion_type", "replace")
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


class MMAudioProjector(nn.Module):
    def __init__(self, in_dim, out_dim, target_tokens=64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(target_tokens)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        return self.mlp(x)


class CrossAttentionAdapter(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0, eps=1e-6):
        super().__init__()
        self.query_norm = RMSNorm(hidden_size, eps=eps)
        self.key_value_norm = RMSNorm(hidden_size, eps=eps)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, query_states, key_value_states):
        q = self.query_norm(query_states)
        kv = self.key_value_norm(key_value_states)
        attn_output, _ = self.cross_attn(q, kv, kv, need_weights=False)
        return query_states + self.resid_dropout(attn_output)


class QFormerBlock(nn.Module):
    def __init__(self, config: OmniConfig):
        super().__init__()
        self.self_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=config.num_attention_heads, dropout=config.dropout, batch_first=True)
        self.self_dropout = nn.Dropout(config.dropout)
        self.cross_attn = CrossAttentionAdapter(config.hidden_size, config.num_attention_heads, dropout=config.dropout, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = FeedForward(config)

    def forward(self, query_states, source_states):
        q = self.self_norm(query_states)
        self_attn_output, _ = self.self_attn(q, q, q, need_weights=False)
        query_states = query_states + self.self_dropout(self_attn_output)
        query_states = self.cross_attn(query_states, source_states)
        query_states = query_states + self.ffn(self.ffn_norm(query_states))
        return query_states


class MiniMindQFormer(nn.Module):
    def __init__(self, input_dim: int, config: OmniConfig):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, config.qformer_num_queries, config.hidden_size) * 0.02)
        self.input_proj = nn.Linear(input_dim, config.hidden_size)
        self.layers = nn.ModuleList([QFormerBlock(config) for _ in range(config.qformer_num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, source_states):
        query_states = self.query_tokens.expand(source_states.size(0), -1, -1)
        source_states = self.input_proj(source_states)
        for layer in self.layers:
            query_states = layer(query_states, source_states)
        return self.norm(query_states)


class MiniMindOmni(MiniMindForCausalLM):
    config_class = OmniConfig

    def __init__(self, config: OmniConfig = None, vision_model_path="./model/siglip2-base-p16-ve", audio_model_path="./model/whisper-base"):
        self.config = config or OmniConfig()
        super().__init__(self.config)
        self.vision_encoder, self.image_processor = self.__class__.get_vision_model(vision_model_path)
        self.audio_encoder, self.audio_processor = self.__class__.get_audio_model(audio_model_path)
        self.vision_proj = MMVisionProjector(self.config.image_hidden_size, self.config.hidden_size, target_tokens=self.config.image_token_len)
        self.audio_proj = MMAudioProjector(self.config.audio_hidden_size, self.config.hidden_size, target_tokens=self.config.audio_token_len)
        self.image_qformer = None
        self.audio_qformer = None
        self.text_cross_attn_layers = None
        if self.config.fusion_type == "qformer_cross_attn":
            self.image_qformer = MiniMindQFormer(self.config.image_hidden_size, self.config)
            self.audio_qformer = MiniMindQFormer(self.config.audio_hidden_size, self.config)
            every_n = max(1, int(self.config.text_cross_attn_every_n_layers))
            self.text_cross_attn_layers = nn.ModuleList(
                [
                    CrossAttentionAdapter(self.config.hidden_size, self.config.num_attention_heads, dropout=self.config.dropout, eps=self.config.rms_norm_eps)
                    if layer_idx % every_n == 0 else nn.Identity()
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
    def get_audio_model(model_path: str):
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        if not os.path.exists(model_path) and "/" not in model_path:
            return None, None
        model = WhisperModel.from_pretrained(model_path)
        processor = WhisperProcessor.from_pretrained(model_path)
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processor

    @staticmethod
    def image2tensor(image, processor):
        if image.mode in ["RGBA", "LA"]:
            image = image.convert("RGB")
        return processor(images=image, return_tensors="pt")

    @staticmethod
    def audio2tensor(audio_array, sample_rate, processor):
        return processor(audio=audio_array, sampling_rate=sample_rate, return_tensors="pt")

    @staticmethod
    def get_image_embeddings(image_inputs, vision_model):
        if image_inputs is None or vision_model is None:
            return None
        if hasattr(image_inputs, "keys"):
            image_inputs = {k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v for k, v in image_inputs.items()}
        with torch.no_grad():
            outputs = vision_model(**image_inputs)
        return outputs.last_hidden_state

    @staticmethod
    def get_audio_embeddings(audio_inputs, audio_model):
        if audio_inputs is None or audio_model is None:
            return None
        if hasattr(audio_inputs, "keys"):
            audio_inputs = {k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v for k, v in audio_inputs.items()}
        encoder_kwargs = {"input_features": audio_inputs["input_features"]}
        if "attention_mask" in audio_inputs:
            encoder_kwargs["attention_mask"] = audio_inputs["attention_mask"]
        with torch.no_grad():
            outputs = audio_model.encoder(**encoder_kwargs)
        return outputs.last_hidden_state

    def _encode_stacked_inputs(self, inputs, encoder_fn, replace_module, qformer_module):
        if inputs is None:
            return None, None
        if hasattr(inputs, "keys"):
            any_tensor = next(iter(inputs.values()))
            if any_tensor.ndim >= 5 or (any_tensor.ndim == 4 and any_tensor.shape[1] > 1):
                num_items = any_tensor.shape[1]
                replace_states, qformer_states = [], []
                for item_idx in range(num_items):
                    item_inputs = {k: v[:, item_idx] for k, v in inputs.items()}
                    item_emb = encoder_fn(item_inputs)
                    replace_states.append(replace_module(item_emb))
                    if qformer_module is not None:
                        qformer_states.append(qformer_module(item_emb))
                replace_states = torch.stack(replace_states, dim=1)
                qformer_states = torch.stack(qformer_states, dim=1) if qformer_states else None
                return replace_states, qformer_states
            source_emb = encoder_fn(inputs)
            replace_states = replace_module(source_emb)
            qformer_states = qformer_module(source_emb) if qformer_module is not None else None
            return replace_states, qformer_states
        source_emb = encoder_fn(inputs)
        replace_states = replace_module(source_emb)
        qformer_states = qformer_module(source_emb) if qformer_module is not None else None
        return replace_states, qformer_states

    def encode_images(self, pixel_values):
        return self._encode_stacked_inputs(pixel_values, lambda x: MiniMindOmni.get_image_embeddings(x, self.vision_encoder), self.vision_proj, self.image_qformer)

    def encode_audios(self, audio_values):
        return self._encode_stacked_inputs(audio_values, lambda x: MiniMindOmni.get_audio_embeddings(x, self.audio_encoder), self.audio_proj, self.audio_qformer)

    def _normalize_replace_tensors(self, replace_tensors: Dict[int, Optional[torch.Tensor]]):
        normalized = {}
        for marker, tensor in replace_tensors.items():
            if tensor is None:
                continue
            normalized[marker] = tensor.unsqueeze(1) if tensor.dim() == 3 else tensor
        return normalized

    @torch.compiler.disable
    def replace_modal_embeddings(self, tokens, hidden_states, replace_tensors, seqlen=512):
        normalized = self._normalize_replace_tensors(replace_tensors)
        if not normalized:
            return hidden_states
        out = []
        for batch_idx in range(hidden_states.size(0)):
            hb = hidden_states[batch_idx]
            seq = tokens[batch_idx].tolist()
            counters = {marker: 0 for marker in normalized.keys()}
            i = 0
            while i < len(seq):
                marker = seq[i]
                if marker not in normalized:
                    i += 1
                    continue
                start = i
                while i < len(seq) and seq[i] == marker:
                    i += 1
                item_idx = counters[marker]
                marker_values = normalized[marker]
                if item_idx < marker_values.size(1):
                    hb = torch.cat((hb[:start], marker_values[batch_idx][item_idx][: i - start], hb[i:]), dim=0)[:seqlen]
                counters[marker] += 1
            out.append(hb)
        return torch.stack(out)

    def merge_modal_contexts(self, *modal_contexts):
        merged = []
        for context in modal_contexts:
            if context is None:
                continue
            if context.dim() == 4:
                b, n, q, h = context.shape
                context = context.reshape(b, n * q, h)
            merged.append(context)
        if not merged:
            return None
        return torch.cat(merged, dim=1)

    def forward(self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, use_cache: bool = False, logits_to_keep: Union[int, torch.Tensor] = 0, labels: Optional[torch.Tensor] = None, pixel_values: Optional[torch.FloatTensor] = None, audio_values: Optional[Union[torch.FloatTensor, Dict[str, torch.Tensor]]] = None, **args):
        _, seq_length = input_ids.shape
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))
        image_replace, image_context = self.encode_images(pixel_values)
        audio_replace, audio_context = self.encode_audios(audio_values)
        if self.config.fusion_type == "replace" and start_pos == 0:
            hidden_states = self.replace_modal_embeddings(
                tokens=input_ids,
                hidden_states=hidden_states,
                replace_tensors={self.config.image_ids[0]: image_replace, self.config.audio_ids[0]: audio_replace},
                seqlen=input_ids.shape[1],
            )
        multimodal_context = self.merge_modal_contexts(image_context, audio_context)
        position_embeddings = (self.model.freqs_cos[start_pos : start_pos + seq_length], self.model.freqs_sin[start_pos : start_pos + seq_length])
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            hidden_states, present = layer(hidden_states, position_embeddings, past_key_value=past_key_value, use_cache=use_cache, attention_mask=attention_mask)
            if self.config.fusion_type == "qformer_cross_attn" and multimodal_context is not None and self.text_cross_attn_layers is not None and not isinstance(self.text_cross_attn_layers[layer_idx], nn.Identity):
                hidden_states = self.text_cross_attn_layers[layer_idx](hidden_states, multimodal_context)
            presents.append(present)
        hidden_states = self.model.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.model.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=presents, hidden_states=hidden_states)

import io
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    import soundfile as sf
except ImportError:
    sf = None

__package__ = "dataset"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.lm_dataset import post_processing_chat, pre_processing_chat
from model.model_omni import MiniMindOmni

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    old_indices = np.arange(len(audio), dtype=np.float32)
    new_length = max(1, int(round(len(audio) * target_sr / orig_sr)))
    new_indices = np.linspace(0, max(len(audio) - 1, 0), num=new_length, dtype=np.float32)
    return np.interp(new_indices, old_indices, audio).astype(np.float32)


class OmniDataset(Dataset):
    def __init__(
        self,
        parquet_path,
        tokenizer,
        image_preprocess=None,
        audio_preprocess=None,
        max_length=512,
        image_special_token="<|image_pad|>",
        image_token_len=64,
        audio_special_token="<|audio_pad|>",
        audio_token_len=64,
        audio_sample_rate=16000,
        audio_max_seconds=30,
    ):
        super().__init__()
        self.table = pa.Table.from_batches(pq.ParquetFile(parquet_path).iter_batches())
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_preprocess = image_preprocess
        self.audio_preprocess = audio_preprocess
        self.image_special_token = image_special_token * image_token_len
        self.audio_special_token = audio_special_token * audio_token_len
        self.audio_sample_rate = audio_sample_rate
        self.audio_max_seconds = audio_max_seconds
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
        self.columns = set(self.table.column_names)

    def __len__(self):
        return len(self.table)

    def _get_column_value(self, column_name, index):
        if column_name not in self.columns:
            return None
        value = self.table[column_name][index].as_py()
        return value

    def create_chat_prompt(self, conversations):
        messages = []
        for turn in conversations:
            if turn.get("role") == "system":
                content = turn["content"]
            else:
                content = turn["content"].replace("<image>", self.image_special_token).replace("<audio>", self.audio_special_token)
            messages.append({"role": turn["role"], "content": content})
        tools = conversations[0]["functions"] if (conversations and conversations[0]["role"] == "system" and conversations[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools)

    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def _load_images(self, image_bytes):
        if image_bytes is None:
            return None
        if not isinstance(image_bytes, list):
            image_bytes = [image_bytes]
        image_inputs_list = [
            MiniMindOmni.image2tensor(Image.open(io.BytesIO(img)), self.image_preprocess)
            for img in image_bytes
        ]
        return {k: torch.cat([inp[k] for inp in image_inputs_list], dim=0) for k in image_inputs_list[0].keys()}

    def _load_audios(self, audio_bytes):
        if audio_bytes is None:
            return None
        if sf is None:
            raise ImportError("soundfile is required for OmniDataset audio decoding. Please install soundfile.")
        if not isinstance(audio_bytes, list):
            audio_bytes = [audio_bytes]

        audio_inputs_list = []
        max_samples = self.audio_sample_rate * self.audio_max_seconds
        for clip in audio_bytes:
            audio_array, sr = sf.read(io.BytesIO(clip), dtype="float32")
            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=-1)
            if sr != self.audio_sample_rate:
                audio_array = _resample_audio(audio_array, sr, self.audio_sample_rate)
            audio_array = audio_array[:max_samples]
            audio_inputs_list.append(MiniMindOmni.audio2tensor(audio_array, self.audio_sample_rate, self.audio_preprocess))
        return {k: torch.cat([inp[k] for inp in audio_inputs_list], dim=0) for k in audio_inputs_list[0].keys()}

    def __getitem__(self, index):
        conversations = json.loads(self.table["conversations"][index].as_py())
        image_bytes = self._get_column_value("image_bytes", index)
        audio_bytes = self._get_column_value("audio_bytes", index)

        conversations = pre_processing_chat(conversations)
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)

        image_data = self._load_images(image_bytes)
        audio_data = self._load_audios(audio_bytes)
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            image_data,
            audio_data,
        )


def _collate_optional_modality(items):
    if all(item is None for item in items):
        return None
    if any(item is None for item in items):
        raise ValueError("A batch mixes present and missing modalities. Use separate datasets or homogeneous batches.")
    if hasattr(items[0], "keys"):
        return {k: torch.stack([item[k] for item in items]) for k in items[0].keys()}
    return torch.stack(items)


def omni_collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    image_values = _collate_optional_modality([b[2] for b in batch])
    audio_values = _collate_optional_modality([b[3] for b in batch])
    return input_ids, labels, image_values, audio_values

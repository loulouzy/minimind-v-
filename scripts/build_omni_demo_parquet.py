import argparse
import io
import json
import os
import random
import sys
from typing import Dict, Iterable, List

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from datasets import load_dataset

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def audio_example_to_wav_bytes(audio_dict) -> bytes:
    array = audio_dict["array"]
    sample_rate = int(audio_dict["sampling_rate"])
    buffer = io.BytesIO()
    sf.write(buffer, array, sample_rate, format="WAV")
    return buffer.getvalue()


def make_audio_caption_conversation(caption: str) -> str:
    conversations = [
        {"role": "user", "content": "请仔细听这段音频，并描述你听到的内容：\n\n<audio>"},
        {"role": "assistant", "content": caption.strip()},
    ]
    return json.dumps(conversations, ensure_ascii=False)


def make_asr_conversation(transcript: str) -> str:
    conversations = [
        {"role": "user", "content": "请转写这段英语语音：\n\n<audio>"},
        {"role": "assistant", "content": transcript.strip()},
    ]
    return json.dumps(conversations, ensure_ascii=False)


def maybe_shuffle_dataset(dataset, seed: int, buffer_size: int):
    if hasattr(dataset, "shuffle"):
        return dataset.shuffle(seed=seed, buffer_size=buffer_size)
    return dataset


def build_rows_from_audiocaps(dataset: Iterable, max_samples: int) -> List[Dict]:
    rows = []
    for idx, example in enumerate(dataset):
        if max_samples > 0 and idx >= max_samples:
            break
        rows.append(
            {
                "source": "audiocaps",
                "conversations": make_audio_caption_conversation(example["caption"]),
                "image_bytes": None,
                "audio_bytes": audio_example_to_wav_bytes(example["audio"]),
            }
        )
    return rows


def build_rows_from_librispeech(dataset: Iterable, target_hours: float, max_samples: int) -> List[Dict]:
    rows = []
    target_seconds = target_hours * 3600 if target_hours > 0 else None
    accumulated_seconds = 0.0
    for idx, example in enumerate(dataset):
        if max_samples > 0 and idx >= max_samples:
            break
        duration = len(example["audio"]["array"]) / float(example["audio"]["sampling_rate"])
        if target_seconds is not None and accumulated_seconds >= target_seconds:
            break
        rows.append(
            {
                "source": "librispeech",
                "conversations": make_asr_conversation(example["text"]),
                "image_bytes": None,
                "audio_bytes": audio_example_to_wav_bytes(example["audio"]),
            }
        )
        accumulated_seconds += duration
    return rows


def rows_to_parquet(rows: List[Dict], output_path: str):
    table = pa.table(
        {
            "source": [row["source"] for row in rows],
            "conversations": [row["conversations"] for row in rows],
            "image_bytes": [row["image_bytes"] for row in rows],
            "audio_bytes": [row["audio_bytes"] for row in rows],
        }
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pq.write_table(table, output_path)


def load_audiocaps(args):
    dataset = load_dataset(args.audiocaps_repo, split=args.audiocaps_split, streaming=args.streaming)
    return maybe_shuffle_dataset(dataset, args.seed, args.shuffle_buffer_size) if args.streaming else dataset


def load_librispeech(args):
    dataset = load_dataset(args.librispeech_repo, args.librispeech_config, split=args.librispeech_split, streaming=args.streaming)
    return maybe_shuffle_dataset(dataset, args.seed, args.shuffle_buffer_size) if args.streaming else dataset


def main():
    parser = argparse.ArgumentParser(description="Build small demo parquet files for MiniMind-Omni from AudioCaps and LibriSpeech.")
    parser.add_argument("--output_dir", type=str, default="../dataset/demo_omni")
    parser.add_argument("--streaming", action="store_true", help="使用 Hugging Face streaming 模式，避免先完整下载")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle_buffer_size", type=int, default=5000)

    parser.add_argument("--build_audiocaps", action="store_true")
    parser.add_argument("--audiocaps_repo", type=str, default="OpenSound/AudioCaps")
    parser.add_argument("--audiocaps_split", type=str, default="train")
    parser.add_argument("--audiocaps_samples", type=int, default=5000)
    parser.add_argument("--audiocaps_output", type=str, default="audiocaps_demo.parquet")

    parser.add_argument("--build_librispeech", action="store_true")
    parser.add_argument("--librispeech_repo", type=str, default="openslr/librispeech_asr")
    parser.add_argument("--librispeech_config", type=str, default="clean")
    parser.add_argument("--librispeech_split", type=str, default="train.100")
    parser.add_argument("--librispeech_hours", type=float, default=10.0)
    parser.add_argument("--librispeech_max_samples", type=int, default=0)
    parser.add_argument("--librispeech_output", type=str, default="librispeech_demo.parquet")

    parser.add_argument("--build_mix", action="store_true", help="额外生成一个混合 parquet")
    parser.add_argument("--mix_output", type=str, default="omni_audio_demo_mix.parquet")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    all_rows = []

    if args.build_audiocaps:
        audiocaps_rows = build_rows_from_audiocaps(load_audiocaps(args), args.audiocaps_samples)
        audiocaps_path = os.path.join(args.output_dir, args.audiocaps_output)
        rows_to_parquet(audiocaps_rows, audiocaps_path)
        print(f"Saved AudioCaps demo parquet: {audiocaps_path} ({len(audiocaps_rows)} rows)")
        all_rows.extend(audiocaps_rows)

    if args.build_librispeech:
        librispeech_rows = build_rows_from_librispeech(load_librispeech(args), args.librispeech_hours, args.librispeech_max_samples)
        librispeech_path = os.path.join(args.output_dir, args.librispeech_output)
        rows_to_parquet(librispeech_rows, librispeech_path)
        print(f"Saved LibriSpeech demo parquet: {librispeech_path} ({len(librispeech_rows)} rows)")
        all_rows.extend(librispeech_rows)

    if args.build_mix and all_rows:
        random.shuffle(all_rows)
        mix_path = os.path.join(args.output_dir, args.mix_output)
        rows_to_parquet(all_rows, mix_path)
        print(f"Saved mixed demo parquet: {mix_path} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()

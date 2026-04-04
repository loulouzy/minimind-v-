import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.lm_dataset import VLMDataset
from model.model_vlm import VLMConfig
from trainer.trainer_utils import init_vlm_model, setup_seed, vlm_collate_fn


@dataclass
class EvalMetrics:
    name: str
    total_params_m: float
    trainable_params_m: float
    avg_nll: float
    perplexity: float
    tokens_per_sec: float
    samples_per_sec: float
    peak_memory_gb: float
    valid_tokens: int
    elapsed_sec: float


def metrics_to_row(metrics: EvalMetrics):
    return {
        "mode": metrics.name,
        "avg_nll": metrics.avg_nll,
        "perplexity": metrics.perplexity,
        "tokens_per_sec": metrics.tokens_per_sec,
        "samples_per_sec": metrics.samples_per_sec,
        "peak_memory_gb": metrics.peak_memory_gb,
        "total_params_m": metrics.total_params_m,
        "trainable_params_m": metrics.trainable_params_m,
        "valid_tokens": metrics.valid_tokens,
        "elapsed_sec": metrics.elapsed_sec,
    }


def build_config(args, fusion_type):
    return VLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=args.max_seq_len,
        use_moe=bool(args.use_moe),
        vision_fusion_type=fusion_type,
        qformer_num_queries=args.qformer_num_queries,
        qformer_num_layers=args.qformer_num_layers,
        text_cross_attn_every_n_layers=args.text_cross_attn_every_n_layers,
    )


def count_params(model):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return total, trainable


def move_pixel_values(pixel_values, device):
    if isinstance(pixel_values, dict):
        return {k: v.to(device) for k, v in pixel_values.items()}
    return pixel_values.to(device)


def create_loader(args, tokenizer, preprocess, config):
    dataset = VLMDataset(
        parquet_path=args.data_path,
        tokenizer=tokenizer,
        preprocess=preprocess,
        max_length=args.max_seq_len,
        image_special_token=config.image_special_token,
        image_token_len=config.image_token_len,
    )
    if args.max_samples > 0:
        max_samples = min(args.max_samples, len(dataset))
        dataset = Subset(dataset, list(range(max_samples)))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=vlm_collate_fn,
    )


@torch.inference_mode()
def evaluate_model(args, model, loader, name):
    model.eval()
    total_params_m, trainable_params_m = count_params(model)
    total_loss_sum = 0.0
    total_valid_tokens = 0
    total_samples = 0

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(args.device)
        torch.cuda.synchronize(args.device)

    start_time = time.perf_counter()
    for batch_idx, (input_ids, labels, pixel_values) in enumerate(loader):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        pixel_values = move_pixel_values(pixel_values, args.device)

        outputs = model(input_ids, labels=None, pixel_values=pixel_values)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        valid_mask = shift_labels.ne(-100)
        valid_tokens = int(valid_mask.sum().item())
        if valid_tokens == 0:
            continue

        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_loss_sum += float(token_losses.item())
        total_valid_tokens += valid_tokens
        total_samples += int(input_ids.size(0))

        if args.max_batches > 0 and (batch_idx + 1) >= args.max_batches:
            break

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(args.device)
        peak_memory_gb = torch.cuda.max_memory_allocated(args.device) / (1024 ** 3)
    else:
        peak_memory_gb = 0.0

    elapsed_sec = max(time.perf_counter() - start_time, 1e-8)
    avg_nll = total_loss_sum / max(total_valid_tokens, 1)
    perplexity = math.exp(min(avg_nll, 20))
    tokens_per_sec = total_valid_tokens / elapsed_sec
    samples_per_sec = total_samples / elapsed_sec
    return EvalMetrics(
        name=name,
        total_params_m=total_params_m,
        trainable_params_m=trainable_params_m,
        avg_nll=avg_nll,
        perplexity=perplexity,
        tokens_per_sec=tokens_per_sec,
        samples_per_sec=samples_per_sec,
        peak_memory_gb=peak_memory_gb,
        valid_tokens=total_valid_tokens,
        elapsed_sec=elapsed_sec,
    )


def print_summary(results):
    print("\n=== Primary Metric ===")
    print("Primary metric: held-out average NLL per valid target token (lower is better).")
    print("Secondary metrics: perplexity, throughput, peak memory, and parameter count.")
    header = (
        f"{'mode':<22}"
        f"{'NLL':>10}"
        f"{'PPL':>12}"
        f"{'tok/s':>12}"
        f"{'GB':>10}"
        f"{'total(M)':>12}"
        f"{'train(M)':>12}"
    )
    print("\n" + header)
    print("-" * len(header))
    for item in results:
        print(
            f"{item.name:<22}"
            f"{item.avg_nll:>10.4f}"
            f"{item.perplexity:>12.2f}"
            f"{item.tokens_per_sec:>12.1f}"
            f"{item.peak_memory_gb:>10.2f}"
            f"{item.total_params_m:>12.2f}"
            f"{item.trainable_params_m:>12.2f}"
        )
    if len(results) == 2:
        base, other = results
        print("\n=== Delta ===")
        print(f"NLL delta ({other.name} - {base.name}): {other.avg_nll - base.avg_nll:+.4f}")
        print(f"PPL ratio ({other.name} / {base.name}): {other.perplexity / max(base.perplexity, 1e-8):.3f}")
        print(f"Throughput ratio ({other.name} / {base.name}): {other.tokens_per_sec / max(base.tokens_per_sec, 1e-8):.3f}")
        print(f"Peak memory delta ({other.name} - {base.name}): {other.peak_memory_gb - base.peak_memory_gb:+.2f} GB")


def write_csv(csv_path, results):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fieldnames = [
        "mode",
        "avg_nll",
        "perplexity",
        "tokens_per_sec",
        "samples_per_sec",
        "peak_memory_gb",
        "total_params_m",
        "trainable_params_m",
        "valid_tokens",
        "elapsed_sec",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(metrics_to_row(item))


def append_delta_rows(csv_path, results):
    if len(results) != 2:
        return
    base, other = results
    delta_row = {
        "mode": f"delta:{other.name}-{base.name}",
        "avg_nll": other.avg_nll - base.avg_nll,
        "perplexity": other.perplexity / max(base.perplexity, 1e-8),
        "tokens_per_sec": other.tokens_per_sec / max(base.tokens_per_sec, 1e-8),
        "samples_per_sec": other.samples_per_sec / max(base.samples_per_sec, 1e-8),
        "peak_memory_gb": other.peak_memory_gb - base.peak_memory_gb,
        "total_params_m": other.total_params_m - base.total_params_m,
        "trainable_params_m": other.trainable_params_m - base.trainable_params_m,
        "valid_tokens": other.valid_tokens - base.valid_tokens,
        "elapsed_sec": other.elapsed_sec - base.elapsed_sec,
    }
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(delta_row.keys()))
        writer.writerow(delta_row)


def main():
    parser = argparse.ArgumentParser(description="Compare two VLM fusion modes on the same held-out dataset.")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_i2t.parquet", help="验证数据路径")
    parser.add_argument("--batch_size", type=int, default=4, help="评测 batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--max_samples", type=int, default=512, help="最多评测多少个样本，0 表示全量")
    parser.add_argument("--max_batches", type=int, default=0, help="最多评测多少个 batch，0 表示不限制")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--save_dir", type=str, default="../out", help="权重目录")
    parser.add_argument("--tokenizer_path", type=str, default="../model", help="tokenizer 路径")
    parser.add_argument("--vision_model_path", type=str, default="../model/siglip2-base-p16-ve", help="视觉编码器路径")
    parser.add_argument("--replace_weight", type=str, default="sft_vlm", help="replace 模式权重前缀")
    parser.add_argument("--qformer_weight", type=str, default="sft_vlm", help="qformer 模式权重前缀")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--freeze_llm", type=int, default=0, choices=[0, 1, 2], help="仅用于报告 trainable params 时复现训练配置")
    parser.add_argument("--qformer_num_queries", type=int, default=32)
    parser.add_argument("--qformer_num_layers", type=int, default=2)
    parser.add_argument("--text_cross_attn_every_n_layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv_path", type=str, default="./compare_fusion_modes.csv", help="对比结果 CSV 输出路径")
    args = parser.parse_args()

    setup_seed(args.seed)
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    use_amp = args.device.startswith("cuda") and args.dtype != "float32"
    specs = [("replace", args.replace_weight), ("qformer_cross_attn", args.qformer_weight)]
    results = []

    for fusion_type, weight_name in specs:
        config = build_config(args, fusion_type)
        model, tokenizer, preprocess = init_vlm_model(
            config,
            from_weight=weight_name,
            tokenizer_path=args.tokenizer_path,
            vision_model_path=args.vision_model_path,
            save_dir=args.save_dir,
            device=args.device,
            freeze_llm=args.freeze_llm,
        )
        loader = create_loader(args, tokenizer, preprocess, config)
        with torch.autocast(device_type="cuda", dtype=dtype_map[args.dtype], enabled=use_amp):
            results.append(evaluate_model(args, model, loader, fusion_type))
        del model
        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_summary(results)
    write_csv(args.csv_path, results)
    append_delta_rows(args.csv_path, results)
    print(f"\nCSV saved to: {os.path.abspath(args.csv_path)}")


if __name__ == "__main__":
    main()

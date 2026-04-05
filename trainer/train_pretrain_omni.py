import os
import sys
import time
import warnings
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

from dataset.omni_dataset import OmniDataset
from model.model_omni import OmniConfig, get_omni_arch_suffix
from trainer.trainer_utils import (
    Logger,
    SkipBatchSampler,
    get_lr,
    init_distributed_mode,
    init_omni_model,
    is_main_process,
    omni_checkpoint,
    omni_collate_fn,
    setup_seed,
)

warnings.filterwarnings("ignore")


def move_optional_modality(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: v.to(args.device) for k, v in value.items()}
    return value.to(args.device)


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    for step, (input_ids, labels, image_values, audio_values) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        image_values = move_optional_modality(image_values)
        audio_values = move_optional_modality(audio_values)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels, pixel_values=image_values, audio_values=audio_values)
            loss = (res.loss + res.aux_loss) / args.accumulation_steps

        scaler.scale(loss).backward()
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]["lr"]
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                f"loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, "
                f"aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min"
            )
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "logits_loss": current_logits_loss,
                    "aux_loss": current_aux_loss,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min,
                })

        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            arch_suffix = get_omni_arch_suffix(omni_config)
            ckp = f"{args.save_dir}/{args.save_weight}_{omni_config.hidden_size}{arch_suffix}.pth"
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, "_orig_mod", raw_model)
            state_dict = raw_model.state_dict()
            clean_state_dict = {
                key: value
                for key, value in state_dict.items()
                if not key.startswith("vision_encoder.") and not key.startswith("audio_encoder.")
            }
            torch.save({k: v.half().cpu() for k, v in clean_state_dict.items()}, ckp)
            omni_checkpoint(
                omni_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir="../checkpoints",
                scaler=scaler,
            )
            model.train()
            del state_dict, clean_state_dict

        del input_ids, labels, image_values, audio_values, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-Omni Audio/Image Pretrain")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument("--save_weight", type=str, default="pretrain_omni")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--fusion_type", type=str, default="replace", choices=["replace", "qformer_cross_attn"])
    parser.add_argument("--qformer_num_queries", type=int, default=32)
    parser.add_argument("--qformer_num_layers", type=int, default=2)
    parser.add_argument("--text_cross_attn_every_n_layers", type=int, default=1)
    parser.add_argument("--image_token_len", type=int, default=64)
    parser.add_argument("--audio_token_len", type=int, default=64)
    parser.add_argument("--audio_sample_rate", type=int, default=16000)
    parser.add_argument("--audio_max_seconds", type=int, default=30)
    parser.add_argument("--data_path", type=str, required=True, help="支持 image_bytes / audio_bytes 的 parquet 数据")
    parser.add_argument("--from_weight", type=str, default="sft_vlm", help="建议从已训练好的视觉模型权重启动")
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1])
    parser.add_argument("--freeze_llm", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--use_compile", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Omni-Pretrain")
    parser.add_argument("--tokenizer_path", type=str, default="../model")
    parser.add_argument("--vision_model_path", type=str, default="../model/siglip2-base-p16-ve")
    parser.add_argument("--audio_model_path", type=str, default="../model/whisper-base")
    args = parser.parse_args()

    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    os.makedirs(args.save_dir, exist_ok=True)
    omni_config = OmniConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=args.max_seq_len,
        use_moe=bool(args.use_moe),
        fusion_type=args.fusion_type,
        qformer_num_queries=args.qformer_num_queries,
        qformer_num_layers=args.qformer_num_layers,
        text_cross_attn_every_n_layers=args.text_cross_attn_every_n_layers,
        image_token_len=args.image_token_len,
        audio_token_len=args.audio_token_len,
        audio_sample_rate=args.audio_sample_rate,
        audio_max_seconds=args.audio_max_seconds,
    )
    ckp_data = omni_checkpoint(omni_config, weight=args.save_weight, save_dir="../checkpoints") if args.from_resume == 1 else None

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb.init(project=args.wandb_project, name=f"MiniMind-Omni-Pretrain-{args.fusion_type}", id=wandb_id, resume=resume)

    model, tokenizer, image_preprocess, audio_preprocess = init_omni_model(
        omni_config,
        from_weight=args.from_weight,
        tokenizer_path=args.tokenizer_path,
        vision_model_path=args.vision_model_path,
        audio_model_path=args.audio_model_path,
        save_dir=args.save_dir,
        device=args.device,
        freeze_llm=args.freeze_llm,
    )
    train_ds = OmniDataset(
        args.data_path,
        tokenizer,
        image_preprocess=image_preprocess,
        audio_preprocess=audio_preprocess,
        max_length=args.max_seq_len,
        image_special_token=omni_config.image_special_token,
        image_token_len=omni_config.image_token_len,
        audio_special_token=omni_config.audio_special_token,
        audio_token_len=omni_config.audio_token_len,
        audio_sample_rate=omni_config.audio_sample_rate,
        audio_max_seconds=omni_config.audio_max_seconds,
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"], strict=False)
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger("torch.compile enabled")
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=omni_collate_fn)
        if skip > 0:
            Logger(f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始")
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

    if dist.is_initialized():
        dist.destroy_process_group()

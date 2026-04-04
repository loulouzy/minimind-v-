import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from model.model_vlm import MiniMindVLM, VLMConfig, get_vlm_arch_suffix
from dataset.lm_dataset import VLMDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, init_distributed_mode, setup_seed, init_vlm_model, vlm_checkpoint, SkipBatchSampler, vlm_collate_fn

warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    # 记录当前 epoch 开始时间，用于统计耗时
    start_time = time.time()

    # 遍历一个 epoch 内的所有 batch
    # step 从 start_step + 1 开始，通常用于断点续训时接着之前的 step 编号继续
    for step, (input_ids, labels, pixel_values) in enumerate(loader, start=start_step + 1):
        # 将输入文本 token 搬到训练设备（如 GPU）
        input_ids = input_ids.to(args.device)

        # 将标签搬到训练设备
        labels = labels.to(args.device)

        # 将图像输入搬到训练设备
        # pixel_values 可能是 dict，也可能是 tensor，因此这里分别处理
        pixel_values = (
            {k: v.to(args.device) for k, v in pixel_values.items()}
            if isinstance(pixel_values, dict)
            else pixel_values.to(args.device)
        )

        # 根据当前全局训练步数计算学习率
        # 当前全局 step = 当前 epoch 偏移量 + 当前 step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)

        # 将新学习率写入优化器的每个参数组
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 混合精度前向计算上下文（通常是 torch.cuda.amp.autocast）
        with autocast_ctx:
            # 前向传播，得到模型输出
            res = model(input_ids, labels=labels, pixel_values=pixel_values)

            # 总损失 = 主损失 + 辅助损失
            loss = res.loss + res.aux_loss

            # 梯度累积：将 loss 平均到每个累积 step 上
            loss = loss / args.accumulation_steps

        # 使用 GradScaler 缩放 loss 后再反向传播，适配混合精度训练
        scaler.scale(loss).backward()

        # 达到梯度累积步数后，才真正更新一次参数
        if (step + 1) % args.accumulation_steps == 0:
            # 在做梯度裁剪前，先反缩放梯度
            scaler.unscale_(optimizer)

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 执行优化器更新
            scaler.step(optimizer)

            # 更新 scaler 的缩放因子
            scaler.update()

            # 清空梯度，set_to_none=True 能更省显存
            optimizer.zero_grad(set_to_none=True)

        # 按日志间隔打印训练信息，或者在 epoch 最后一步时打印
        if step % args.log_interval == 0 or step == iters - 1:
            # 当前 epoch 已花费时间
            spend_time = time.time() - start_time

            # 因为前面除过 accumulation_steps，这里乘回来得到原始 loss 大小
            current_loss = loss.item() * args.accumulation_steps

            # 辅助损失，若不存在则记为 0
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0

            # 主 loss（logits_loss）= 总 loss - 辅助损失
            current_logits_loss = current_loss - current_aux_loss

            # 当前学习率（取最后一个 param_group 的 lr）
            current_lr = optimizer.param_groups[-1]['lr']

            # 估算本 epoch 剩余时间（分钟）
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60

            # 打印日志
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'loss: {current_loss:.4f}, '
                f'logits_loss: {current_logits_loss:.4f}, '
                f'aux_loss: {current_aux_loss:.4f}, '
                f'lr: {current_lr:.8f}, '
                f'epoch_time: {eta_min:.1f}min'
            )

            # 如果启用了 wandb，则同步记录日志
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "logits_loss": current_logits_loss,
                    "aux_loss": current_aux_loss,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min
                })

        # 按保存间隔保存模型，或者在 epoch 最后一步保存
        # 并且只在主进程保存，避免分布式训练下重复保存
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            # 切换到评估模式，避免保存过程受 dropout 等影响
            model.eval()

            # 如果用了 MoE，则给文件名加上 _moe 后缀
            arch_suffix = get_vlm_arch_suffix(vlm_config)
            ckp = f'{args.save_dir}/{args.save_weight}_{vlm_config.hidden_size}{arch_suffix}.pth'

            # 如果模型被 DDP 包装，取出原始模型
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model

            # 如果模型经过 torch.compile 等包装，进一步取出原始模块
            raw_model = getattr(raw_model, '_orig_mod', raw_model)

            # 获取完整参数字典
            state_dict = raw_model.state_dict()

            # 过滤掉 vision_encoder 的参数，不保存视觉编码器部分
            clean_state_dict = {
                key: value for key, value in state_dict.items()
                if not key.startswith('vision_encoder.')
            }

            # 将参数转为半精度并搬到 CPU，减少保存体积和显存占用
            clean_state_dict = {k: v.half().cpu() for k, v in clean_state_dict.items()}

            # 保存筛选后的模型权重
            torch.save(clean_state_dict, ckp)

            # 额外保存训练检查点（包含优化器、scaler、epoch、step 等）
            vlm_checkpoint(
                vlm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=step,
                wandb=wandb,
                save_dir='../checkpoints',
                scaler=scaler
            )

            # 保存结束后切回训练模式
            model.train()

            # 显式删除临时变量，帮助释放内存
            del state_dict, clean_state_dict

        # 删除当前 batch 的中间变量，减少显存占用
        del input_ids, labels, pixel_values, res, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind-V Pretrain")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain_vlm', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=360, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--vision_fusion_type', default='qformer_cross_attn', type=str, choices=['replace', 'qformer_cross_attn'], help="视觉融合方式")
    parser.add_argument('--qformer_num_queries', default=32, type=int, help="Q-Former query 数量")
    parser.add_argument('--qformer_num_layers', default=2, type=int, help="Q-Former 层数")
    parser.add_argument('--text_cross_attn_every_n_layers', default=1, type=int, help="每隔多少层插入一次 text cross-attention")
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_i2t.parquet", help="训练数据路径")
    parser.add_argument('--from_weight', default='llm', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--freeze_llm', default=1, type=int, choices=[0, 1, 2], help="冻结策略（0=完全可训练，1=冻结+解冻第0层，2=完全冻结仅训练proj）")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-V-Pretrain", help="wandb项目名")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    vlm_config = VLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len,
        use_moe=bool(args.use_moe),
        vision_fusion_type=args.vision_fusion_type,
        qformer_num_queries=args.qformer_num_queries,
        qformer_num_layers=args.qformer_num_layers,
        text_cross_attn_every_n_layers=args.text_cross_attn_every_n_layers,
    )
    ckp_data = vlm_checkpoint(vlm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-V-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer, preprocess = init_vlm_model(vlm_config, from_weight=args.from_weight, device=args.device, freeze_llm=args.freeze_llm)
    train_ds = VLMDataset(args.data_path, tokenizer, preprocess=preprocess, image_special_token=vlm_config.image_special_token, image_token_len=vlm_config.image_token_len, max_length=vlm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=vlm_collate_fn)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()

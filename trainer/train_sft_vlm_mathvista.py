"""
使用 AI4Math/MathVista 数据集进行 VLM SFT（监督微调）- 原生 PyTorch 版本

主要区别（相比预训练）：
- 不冻结 LLM 参数（全参数微调）
- 学习率更低（2e-5 vs 1e-4）
- 基于预训练权重开始
- 使用 <reasoning> 和 <answer> 格式

使用方法：
    python trainer/train_sft_vlm_mathvista.py \\
        --epochs 2 \\
        --batch_size 4 \\
        --learning_rate 2e-5 \\
        --max_samples 300
"""
import os
import sys
import time

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import warnings
import torch
from contextlib import nullcontext
from torch import optim, nn
from torch.utils.data import DataLoader
from model.model_vlm import VLMConfig
from dataset.lm_dataset import MathVistaDataset
from trainer.trainer_utils import (
    Logger,
    is_main_process,
    setup_seed,
    init_vlm_model,
    vlm_checkpoint,
    get_lr,
)

warnings.filterwarnings('ignore')


# ==================== 训练函数 ====================
def train_epoch(epoch, model, loader, optimizer, scaler, autocast_ctx, args, vlm_config, wandb=None):
    """训练一个 epoch"""
    model.train()
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    iters = len(loader)
    
    for step, (X, Y, loss_mask, pixel_values) in enumerate(loader, start=1):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        pixel_values = pixel_values.to(args.device)
        
        # 更新学习率
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播
        with autocast_ctx:
            res = model(X, pixel_values=pixel_values)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            loss = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)
            if hasattr(res, 'aux_loss') and res.aux_loss is not None:
                loss = loss + res.aux_loss
            loss = loss / args.accumulation_steps

        # 反向传播
        scaler.scale(loss).backward()

        # 梯度累积
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 日志
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / step * iters // 60 - spend_time // 60
            
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min')
            
            if wandb:
                wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})

        # 保存检查点
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            vlm_checkpoint(vlm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                          epoch=epoch, step=step, wandb=wandb, save_dir=args.save_dir, scaler=scaler)
            model.train()

        # 清理内存
        del X, Y, loss_mask, pixel_values, res, loss
        torch.cuda.empty_cache()


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="MiniMind-V SFT with MathVista (PyTorch)")
    
    # 数据相关参数
    parser.add_argument("--split", type=str, default="testmini", help="数据集分割 (testmini, test)")
    parser.add_argument("--start_idx", type=int, default=500, help="数据起始索引")
    parser.add_argument("--end_idx", type=int, default=800, help="数据结束索引")
    
    # 模型相关参数
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument('--vision_fusion_type', default='replace', type=str, choices=['replace', 'qformer_cross_attn'], help="视觉融合方式")
    parser.add_argument('--qformer_num_queries', default=32, type=int, help="Q-Former query 数量")
    parser.add_argument('--qformer_num_layers', default=2, type=int, help="Q-Former 层数")
    parser.add_argument('--text_cross_attn_every_n_layers', default=1, type=int, help="每隔多少层插入一次 text cross-attention")
    
    # 训练相关参数（SFT 特有配置）
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="初始学习率（SFT 使用更低的学习率）")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
    
    # 保存和日志相关参数
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='sft_vlm_mathvista', type=str, help="保存权重的前缀名")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    
    # 其他参数（SFT 特有配置）
    parser.add_argument('--from_weight', default='pretrain_vlm', type=str, help="基于哪个权重训练（SFT 基于预训练权重）")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--use_wandb", type=int, default=1, choices=[0, 1], help="是否使用wandb (默认开启)")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-V-SFT-MathVista", help="wandb项目名")
    
    args = parser.parse_args()

    # ========== 1. 初始化随机种子 ==========
    setup_seed(42)

    # ========== 2. 配置目录、模型参数 ==========
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    save_dir = os.path.join(project_root, args.save_dir.replace('../', ''))
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = save_dir
    
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

    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

    # ========== 4. 配置 wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_run_name = f"MiniMind-V-SFT-MathVista-Epoch-{args.epochs}"
        wandb.init(project=args.wandb_project, name=wandb_run_name)

    # ========== 5. 初始化模型（SFT 不冻结 LLM） ==========
    model, tokenizer, preprocess = init_vlm_model(
        vlm_config,
        from_weight=args.from_weight,
        tokenizer_path=os.path.join(project_root, 'model'),
        vision_model_path=os.path.join(project_root, 'model/vision_model/clip-vit-base-patch16'),
        save_dir=os.path.join(project_root, 'out'),
        device=args.device,
        freeze_llm=False  # SFT 不冻结 LLM，全参数微调
    )

    # ========== 6. 加载 MathVista 数据集 ==========
    Logger("=" * 50)
    Logger("加载 AI4Math/MathVista 数据集 (SFT)")
    Logger("=" * 50)
    
    train_ds = MathVistaDataset(
        tokenizer=tokenizer,
        preprocess=preprocess,
        image_special_token=vlm_config.image_special_token,
        max_length=vlm_config.max_seq_len,
        split=args.split,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        mode='sft',
    )

    # ========== 7. 创建数据加载器和优化器 ==========
    loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    # SFT 全参数微调，不过滤参数
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=0.01,  # 添加权重衰减
        betas=(0.9, 0.95)   # 优化 beta 参数
    )

    # ========== 8. 从检查点恢复（如果需要）==========
    start_epoch = 0
    if args.from_resume == 1:
        ckp_data = vlm_checkpoint(vlm_config, weight=args.save_weight, save_dir=args.save_dir)
        if ckp_data:
            model.load_state_dict(ckp_data['model'], strict=False)
            optimizer.load_state_dict(ckp_data['optimizer'])
            if 'scaler' in ckp_data:
                scaler.load_state_dict(ckp_data['scaler'])
            start_epoch = ckp_data.get('epoch', 0)
            Logger(f"从检查点恢复，从 epoch {start_epoch + 1} 开始")

    # ========== 9. 开始训练 ==========
    Logger("=" * 50)
    Logger("开始 SFT 训练 (使用 AI4Math/MathVista 数据集)")
    Logger("=" * 50)
    Logger(f"训练样本数: {len(train_ds)}")
    Logger(f"批次大小: {args.batch_size}")
    Logger(f"梯度累积步数: {args.accumulation_steps}")
    Logger(f"有效批次大小: {args.batch_size * args.accumulation_steps}")
    Logger(f"学习率: {args.learning_rate}")
    Logger(f"全参数微调: True (freeze_llm=False)")
    
    for epoch in range(start_epoch, args.epochs):
        train_epoch(epoch, model, loader, optimizer, scaler, autocast_ctx, args, vlm_config, wandb)
    
    # ========== 10. 保存最终模型 ==========
    Logger("SFT 训练完成！")
    model.eval()
    vlm_checkpoint(vlm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                  epoch=args.epochs - 1, step=len(loader), wandb=wandb, save_dir=args.save_dir, scaler=scaler)
    Logger(f"最终模型已保存到: {args.save_dir}")


if __name__ == "__main__":
    main()

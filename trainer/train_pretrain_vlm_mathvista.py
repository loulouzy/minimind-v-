
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
from dataset.lm_dataset_math import MathVistaDataset
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
        """
        数据说明:
        - X: 输入 token 序列 [batch_size, seq_len - 1]（去掉最后一个 token）
        - Y: 目标 token 序列 [batch_size, seq_len - 1]（去掉第一个 token，用于预测下一个 token）
        - loss_mask: 损失掩码，只对 assistant 回复部分计算损失，1 表示计算，0 表示忽略
        - pixel_values: 图像张量 [batch_size, num_images, channels, height, width]
        """
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        pixel_values = pixel_values.to(args.device)
        
        # 动态学习率调整
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            # ==== 模型前向传播 ====
            # 1. vision_encoder 编码图像 → 图像特征
            # 2. vision_proj 将图像特征投影到 LLM 维度
            # 3. 将图像特征嵌入到文本序列的对应位置
            # 4. LLM 处理融合后的序列，输出预测 logits
            res = model(X, pixel_values=pixel_values)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            # 只计算 mask=1 位置的平均损失（即 assistant 回复部分）
            loss = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)
            
            # 添加 MoE 辅助损失（用于负载均衡，防止部分专家过载）
            if hasattr(res, 'aux_loss') and res.aux_loss is not None:
                loss = loss + res.aux_loss
            
            # 梯度累积：将损失除以累积步数
            loss = loss / args.accumulation_steps

        # scaler.scale() 会将损失乘以缩放因子，防止 float16 梯度下溢
        scaler.scale(loss).backward()


        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 恢复真实梯度值
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪，防止爆炸
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps  # 恢复真实损失值
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / step * iters // 60 - spend_time // 60
            
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}) loss:{current_loss:.6f} lr:{current_lr:.12f} epoch_Time:{eta_min}min')
            
            if wandb:
                wandb.log({"loss": current_loss, "lr": current_lr, "epoch_Time": eta_min})

        # 保存模型检查点（包含优化器状态，用于断点续训）
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            vlm_checkpoint(vlm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                          epoch=epoch, step=step, wandb=wandb, save_dir=args.save_dir, scaler=scaler)
            model.train()

        # 清理显存，防止碎片化
        del X, Y, loss_mask, pixel_values, res, loss
        torch.cuda.empty_cache()


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="MiniMind-V Pretrain with MathVista (PyTorch)")
    
    # ---------- 数据相关参数 ----------
    parser.add_argument("--split", type=str, default="testmini", help="数据集分割 (testmini, test)")
    parser.add_argument("--start_idx", type=int, default=0, help="数据起始索引")
    parser.add_argument("--end_idx", type=int, default=500, help="数据结束索引")
    
    # ---------- 模型结构参数 ----------
    parser.add_argument('--hidden_size', default=768, type=int, help="LLM 隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="LLM Transformer 层数")
    parser.add_argument('--max_seq_len', default=360, type=int, help="训练的最大序列长度（包含图像 token）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用 MoE 架构")
    
    # ---------- 训练超参数 ----------
    parser.add_argument("--epochs", type=int, default=4, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="每个设备的 batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率（余弦退火调整）")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值，防止梯度爆炸")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数，有效 batch = batch_size * accumulation_steps")
    
    # ---------- 保存和日志 ----------
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain_vlm_mathvista', type=str, help="保存权重的前缀名")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔（步数）")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔（步数）")
    
    # ---------- 模型加载和训练策略 ----------
    parser.add_argument('--from_weight', default='llm', type=str, help="基于哪个预训练权重初始化")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测并续训")
    parser.add_argument('--freeze_llm', default=0, type=int, choices=[0, 1], help="是否冻结 LLM 参数（预训练阶段建议全参数训练）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"], help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载并行线程数")
    parser.add_argument("--use_wandb", type=int, default=1, choices=[0, 1], help="是否使用 wandb/swanlab 记录日志 (默认开启)")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-V-MathVista", help="wandb 项目名称")
    parser.add_argument("--data_path", type=str, default="../MathVista/data", help="训练数据路径")
    args = parser.parse_args()

    # 第 1 部分：初始化随机种子
    setup_seed(42)

    # 第 2 部分：配置目录、模型参数
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    save_dir = os.path.join(project_root, args.save_dir.replace('../', ''))
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = save_dir
    
    vlm_config = VLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len,
        use_moe=bool(args.use_moe)
    )

    # 第 3 部分：设置混合精度训练
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

    # 第 4 部分：配置 wandb/swanlab 日志记录
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_run_name = f"MiniMind-V-MathVista-Epoch-{args.epochs}"
        wandb.init(project=args.wandb_project, name=wandb_run_name)

    # 第 5 部分：初始化模型
    model, tokenizer, preprocess = init_vlm_model(
        vlm_config,
        from_weight=args.from_weight,
        tokenizer_path=os.path.join(project_root, 'model'),
        vision_model_path=os.path.join(project_root, 'model/vision_model/clip-vit-base-patch16'),
        save_dir=os.path.join(project_root, 'out'),
        device=args.device,
        freeze_llm=bool(args.freeze_llm)
    )

    # 第 6 部分：加载 MathVista 数据集
    Logger("=" * 50)
    Logger("加载 AI4Math/MathVista 数据集")
    Logger("=" * 50)
    
    train_ds = MathVistaDataset(
        tokenizer=tokenizer,
        preprocess=preprocess,
        image_special_token=vlm_config.image_special_token,
        max_length=vlm_config.max_seq_len,
        split=args.split,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        mode='pretrain',
    )

    # 第 7 部分：创建数据加载器和优化器
    loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True
    )
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.learning_rate,
        weight_decay=0.01,  # 添加权重衰减
        betas=(0.9, 0.95)   # 优化 beta 参数
    )

    # 第 8 部分：从检查点恢复状态
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

    # 第 9 部分：开始训练循环
    Logger("=" * 50)
    Logger("开始训练 (使用 AI4Math/MathVista 数据集)")
    Logger("=" * 50)
    Logger(f"训练样本数: {len(train_ds)}")
    Logger(f"批次大小: {args.batch_size}")
    Logger(f"梯度累积步数: {args.accumulation_steps}")
    Logger(f"有效批次大小: {args.batch_size * args.accumulation_steps}")
    Logger(f"冻结 LLM: {bool(args.freeze_llm)}")
    
    for epoch in range(start_epoch, args.epochs):
        train_epoch(epoch, model, loader, optimizer, scaler, autocast_ctx, args, vlm_config, wandb)
    
    # 第 10 部分：保存最终模型
    Logger("训练完成！")
    model.eval()
    vlm_checkpoint(vlm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                  epoch=args.epochs - 1, step=len(loader), wandb=wandb, save_dir=args.save_dir, scaler=scaler)
    Logger(f"最终模型已保存到: {args.save_dir}")


if __name__ == "__main__":
    main()

"""
使用 GRPO (Group Relative Policy Optimization) 进行 VLM 强化学习训练
基于 MathVista 数据集，使用原生 PyTorch 实现

参考：
- DeepSeek-R1 论文
- https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2_5_7B_VL_GRPO.ipynb

奖励函数：
1. 正确性奖励：答案是否正确（精确匹配或数值匹配）
2. 长度奖励：惩罚过长的响应，鼓励简洁

使用方法：
    python trainer/train_grpo_vlm_mathvista.py \\
        --epochs 1 \\
        --batch_size 2 \\
        --learning_rate 5e-7 \\
        --from_weight sft_vlm_mathvista \\
        --max_samples 100
"""
import os
import sys
import re
import time

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import warnings
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from model.model_vlm import VLMConfig
from dataset.lm_dataset import MathVistaDataset
from trainer.trainer_utils import (
    Logger,
    is_main_process,
    setup_seed,
    init_vlm_model,
    get_lr,
    GRPOCollator,
    GRPOTrainerBase,
)

warnings.filterwarnings('ignore')


# ==================== 奖励函数 ====================
def extract_json_answer(response: str) -> str:
    """从响应中提取 JSON 答案，失败返回 None"""
    if not response:
        return None
    match = re.search(r'"answer"\s*:\s*"([^"]*)"', response)
    return match.group(1).strip() if match else None


def compute_reward(response: str, ground_truth: str) -> float:
    """
    计算奖励 = 格式奖励 + 准确性奖励
    
    - 格式奖励: 输出正确 JSON 格式 +1，否则 0
    - 准确性奖励: 答案正确 +2，否则 0
    """
    pred = extract_json_answer(response)
    
    # 格式奖励
    format_r = 1.0 if pred is not None else 0.0
    
    # 准确性奖励
    accuracy_r = 0.0
    if pred is not None and ground_truth:
        if pred.strip().lower() == str(ground_truth).strip().lower():
            accuracy_r = 2.0
    
    return format_r + accuracy_r


# ==================== GRPO Trainer ====================
class GRPOTrainer(GRPOTrainerBase):
    """GRPO 训练器（MathVista 数学问答任务）"""
    
    def train_step(self, batch, answers):
        """执行一个 GRPO 训练步骤"""
        prompt_ids = batch['prompt_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        pixel_values = batch['pixel_values'].to(self.device)
        
        batch_size = prompt_ids.size(0)
        prompt_len = prompt_ids.size(1)
        
        # 1. 生成多个响应
        generated_ids, generated_attention_mask, response_texts = self.generate_responses(
            prompt_ids, attention_mask, pixel_values
        )
        
        # 2. 计算奖励
        rewards = []
        for i, response in enumerate(response_texts):
            answer_idx = i // self.num_generations
            answer = answers[answer_idx] if answers[answer_idx] else None
            reward = compute_reward(response, answer)
            rewards.append(reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        
        # 3. 计算组内相对优势（GRPO 核心）
        rewards_grouped = rewards.view(batch_size, self.num_generations)
        mean_rewards = rewards_grouped.mean(dim=1, keepdim=True)
        std_rewards = rewards_grouped.std(dim=1, keepdim=True) + 1e-8
        advantages = (rewards_grouped - mean_rewards) / std_rewards
        advantages = advantages.view(-1).detach()
        
        # 4. 准备训练数据
        expanded_pixel_values = pixel_values.repeat_interleave(self.num_generations, dim=0)
        # 使用生成时产生的 attention_mask，正确标记 pad 位置
        expanded_attention_mask = generated_attention_mask
        
        # 创建 prompt mask（标记 prompt 部分）
        prompt_mask = torch.zeros_like(generated_ids)
        prompt_mask[:, :prompt_len] = 1
        
        # 5. 计算 π_old 的 per-token log prob（生成时的策略，固定不更新）
        # 参考 TRL: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py
        with torch.no_grad():
            _, old_token_log_probs, response_mask = self.compute_log_probs(
                self.model, generated_ids, expanded_attention_mask,
                expanded_pixel_values, generated_ids, prompt_mask
            )
        
        # 6. 计算 π_ref 的 per-token log prob（参考策略，用于 KL 惩罚）
        with torch.no_grad():
            _, ref_token_log_probs, _ = self.compute_log_probs(
                self.ref_model, generated_ids, expanded_attention_mask,
                expanded_pixel_values, generated_ids, prompt_mask
            )
        
        # advantages 扩展到 token 维度 [batch] -> [batch, 1]
        token_advantages = advantages.unsqueeze(-1)
        
        # 7. 在同一批数据上做多次梯度更新，让 ratio 逐渐偏离 1
        num_inner_updates = 4
        for _ in range(num_inner_updates):
            # 计算 π_θ 的 per-token log prob
            _, token_log_probs, _ = self.compute_log_probs(
                self.model, generated_ids, expanded_attention_mask,
                expanded_pixel_values, generated_ids, prompt_mask
            )
            
            # ==== Per-token ratio（TRL 风格）====
            log_ratio = token_log_probs - old_token_log_probs
            ratio = torch.exp(log_ratio.clamp(-10, 10))
            
            # ==== GRPO Loss: -min(ratio * A, clip(ratio) * A) ====
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
            per_token_loss1 = ratio * token_advantages
            per_token_loss2 = clipped_ratio * token_advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
            
            # ==== Per-token KL 惩罚（相对于参考模型）====
            per_token_kl = torch.zeros_like(per_token_loss)
            if self.beta > 0:
                ref_log_ratio = token_log_probs - ref_token_log_probs
                ref_ratio = torch.exp(ref_log_ratio.clamp(-10, 10))
                per_token_kl = (ref_ratio - 1) - ref_log_ratio
            
            # 加入 KL 惩罚
            per_token_loss = per_token_loss + self.beta * per_token_kl
            
            # ==== 计算 masked loss（只对 response 部分）====
            masked_loss = (per_token_loss * response_mask).sum(dim=-1)
            seq_lengths = response_mask.sum(dim=-1).clamp(min=1)
            loss = (masked_loss / seq_lengths).mean()
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        # 计算指标
        avg_ratio = ((ratio * response_mask).sum() / response_mask.sum().clamp(min=1)).item()
        avg_kl = ((per_token_kl * response_mask).sum() / response_mask.sum().clamp(min=1)).item()
        
        metrics = {
            'loss': loss.item(),
            'kl_div': avg_kl,
            'mean_reward': rewards.mean().item(),
            'max_reward': rewards.max().item(),
            'min_reward': rewards.min().item(),
            'ratio': avg_ratio,
        }
        
        return loss, metrics


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(description="MiniMind-V GRPO with MathVista (PyTorch)")
    
    # 数据相关参数
    parser.add_argument("--split", type=str, default="testmini", help="数据集分割")
    parser.add_argument("--start_idx", type=int, default=800, help="数据起始索引")
    parser.add_argument("--end_idx", type=int, default=1000, help="数据结束索引")
    
    # 模型相关参数
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=640, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    
    # GRPO 相关参数
    parser.add_argument("--num_generations", type=int, default=4, help="每个prompt生成的响应数量")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    parser.add_argument("--beta", type=float, default=0.04, help="KL散度系数")
    parser.add_argument("--clip_range", type=float, default=0.2, help="PPO clip范围")
    
    # 训练相关参数
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size（GRPO实际计算量=batch_size*num_generations）")
    parser.add_argument("--learning_rate", type=float, default=5e-7, help="学习率")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    
    # 保存和日志
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument('--save_weight', default='grpo_vlm_mathvista', type=str)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=50)
    
    # 其他参数
    parser.add_argument('--from_weight', default='sft_vlm_mathvista', type=str)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_wandb", type=int, default=1, choices=[0, 1], help="是否使用 wandb (默认开启)")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-V-GRPO")
    
    args = parser.parse_args()

    # ========== 1. 初始化 ==========
    setup_seed(42)
    
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

    # ========== 2. 初始化模型 ==========
    Logger("初始化策略模型...")
    model, tokenizer, preprocess = init_vlm_model(
        vlm_config,
        from_weight=args.from_weight,
        tokenizer_path=os.path.join(project_root, 'model'),
        vision_model_path=os.path.join(project_root, 'model/vision_model/clip-vit-base-patch16'),
        save_dir=os.path.join(project_root, 'out'),
        device=args.device,
        freeze_llm=False
    )
    
    Logger("初始化参考模型...")
    ref_model, _, _ = init_vlm_model(
        vlm_config,
        from_weight=args.from_weight,
        tokenizer_path=os.path.join(project_root, 'model'),
        vision_model_path=os.path.join(project_root, 'model/vision_model/clip-vit-base-patch16'),
        save_dir=os.path.join(project_root, 'out'),
        device=args.device,
        freeze_llm=False
    )
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    # ========== 3. 准备数据集 ==========
    Logger("=" * 50)
    Logger("加载 AI4Math/MathVista 数据集 (GRPO)")
    Logger("=" * 50)
    
    train_ds = MathVistaDataset(
        tokenizer=tokenizer,
        preprocess=preprocess,
        image_special_token=vlm_config.image_special_token,
        max_length=vlm_config.max_seq_len,
        split=args.split,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        mode='grpo',
    )
    
    data_collator = GRPOCollator(pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ========== 4. 配置 wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb.init(project=args.wandb_project, name=f"GRPO-MathVista-Epoch-{args.epochs}")

    # ========== 5. 初始化优化器和训练器 ==========
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
    )
    
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        device=args.device,
        num_generations=args.num_generations,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        beta=args.beta,
        clip_range=args.clip_range,
    )

    # ========== 6. 开始训练 ==========
    Logger("=" * 50)
    Logger("开始 GRPO 训练")
    Logger("=" * 50)
    Logger(f"训练样本数: {len(train_ds)}")
    Logger(f"批次大小: {args.batch_size}")
    Logger(f"每prompt生成数量: {args.num_generations}")
    Logger(f"学习率: {args.learning_rate}")
    
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_reward = 0.0
        
        for step, batch in enumerate(train_loader):
            answers = batch.pop('answers')
            
            loss, metrics = trainer.train_step(batch, answers)
            
            epoch_loss += metrics['loss']
            epoch_reward += metrics['mean_reward']
            global_step += 1
            
            # 更新学习率
            lr = get_lr(global_step, args.epochs * len(train_loader), args.learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # 日志
            if step % args.log_interval == 0:
                Logger(f"Epoch [{epoch+1}/{args.epochs}] Step [{step}/{len(train_loader)}] "
                       f"Loss: {metrics['loss']:.4f} "
                       f"Reward: {metrics['mean_reward']:.4f} "
                       f"KL: {metrics['kl_div']:.4f} "
                       f"Ratio: {metrics['ratio']:.4f}")
                
                if wandb:
                    wandb.log({
                        "loss": metrics['loss'],
                        "mean_reward": metrics['mean_reward'],
                        "kl_div": metrics['kl_div'],
                        "learning_rate": lr,
                    })
            
            # 保存检查点
            if step > 0 and step % args.save_interval == 0:
                save_checkpoint(model, vlm_config, args)
        
        avg_loss = epoch_loss / len(train_loader)
        avg_reward = epoch_reward / len(train_loader)
        Logger(f"Epoch [{epoch+1}/{args.epochs}] 完成 - Avg Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.4f}")

    # ========== 7. 保存最终模型 ==========
    save_checkpoint(model, vlm_config, args)
    Logger("GRPO 训练完成！")


def save_checkpoint(model, vlm_config, args):
    """保存检查点"""
    model.eval()
    moe_suffix = '_moe' if vlm_config.use_moe else ''
    ckp = f'{args.save_dir}/{args.save_weight}_{vlm_config.hidden_size}{moe_suffix}.pth'
    
    state_dict = model.state_dict()
    # 保存完整模型（包括 vision_encoder）
    clean_state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
    torch.save(clean_state_dict, ckp)
    Logger(f"模型已保存到: {ckp}")
    model.train()


if __name__ == "__main__":
    main()

# MiniMind-V 魔改记录

更新时间：2026-04-03

## 原项目与作者

本仓库改动基于原作者 **jingyaogong** 的 **MiniMind-V** 项目。

- 原项目地址：https://github.com/jingyaogong/minimind-v
- 原作者主页：https://github.com/jingyaogong

以下内容主要记录在原项目基础上的实验性修改，不替代原作者项目说明。

## 当前状态

项目已在保留原始 `replace` 视觉融合路径的前提下，新增一条可选的 `Q-Former + text cross-attention` 路径。

默认行为不变：

- `vision_fusion_type=replace` 时，继续使用原有的“图像投影后直接替换文本占位 token embedding”方案。
- `vision_fusion_type=qformer_cross_attn` 时，启用新的 Q-Former 和文本 cross-attention 路径。

## 新增架构

新增模块位于 [model/model_vlm.py](/e:/VLM_LLM/from-minimind-to-more-main/minimind-v/model/model_vlm.py)：

- `MiniMindQFormer`
- `QFormerBlock`
- `CrossAttentionAdapter`

新路径的数据流如下：

1. 图像先经过 `vision_encoder` 得到视觉 token。
2. `Q-Former` 使用可学习 query 从视觉 token 中抽取固定数量的视觉查询向量。
3. 在语言模型每隔若干层插入一次 `text cross-attention`，让文本隐藏状态读取这些视觉查询向量。
4. 原始 `replace` 路径仍然保留，便于对比实验和兼容旧权重。

## 新增参数

以下脚本已经支持新架构开关：

- `trainer/train_pretrain_vlm.py`
- `trainer/train_sft_vlm.py`
- `trainer/train_pretrain_vlm_mathvista.py`
- `trainer/train_sft_vlm_mathvista.py`
- `trainer/train_grpo_vlm.py`
- `trainer/train_grpo_vlm_mathvista.py`
- `eval_vlm.py`

新增命令行参数：

```bash
--vision_fusion_type replace|qformer_cross_attn
--qformer_num_queries 32
--qformer_num_layers 2
--text_cross_attn_every_n_layers 1
```

## 权重命名

为了保留旧权重并避免冲突，新增架构后缀规则：

- 原始路径：`sft_vlm_768.pth`
- Q-Former 路径：`sft_vlm_768_qformer.pth`
- MoE 路径：`sft_vlm_768_moe.pth`
- Q-Former + MoE 路径：`sft_vlm_768_qformer_moe.pth`

断点续训文件也会使用同样的后缀规则。

## 使用示例

### 1. 预训练 Q-Former 版本

```bash
python trainer/train_pretrain_vlm.py ^
  --vision_fusion_type qformer_cross_attn ^
  --qformer_num_queries 32 ^
  --qformer_num_layers 2 ^
  --text_cross_attn_every_n_layers 1
```

### 2. SFT Q-Former 版本

```bash
python trainer/train_sft_vlm.py ^
  --vision_fusion_type qformer_cross_attn ^
  --qformer_num_queries 32 ^
  --qformer_num_layers 2 ^
  --text_cross_attn_every_n_layers 1 ^
  --from_weight pretrain_vlm
```

### 3. 评测 Q-Former 版本

```bash
python eval_vlm.py ^
  --load_from model ^
  --weight sft_vlm ^
  --vision_fusion_type qformer_cross_attn ^
  --qformer_num_queries 32 ^
  --qformer_num_layers 2 ^
  --text_cross_attn_every_n_layers 1
```

## 说明

- 这次改动是“新增可选分支”，不是覆盖原实现。
- 旧的 `replace` 路径和旧权重命名逻辑仍然可用。
- `Q-Former + text cross-attention` 版本目前主要用于架构实验和对比。

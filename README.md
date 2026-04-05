# MiniMind-V 魔改记录

更新时间：2026-04-05

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

## 实验对比结果

在同一 held-out 验证集上，使用 [scripts/compare_fusion_modes.py](/e:/VLM_LLM/from-minimind-to-more-main/minimind-v/scripts/compare_fusion_modes.py) 对 `replace` 和 `qformer_cross_attn` 两种视觉接入方式做了对比。主指标为 **每个有效目标 token 的平均负对数似然 NLL**，辅助指标包括 `PPL`、吞吐、显存峰值和参数量。

实验结果如下：

| mode | NLL | PPL | tok/s | peak memory | total params | trainable params |
|---|---:|---:|---:|---:|---:|---:|
| replace | 1.9922 | 7.33 | 57945.6 | 0.94 GB | 159.79M | 66.86M |
| qformer_cross_attn | 1.7789 | 5.92 | 51607.5 | 1.14 GB | 199.98M | 107.05M |

Delta：

- `NLL delta (qformer_cross_attn - replace) = -0.2133`
- `PPL ratio = 0.808`
- `Throughput ratio = 0.891`
- `Peak memory delta = +0.20 GB`

分析：

- `qformer_cross_attn` 的 `NLL` 从 `1.9922` 降到 `1.7789`，相对下降约 `10.7%`，说明在 held-out 集上的条件建模能力更强。
- `PPL` 从 `7.33` 降到 `5.92`，相对下降约 `19.2%`，这不是边缘提升，而是比较明确的效果收益。
- 成本方面，`qformer_cross_attn` 的吞吐下降约 `10.9%`，显存增加 `0.20 GB`，总参数与可训练参数也明显高于 `replace`。
- 从这组实验看，`qformer_cross_attn` 用中等幅度的额外计算与显存成本，换来了较明显的验证集收益，当前是更优的视觉接入方案。

当前结论：

- 如果优先追求效果，优先使用 `qformer_cross_attn`
- 如果更看重训练速度、显存占用和部署成本，`replace` 仍然是更轻量的备选方案
- 更稳妥的结论还需要补多随机种子或更多下游任务结果，但这次实验已经足以说明 `qformer_cross_attn` 在当前任务分布下具备明显优势

## MiniMind-Omni

在现有视觉分支之外，项目已新增一个独立的 Omni 分支：

- 模型文件：`model/model_omni.py`
- 数据集：`dataset/omni_dataset.py`
- 训练脚本：`trainer/train_pretrain_omni.py`、`trainer/train_sft_omni.py`

当前 Omni 设计：

1. 图像编码器继续使用 `SigLIP2`
2. 音频编码器使用 `Whisper`（默认路径 `./model/whisper-base`）
3. 图像和音频都支持两种接入方式：
   - `fusion_type=replace`
   - `fusion_type=qformer_cross_attn`
4. 在 `qformer_cross_attn` 模式下，图像和音频的 Q-Former 输出会拼接为统一的多模态上下文，再供文本 cross-attention 读取

### Omni 数据格式

`trainer/train_*_omni.py` 读取 parquet，最少需要：

- `conversations`
- `image_bytes` 可选
- `audio_bytes` 可选

文本中占位符规则：

- 图像：`<image>`
- 音频：`<audio>`

对应 tokenizer 中的特殊 token：

- `image_token = <|image_pad|>`
- `audio_token = <|audio_pad|>`

当前实现约束：

- 一个 batch 内不要混合“有图/无图”或“有音频/无音频”的样本
- 最稳妥的做法是按数据源分别构建 parquet，然后分阶段训练

### Omni 使用示例

音频桥接预训练：

```bash
python trainer/train_pretrain_omni.py ^
  --data_path ../dataset/audiocaps_pretrain.parquet ^
  --from_weight sft_vlm ^
  --fusion_type qformer_cross_attn ^
  --freeze_llm 1 ^
  --audio_model_path ../model/whisper-base
```

音频/Omni 指令微调：

```bash
python trainer/train_sft_omni.py ^
  --data_path ../dataset/audiocaps_sft.parquet ^
  --from_weight pretrain_omni ^
  --fusion_type qformer_cross_attn ^
  --freeze_llm 0 ^
  --audio_model_path ../model/whisper-base
```

### 推荐训练顺序

推荐 **先视觉，后音频，再联合**，而不是直接从零联合训练。

建议流程：

1. 先完成当前视觉模型训练，得到较稳定的 `sft_vlm` 或 `sft_vlm_qformer`
2. 使用 `MiniMind-Omni` 从该视觉权重初始化，只训练音频桥接层和少量 LLM 层，完成音频预训练
3. 再做音频指令微调，学习转写、音频描述、音频问答等能力
4. 最后再用少量联合数据做 Omni 对齐，避免音频分支破坏已有视觉能力

不建议一上来全模态联合训练，原因是：

- 音频分支是新接入的，初期梯度会明显更不稳定
- 图像能力已经收敛，直接联合训练更容易出现对旧能力的干扰
- 先单独把音频桥接训练好，再做联合对齐，通常更省算力也更稳

### 可用于训练的音频数据方向

建议按三层目标组织数据：

1. 音频描述/环境声音理解
   - 例如 AudioCaps、Clotho、WavCaps
2. 语音识别/语音转文本
   - 例如 LibriSpeech、Common Voice
3. 音频问答或音视频联合理解
   - 例如 AVQA 等音视频问答数据

一个实用配方是：

- Pretrain：音频描述 + ASR 混合
- SFT：音频指令问答/描述/转写
- Joint alignment：少量图像指令数据 + 少量音频指令数据 + 少量音视频联合数据

## 说明

- 这次改动是“新增可选分支”，不是覆盖原实现。
- 旧的 `replace` 路径和旧权重命名逻辑仍然可用。
- `Q-Former + text cross-attention` 版本目前主要用于架构实验和对比。

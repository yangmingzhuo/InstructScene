# InstructScene 中文文档

## 2️. 🤖 Semantic Graph Prior: 从指令设计语义图

### 📊 数据

Semantic Graph Prior 使用来自 **3D-FRONT** 数据集和 **ChatGPT 生成的指令**进行训练：

- **输入数据**: 
  - 文本指令：自然语言描述（如 "一个带有大床和两个床头柜的现代卧室"）
  - CLIP 文本特征：使用 CLIP ViT-B/32 编码器提取
- **输出数据**:
  - 语义图节点：物体类别（如 bed, nightstand, dresser 等）
  - 语义图边：物体之间的空间关系（如 left, right, front, behind, bigger, smaller 等）
  - 物体特征索引：量化的 OpenShape 特征索引（4 个索引 × 物体数量）
- **数据来源**: 
  - 3D-FRONT 场景：提取真实场景的语义图结构
  - ChatGPT 指令：为每个场景生成多样化的文本描述
  - 训练集、验证集和测试集分别划分
- **数据文件**:
  - `dataset/InstructScene/threed_front_<room_type>/`: 场景数据目录
  - `dataset/InstructScene/3D-FUTURE-chatgpt/`: ChatGPT 生成的指令文本

**数据处理**:
- 文本编码：CLIP ViT-B/32 (512 维全局特征 + 77×512 token 特征)
- 节点编码：物体类别索引（离散）
- 边编码：关系类型索引（离散，对称）
- 物体特征：VQ-VAE 量化索引（离散，从 64 个码本中选择）

### 🏗️ 模型架构

#### SgObjfeatVQDiffusion 主模型
Semantic Graph Prior 使用 VQ-Diffusion（离散扩散模型）从文本指令生成语义图：

**离散扩散过程**:
- 类型：VQ-Diffusion（离散状态扩散）
- 时间步数：100
- 调度策略：alpha schedule（吸收状态扩散）
  - α_t: 保持原状态的概率
  - β_t: 转换为其他状态的概率
  - γ_t: 转换为 [MASK] 状态的概率
- 参数化：x0 预测（直接预测原始状态）
- 采样方法：Gumbel 采样（从分类分布中采样）

**编码器（Encoder）**:
- 节点嵌入：物体类别 → Embedding (512 dim)
- 边嵌入：关系类型 → Embedding (128 dim)
- 物体特征嵌入：VQ 索引 → Embedding (512 dim) → 自注意力池化 → 512 dim
- 时间步嵌入：Sinusoidal 位置编码 (128 dim)
- 文本条件：
  - 全局条件：CLIP 文本嵌入 (512 dim)
  - 上下文条件：CLIP token 特征 (77×512 dim)
- 输入融合：节点嵌入 + 物体特征嵌入

**Transformer 主干网络**:
- 5 层 Graph Transformer 块
- 每层包含：
  - Graph Self-Attention：节点之间的注意力（受边信息调制）
  - Cross-Attention：与文本 token 的交叉注意力
  - AdaLN：自适应层归一化（由时间步和全局文本特征条件化）
  - Gated Feed-Forward：门控前馈网络
- 注意力头数：8
- Dropout：0.1
- 位置编码：1D sinusoidal（实例 ID）

**解码器（Decoder）**:
- 三个独立的输出头：
  1. **节点解码器**: LayerNorm + Linear → 物体类别分布
  2. **边解码器**: LayerNorm + Linear → 关系类型分布
  3. **物体特征解码器**: 
     - 可学习的 query tokens (4 个)
     - 2 层 Transformer (self-attn + cross-attn)
     - LayerNorm + MLP → VQ 索引分布
- 输出：离散分布的 logits（未归一化概率）

**关键特性**:
- 离散扩散：适合处理离散的图结构和类别标签
- 文本条件化：通过 CLIP 特征实现指令驱动生成
- Classifier-free Guidance：训练时随机丢弃文本条件（20%），推理时提升生成质量
- 联合生成：同时生成节点、边和物体特征，保持一致性
- 对称边处理：自动处理关系的对称性（如 left ↔ right）

### 🎯 训练流程

#### 训练配置
```yaml
epochs: 2000
batch_size: 128
optimizer: AdamW
  - learning_rate: 0.0001
  - weight_decay: 0.02
  
loss_weights:
  - vb_x (节点变分损失): 1.0
  - vb_e (边变分损失): 10.0  # 边权重更高，强调关系准确性
  - vb_o (物体特征变分损失): 1.0

diffusion:
  - num_timesteps: 100
  - parameterization: "x0"  # 预测原始状态
  - sample_method: "importance"  # 重要性采样
  - mask_weight: [1.0, 1.0]  # [MASK] token 的权重
  - auxiliary_loss_weight: 5e-4
  - adaptive_auxiliary_loss: true

cfg_drop_ratio: 0.2  # Classifier-free Guidance 丢弃率

EMA (指数移动平均):
  - 使用 EMA 稳定训练
  - max_decay: 0.9999
  - 带 warmup
```

#### 损失函数
1. **节点变分损失（vb_x）**: 
   - KL 散度：q(x_{t-1}|x_t, x_0) || p_θ(x_{t-1}|x_t)
   - 加权：根据 [MASK] token 位置调整权重
2. **边变分损失（vb_e）**: 
   - KL 散度：q(e_{t-1}|e_t, e_0) || p_θ(e_{t-1}|e_t)
   - 权重 10×：强调关系生成的准确性
3. **物体特征变分损失（vb_o）**: 
   - KL 散度：q(o_{t-1}|o_t, o_0) || p_θ(o_{t-1}|o_t)
4. **辅助损失**: KL(x_0 || p_θ(x_0|x_t))，帮助预测原始状态

#### 训练命令
```bash
# bash scripts/train_sg_vq_objfeat.sh <room_type> <tag> <gpu_id>
bash scripts/train_sg_vq_objfeat.sh bedroom bedroom_sgdiffusion_vq_objfeat 0
```

**注意**: Semantic Graph Prior 的训练独立于 Layout Decoder，可以并行训练。

### 🔍 推理与应用

#### 生成（Generation）
从文本指令生成全新的语义图：
```bash
# bash scripts/inference_sg_vq_objfeat.sh <room_type> <tag> <gpu_id> <epoch> <fvqvae_tag> <sg2sc_tag> <cfg_scale> <sg2sc_cfg_scale>
bash scripts/inference_sg_vq_objfeat.sh bedroom bedroom_sgdiffusion_vq_objfeat 0 -1 threedfront_objfeat_vqvae bedroom_sg2scdiffusion_objfeat 1.0 1.0
# cfg_scale: Semantic Graph Prior 的 CFG 强度
# sg2sc_cfg_scale: Layout Decoder 的 CFG 强度
```

#### 风格化（Stylization）
给定场景结构（节点和边），生成新的物体风格：
```bash
# 将推理脚本中的 python 文件名改为 stylize_sg.py
python3 src/stylize_sg.py configs/bedroom_sg_diffusion_vq_objfeat.yaml \
  --tag bedroom_sgdiffusion_vq_objfeat \
  --fvqvae_tag threedfront_objfeat_vqvae \
  --sg2sc_tag bedroom_sg2scdiffusion_objfeat \
  --cfg_scale 1.0 --sg2sc_cfg_scale 1.0
```

#### 重排列（Rearrangement）
给定物体集合（节点和物体特征），生成新的空间布局：
```bash
# 将推理脚本中的 python 文件名改为 rearrange_sg.py
python3 src/rearrange_sg.py configs/bedroom_sg_diffusion_vq_objfeat.yaml \
  --tag bedroom_sgdiffusion_vq_objfeat \
  --fvqvae_tag threedfront_objfeat_vqvae \
  --sg2sc_tag bedroom_sg2scdiffusion_objfeat \
  --cfg_scale 1.0 --sg2sc_cfg_scale 1.0
```

#### 补全（Completion）
给定部分场景，补全缺失的物体：
```bash
# 将推理脚本中的 python 文件名改为 complete_sg.py
python3 src/complete_sg.py configs/bedroom_sg_diffusion_vq_objfeat.yaml \
  --tag bedroom_sgdiffusion_vq_objfeat \
  --fvqvae_tag threedfront_objfeat_vqvae \
  --sg2sc_tag bedroom_sg2scdiffusion_objfeat \
  --cfg_scale 1.0 --sg2sc_cfg_scale 1.0
```

### 📈 评估指标

#### FID, CLIP-FID 和 KID
评估生成场景的视觉质量和多样性：
```bash
python3 src/compute_fid_scores.py configs/bedroom_sgdiffusion_vq_objfeat.yaml \
  --tag bedroom_sgdiffusion_vq_objfeat \
  --checkpoint_epoch -1
```

- **FID (Fréchet Inception Distance)**: 生成图像与真实图像的分布距离
- **CLIP-FID**: 基于 CLIP 特征的 FID
- **KID (Kernel Inception Distance)**: 无偏的分布距离估计

#### SCA (Scene Classification Accuracy)
评估生成场景与真实场景的可区分性：
```bash
python3 src/synthetic_vs_real_classifier.py configs/bedroom_sgdiffusion_vq_objfeat.yaml \
  --tag bedroom_sgdiffusion_vq_objfeat \
  --checkpoint_epoch -1
```

- 训练分类器区分真实场景和生成场景
- SCA 接近 50% 表示生成场景高度真实

#### 其他指标
- **iRecall**: 场景中物体检索的准确率
- **物体分布**: 生成场景中物体类别的分布
- **关系准确率**: 生成的空间关系的合理性

### 💡 应用场景

Semantic Graph Prior 在 InstructScene 中的作用：

1. **指令理解**: 将自然语言转换为结构化的语义图表示
2. **场景规划**: 在布局生成前规划物体类型和关系
3. **创意生成**: 从多样化的指令生成不同风格的场景
4. **场景编辑**:
   - **风格化**: 改变物体外观但保持布局结构
   - **重排列**: 重新组织物体的空间关系
   - **补全**: 智能添加缺失的物体
5. **端到端生成**: 配合 Layout Decoder 实现从指令到 3D 场景的完整流程

### 📁 相关文件

- **模型代码**: `src/models/sg_diffusion_vq_objfeat.py`
- **训练脚本**: `src/train_sg.py`
- **推理脚本**: 
  - `src/generate_sg.py` (生成)
  - `src/stylize_sg.py` (风格化)
  - `src/rearrange_sg.py` (重排列)
  - `src/complete_sg.py` (补全)
- **配置文件**: 
  - `configs/bedroom_sg_diffusion_vq_objfeat.yaml`
  - `configs/diningroom_sg_diffusion_vq_objfeat.yaml`
  - `configs/livingroom_sg_diffusion_vq_objfeat.yaml`
- **数据加载**: `src/data/threed_front_dataset.py`
- **网络模块**: `src/models/networks/transformer.py`
- **文本编码器**: `src/models/clip_encoders.py`

### 🔗 与其他模块的关系

- **依赖 fVQ-VAE**: 生成物体特征的量化索引
- **输出到 Layout Decoder**: 生成的语义图作为条件输入
- **文本编码**: 使用 CLIP ViT-B/32 编码文本指令
- **完整流程**: 文本指令 → Semantic Graph → Layout Decoder → 3D 场景


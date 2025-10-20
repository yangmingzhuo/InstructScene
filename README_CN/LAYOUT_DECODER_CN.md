# InstructScene 中文文档

## 1️. 🦾 Layout Decoder: 从语义图生成 3D 场景布局

### 📊 数据

Layout Decoder 使用来自 **3D-FRONT** 和 **3D-FUTURE** 数据集的场景数据进行训练：

- **输入数据**: 
  - 语义图节点：物体类别标签
  - 语义图边：物体之间的空间关系（如 left, right, front, behind 等）
  - 物体特征：量化的 OpenShape 特征索引（来自 fVQ-VAE）
- **输出数据**:
  - 物体边界框：位置 (x, y, z)、尺寸 (length, width, height)、旋转角度 (sin θ, cos θ)
- **数据来源**: 
  - 从 3D-FRONT 的三种房间类型（卧室、餐厅、客厅）中收集场景
  - 训练集、验证集和测试集分别划分
  - 使用真实语义图（ground-truth）进行训练
- **数据文件**:
  - `dataset/InstructScene/threed_front_<room_type>/`: 场景数据目录
  - 每个场景包含：物体信息、边界框、语义图结构

**数据编码**:
- 节点类别：one-hot 编码
- 边关系：对称关系编码（如 left ↔ right）
- 物体特征：VQ-VAE 量化索引（4 个索引，每个从 64 个码本中选择）
- 边界框：归一化到 [-1, 1] 范围

### 🏗️ 模型架构

#### Sg2ScDiffusion 主模型
Layout Decoder 使用基于 Diffusion 的条件生成模型，从语义图生成 3D 场景布局：

**扩散调度器（Diffusion Scheduler）**:
- 类型：DDPM（训练）/ DDIM（推理加速）
- 训练时间步：1000
- β 调度：线性调度，从 0.0001 到 0.02
- 预测类型：epsilon（噪声）预测
- 推理时间步：100（可配置）

**编码器（Encoder）**:
- 节点嵌入：物体类别 → Embedding (512 dim)
- 边嵌入：关系类型 → Embedding (128 dim)
- 物体特征：VQ 索引 → OpenShape 特征 (1280 dim) [通过 fVQ-VAE 解码]
- 时间步嵌入：Sinusoidal 位置编码 (128 dim)
- 噪声边界框嵌入：8 维向量（位置 3 + 尺寸 3 + 角度 2）
- 输入融合：[节点嵌入 + 物体特征 + 噪声边界框] → 512 维

**Transformer 主干网络**:
- 5 层 Graph Transformer 块
- 每层包含：
  - Graph Self-Attention：节点之间的注意力（受边信息调制）
  - AdaLN：自适应层归一化（由时间步条件化）
  - Gated Feed-Forward：门控前馈网络
- 注意力头数：8
- Dropout：0.1

**解码器（Decoder）**:
- 层归一化 + 线性投影
- 输出：噪声预测 (8 维) [位置 3 + 尺寸 3 + 角度 2]
- 使用物体掩码处理变长场景

**关键特性**:
- 条件扩散：以语义图结构和物体特征为条件
- 变长支持：通过掩码处理不同物体数量的场景
- Classifier-free Guidance：训练时随机丢弃边信息（20%），推理时提升生成质量
- 物理约束：通过扩散过程保持场景合理性

### 🎯 训练流程

#### 训练配置
```yaml
epochs: 2000
batch_size: 128
optimizer: AdamW
  - learning_rate: 0.0001
  - weight_decay: 0.02
  
loss_weights:
  - pos_mse (位置 MSE 损失): 1.0
  - size_mse (尺寸 MSE 损失): 1.0
  - angle_mse (角度 MSE 损失): 1.0

diffusion:
  - num_train_timesteps: 1000
  - num_inference_timesteps: 100
  - beta_schedule: "linear"
  - prediction_type: "epsilon"

EMA (指数移动平均):
  - 使用 EMA 稳定训练
  - max_decay: 0.9999
  - 带 warmup
```

#### 损失函数
1. **位置损失（pos_mse）**: 预测噪声与真实噪声之间的 MSE（位置部分）
2. **尺寸损失（size_mse）**: 预测噪声与真实噪声之间的 MSE（尺寸部分）
3. **角度损失（angle_mse）**: 预测噪声与真实噪声之间的 MSE（角度部分）

#### 训练命令
```bash
# bash scripts/train_sg2sc_objfeat.sh <room_type> <tag> <gpu_id> <fvqvae_tag>
bash scripts/train_sg2sc_objfeat.sh bedroom bedroom_sg2scdiffusion_objfeat 0 threedfront_objfeat_vqvae
```

**注意**: Layout Decoder 的训练独立于 Semantic Graph Prior，使用真实语义图作为条件。

### 🔍 推理与评估

#### 推理脚本
```bash
# bash scripts/inference_sg2sc_objfeat.sh <room_type> <tag> <gpu_id> <epoch> <fvqvae_tag> <cfg_scale>
bash scripts/inference_sg2sc_objfeat.sh bedroom bedroom_sg2scdiffusion_objfeat 0 -1 threedfront_objfeat_vqvae 1.0
# 使用 -1 表示加载最新的 checkpoint
# cfg_scale: Classifier-free Guidance 强度，建议 1.0-2.0
```

#### 可视化
```bash
# 在推理脚本中添加可视化参数
# 将 --n_scenes 0 替换为 --n_scenes 5 --visualize --resolution 1024
# 生成 5 个场景并渲染为 1024x1024 分辨率图像
```

#### 评估指标
- **iRecall (intersection Recall)**: 场景中物体检索的准确率
  - 基于生成边界框与真实边界框的 IoU
  - 评估布局的准确性和物理合理性
- **KL 散度**: 生成布局分布与真实分布的差异
- **碰撞率**: 物体之间的重叠比例

#### 输出
- `out/<tag>/synthetic_scenes/`: 生成场景目录
- 场景数据：边界框、物体类别、特征等
- 渲染图像：使用 Blender 渲染的场景图像（如启用 --visualize）

### 💡 应用场景

Layout Decoder 在 InstructScene 中的作用：

1. **场景具现化**: 将抽象的语义图转换为具体的 3D 布局
2. **布局优化**: 通过扩散过程生成物理合理的物体摆放
3. **条件生成**: 支持从指令生成的语义图进行场景合成
4. **场景编辑**: 
   - **补全（Completion）**: 在现有场景中添加新物体
   - 配合 Semantic Graph Prior 实现端到端的指令驱动生成

### 📁 相关文件

- **模型代码**: `src/models/sg2sc_diffusion.py`
- **训练脚本**: `src/train_sg2sc.py`
- **推理脚本**: `src/generate_sg2sc.py`
- **配置文件**: 
  - `configs/bedroom_sg2sc_diffusion_objfeat.yaml`
  - `configs/diningroom_sg2sc_diffusion_objfeat.yaml`
  - `configs/livingroom_sg2sc_diffusion_objfeat.yaml`
- **数据加载**: `src/data/threed_front_dataset.py`
- **网络模块**: `src/models/networks/transformer.py`

### 🔗 与其他模块的关系

- **依赖 fVQ-VAE**: 使用预训练的 fVQ-VAE 将量化索引解码为物体特征
- **配合 Semantic Graph Prior**: 接收生成的语义图，输出完整的 3D 场景
- **输出到可视化**: 生成的边界框可用于物体检索和 Blender 渲染


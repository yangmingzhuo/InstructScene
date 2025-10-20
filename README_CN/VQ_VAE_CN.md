# InstructScene 中文文档

## 0️. 📦 fVQ-VAE: 量化物体的 OpenShape/CLIP 特征

### 📊 数据

fVQ-VAE 使用来自 **3D-FRONT** 和 **3D-FUTURE** 数据集的物体特征进行训练：

- **特征类型**: OpenShape ViT-G/14 特征（维度：1280）
- **数据来源**: 
  - 从 3D-FRONT 的三种房间类型（卧室、餐厅、客厅）中收集所有物体
  - 训练集和验证集物体分别保存在 pickle 文件中
  - 物体特征已经过 L2 归一化处理
- **数据文件**:
  - `threed_front_objfeat_openshape_vitg14_train.pkl`: 训练集物体特征
  - `threed_front_objfeat_openshape_vitg14_val.pkl`: 验证集物体特征
  - `objfeat_bounds.pkl`: 特征值的最小/最大边界（用于归一化）

**预处理流程**:
1. 收集所有房间类型的物体（去重）
2. 提取每个物体的 OpenShape 特征
3. 计算训练集特征的最小值和最大值
4. 在训练时将特征归一化到 [-1, 1] 范围

### 🏗️ 模型架构

#### ObjectFeatureVQVAE 主模型
核心模型使用基于 Transformer 的 VQ-VAE 架构来量化物体特征：

**编码器（Encoder）**:
- 输入：OpenShape 特征 (B, 1280)
- 可学习的 token embeddings (4 tokens, 512 dim)
- 4 层 Transformer 编码器（带有 self-attention 和 cross-attention）
- 使用 1D sinusoidal 位置编码
- 输出：量化前的 token 表示 (B, 4, 512)

**向量量化器（Vector Quantizer）**:
- 类型：Gumbel Softmax VQ-VAE
- 码本大小：64 个条目
- 嵌入维度：512
- Gumbel 温度：1.0
- KL 权重：5e-4
- 支持 straight-through 梯度估计

**解码器（Decoder）**:
- 4 层 Transformer 解码器（仅 self-attention）
- 使用 1D sinusoidal 位置编码
- 平均池化所有 tokens
- MLP 输出层：512 → 1280 → 1280
- 输出限制在 [-1, 1] 范围
- 最终输出：重构的 OpenShape 特征 (B, 1280)

**关键特性**:
- 将连续的高维特征（1280维）压缩为 4 个离散索引
- 每个索引从 64 个可能的码本条目中选择
- 使用 Gumbel Softmax 实现可微分的离散采样
- 支持从特征到索引的编码和从索引到特征的解码

### 🎯 VAE 训练

#### 训练配置
```yaml
epochs: 2000
batch_size: 128
optimizer: AdamW
  - learning_rate: 0.0001
  - weight_decay: 0.02
  
loss_weights:
  - qloss (量化损失): 1.0
  - rec_mse (重构 MSE 损失): 1.0

EMA (指数移动平均):
  - 使用 EMA 稳定训练
  - max_decay: 0.9999
  - 带 warmup
```

#### 损失函数
1. **量化损失（qloss）**: Gumbel Softmax 的 KL 散度损失
2. **重构损失（rec_mse）**: 重构特征与原始特征之间的 MSE 损失

#### 训练流程
```bash
# 下载预训练模型
python dataset_download.py

# 或从头训练（需要更新数据集中的量化索引）
bash scripts/train_objfeatvqvae.sh threedfront_objfeat_vqvae_baseline 0
```

**注意**: 如果从头训练 fVQ-VAE，需要使用新模型重新生成数据集中所有物体的量化索引。

### 🔍 推理与评估

#### 推理脚本
```bash
# bash scripts/inference_objfeatvqvae.sh <tag> <gpu_id> <epoch>
bash scripts/inference_objfeatvqvae.sh threedfront_objfeat_vqvae_baseline 0 1999
# 使用 -1 表示加载最新的 checkpoint
```

#### 评估指标
- **检索准确率**: 通过重构特征在特征空间中检索最相似物体的准确率
- 评估流程：
  1. 对原始特征进行编码和量化
  2. 从量化索引重构特征
  3. 使用余弦相似度在所有物体中检索
  4. 计算检索到正确物体的准确率

#### 输出
- `reconstruct_objfeats/epoch_XXXXX/`: 重构结果目录
- `batch_XXX.jpg`: 可视化真实物体图像与检索物体图像的对比
- `eval.txt`: 检索准确率统计

### 💡 应用场景

fVQ-VAE 在 InstructScene 中的作用：

1. **特征压缩**: 将 1280 维的连续特征压缩为 4 个离散索引
2. **语义图先验**: 为语义图节点提供离散的物体表示
3. **布局解码器**: 从量化索引重构物体特征，用于物体检索
4. **下游任务**: 支持场景生成、风格化、重排列和补全等任务

### 📁 相关文件

- **模型代码**: `src/models/objfeat_vqvae.py`
- **训练脚本**: `src/train_objfeatvqvae.py`
- **推理脚本**: `src/reconstruct_objfeatvqvae.py`
- **配置文件**: `configs/threedfront_objfeat_vqvae.yaml`
- **数据加载**: `src/data/threed_future_dataset.py`
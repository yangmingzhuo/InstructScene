# 如何为自己训练的 VQ-VAE 重新生成数据集

## 📌 问题说明

如果你自己训练了一个新的 VQ-VAE 模型，需要重新为数据集生成量化索引（VQ indices），因为：

1. **现有数据集已包含预计算的 VQ 索引**：在每个场景目录的 `models_info.pkl` 文件中存储了物体的量化索引
2. **索引依赖于特定的 VQ-VAE 模型**：不同的 VQ-VAE 模型（不同的码本）会产生不同的量化索引
3. **后续训练需要这些索引**：Semantic Graph Prior 和 Layout Decoder 都依赖这些预计算的索引

## 🔍 数据集结构

每个场景目录包含：
```
threed_front_bedroom/
└── <scene_id>/
    ├── boxes.npz                          # 边界框数据
    ├── descriptions.pkl                   # ChatGPT 生成的文本描述
    ├── models_info.pkl                    # ⭐ 物体信息（包含 VQ 索引）
    ├── openshape_pointbert_vitg14.npy    # OpenShape 特征
    ├── relations.npy                      # 物体关系
    └── ...
```

`models_info.pkl` 结构（列表，每个元素对应一个物体）：
```python
[
    {
        "model_jid": "xxx",                    # 物体 ID
        "category": "bed",                      # 物体类别
        "objfeat_vq_indices": [12, 45, 3, 28], # ⭐ VQ 量化索引（4个）
        ...
    },
    ...
]
```

## 🛠️ 重新生成 VQ 索引的步骤

### 步骤 1: 准备你的 VQ-VAE 模型

确保你已经训练好了 VQ-VAE 模型并保存了检查点：
```bash
# 训练完成后，模型保存在
out/<your_vqvae_tag>/checkpoints/epoch_XXXXX.pth
```

### 步骤 2: 创建索引生成脚本

创建一个 Python 脚本来重新生成所有场景的 VQ 索引：

```python
# regenerate_vq_indices.py
import os
import pickle
import argparse
from tqdm import tqdm
import numpy as np
import torch

from src.models import model_from_config
from src.utils import load_config, load_checkpoints


def regenerate_indices_for_scene(scene_dir, vqvae_model, device):
    """为单个场景重新生成 VQ 索引"""
    models_info_path = os.path.join(scene_dir, "models_info.pkl")
    
    # 检查文件是否存在
    if not os.path.exists(models_info_path):
        return False
    
    # 加载现有的 models_info
    with open(models_info_path, "rb") as f:
        models_info = pickle.load(f)
    
    # 加载 OpenShape 特征
    openshape_feat_path = os.path.join(scene_dir, "openshape_pointbert_vitg14.npy")
    if not os.path.exists(openshape_feat_path):
        return False
    
    openshape_feats = np.load(openshape_feat_path)  # (N, 1280)
    
    # 使用 VQ-VAE 编码为索引
    with torch.no_grad():
        # 归一化特征到 [-1, 1]（使用训练时的边界）
        # 注意：需要使用与训练 VQ-VAE 时相同的归一化参数
        feats_normalized = torch.from_numpy(openshape_feats).float().to(device)
        
        # 编码为 VQ 索引
        vq_indices = vqvae_model.quantize_to_indices(feats_normalized)  # (N, K)
        vq_indices = vq_indices.cpu().numpy()
    
    # 更新 models_info 中的索引
    assert len(models_info) == len(vq_indices), \
        f"物体数量不匹配: {len(models_info)} vs {len(vq_indices)}"
    
    for i, model_info in enumerate(models_info):
        model_info["objfeat_vq_indices"] = vq_indices[i].tolist()
    
    # 保存更新后的 models_info
    with open(models_info_path, "wb") as f:
        pickle.dump(models_info, f)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="重新生成数据集的 VQ 索引"
    )
    parser.add_argument(
        "--vqvae_config",
        type=str,
        required=True,
        help="VQ-VAE 配置文件路径"
    )
    parser.add_argument(
        "--vqvae_tag",
        type=str,
        required=True,
        help="VQ-VAE 实验标签"
    )
    parser.add_argument(
        "--vqvae_epoch",
        type=int,
        default=-1,
        help="VQ-VAE checkpoint epoch（-1 表示最新）"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="数据集目录（如 dataset/InstructScene/threed_front_bedroom）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA 设备"
    )
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")
    
    # 加载 VQ-VAE 模型
    print(f"加载 VQ-VAE 模型: {args.vqvae_tag}")
    config = load_config(args.vqvae_config)
    vqvae_model = model_from_config(config)
    vqvae_model = vqvae_model.to(device)
    vqvae_model.eval()
    
    # 加载检查点
    ckpt_path = os.path.join("out", args.vqvae_tag, "checkpoints")
    vqvae_model = load_checkpoints(vqvae_model, ckpt_path, args.vqvae_epoch, device)
    print(f"已加载 checkpoint\n")
    
    # 遍历所有场景目录
    scene_dirs = [
        os.path.join(args.dataset_dir, d)
        for d in os.listdir(args.dataset_dir)
        if os.path.isdir(os.path.join(args.dataset_dir, d))
        and not d.startswith("_")  # 跳过渲染目录
    ]
    
    print(f"找到 {len(scene_dirs)} 个场景")
    print("开始重新生成 VQ 索引...\n")
    
    success_count = 0
    for scene_dir in tqdm(scene_dirs, desc="处理场景"):
        if regenerate_indices_for_scene(scene_dir, vqvae_model, device):
            success_count += 1
    
    print(f"\n完成！成功处理 {success_count}/{len(scene_dirs)} 个场景")


if __name__ == "__main__":
    main()
```

### 步骤 3: 运行脚本重新生成索引

为每个房间类型运行脚本：

```bash
# 卧室
python -m src.regenerate_vq_indices \
  --vqvae_config configs/threedfront_objfeat_vqvae.yaml \
  --vqvae_tag <your_vqvae_tag> \
  --vqvae_epoch -1 \
  --dataset_dir dataset/InstructScene/threed_front_bedroom \
  --device 0

# 餐厅
python -m src.regenerate_vq_indices \
  --vqvae_config configs/threedfront_objfeat_vqvae.yaml \
  --vqvae_tag <your_vqvae_tag> \
  --vqvae_epoch -1 \
  --dataset_dir dataset/InstructScene/threed_front_diningroom \
  --device 0

# 客厅
python -m src.regenerate_vq_indices \
  --vqvae_config configs/threedfront_objfeat_vqvae.yaml \
  --vqvae_tag <your_vqvae_tag> \
  --vqvae_epoch -1 \
  --dataset_dir dataset/InstructScene/threed_front_livingroom \
  --device 0
```

### 步骤 4: 验证生成的索引

运行一个简单的验证脚本：

```python
# verify_indices.py
import os
import pickle
import numpy as np

def verify_scene(scene_dir):
    """验证单个场景的 VQ 索引"""
    models_info_path = os.path.join(scene_dir, "models_info.pkl")
    
    with open(models_info_path, "rb") as f:
        models_info = pickle.load(f)
    
    for i, model_info in enumerate(models_info):
        if "objfeat_vq_indices" not in model_info:
            print(f"❌ 缺少 VQ 索引: {scene_dir}, 物体 {i}")
            return False
        
        indices = model_info["objfeat_vq_indices"]
        if not isinstance(indices, (list, np.ndarray)):
            print(f"❌ 索引格式错误: {scene_dir}, 物体 {i}")
            return False
        
        if len(indices) != 4:  # 应该有 4 个索引
            print(f"❌ 索引数量错误: {scene_dir}, 物体 {i}, 数量: {len(indices)}")
            return False
        
        if not all(0 <= idx < 64 for idx in indices):  # 索引应该在 [0, 64) 范围内
            print(f"❌ 索引值超出范围: {scene_dir}, 物体 {i}, 索引: {indices}")
            return False
    
    return True

# 验证所有场景
dataset_dir = "dataset/InstructScene/threed_front_bedroom"
scene_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

success = 0
for scene_dir in scene_dirs:
    if verify_scene(os.path.join(dataset_dir, scene_dir)):
        success += 1

print(f"验证完成: {success}/{len(scene_dirs)} 个场景通过")
```

### 步骤 5: 使用新索引训练后续模型

现在你可以使用更新后的数据集训练 Semantic Graph Prior 和 Layout Decoder：

```bash
# 训练 Layout Decoder
bash scripts/train_sg2sc_objfeat.sh bedroom bedroom_sg2scdiffusion_objfeat 0 <your_vqvae_tag>

# 训练 Semantic Graph Prior
bash scripts/train_sg_vq_objfeat.sh bedroom bedroom_sgdiffusion_vq_objfeat 0
```

## ⚠️ 重要注意事项

### 1. 特征归一化
确保使用与训练 VQ-VAE 时**相同的归一化参数**：
```python
# 如果训练时使用了特定的 min/max 归一化
bounds_path = os.path.join("out", vqvae_tag, "objfeat_bounds.pkl")
with open(bounds_path, "rb") as f:
    bounds = pickle.load(f)

feats_min, feats_max = bounds["min"], bounds["max"]
feats_normalized = 2 * (feats - feats_min) / (feats_max - feats_min) - 1
```

### 2. 码本大小
如果你的 VQ-VAE 使用了不同的码本大小（默认是 64），需要：
- 更新脚本中的索引范围检查
- 更新后续模型配置中的 `num_objfeat_embeds` 参数

### 3. Token 数量
如果你的 VQ-VAE 使用了不同数量的 tokens（默认是 4），需要：
- 更新脚本中的索引数量检查
- 更新后续模型配置中相关的维度参数

### 4. 备份原始数据
在运行重新生成脚本之前，**强烈建议备份原始数据集**：
```bash
# 备份整个数据集目录
cp -r dataset/InstructScene dataset/InstructScene_backup
```

## 🔧 故障排除

### 问题 1: 特征文件不存在
**错误**: `FileNotFoundError: openshape_pointbert_vitg14.npy`

**解决方案**: 确保场景目录中包含 OpenShape 特征文件。如果缺失，需要先提取特征。

### 问题 2: 维度不匹配
**错误**: `RuntimeError: Expected input dimension xxx but got yyy`

**解决方案**: 检查 VQ-VAE 配置中的 `kv_dim` 参数是否与特征维度匹配（OpenShape ViT-G/14 是 1280）。

### 问题 3: 索引超出范围
**错误**: 生成的索引值 > 63

**解决方案**: 检查量化器的码本大小设置，确保与配置一致。

## 📚 相关文件

- **VQ-VAE 模型**: `src/models/objfeat_vqvae.py`
- **数据加载**: `src/data/threed_front_dataset.py`
- **配置文件**: `configs/threedfront_objfeat_vqvae.yaml`
- **索引使用示例**: 
  - `src/models/sg2sc_diffusion.py` (Line 72-96)
  - `src/models/sg_diffusion_vq_objfeat.py` (Line 104)

## 💡 进阶：并行处理

如果数据集很大，可以使用多进程加速：

```python
from multiprocessing import Pool

def process_scene_wrapper(args):
    scene_dir, vqvae_model, device = args
    return regenerate_indices_for_scene(scene_dir, vqvae_model, device)

# 在 main() 函数中
with Pool(processes=4) as pool:  # 使用 4 个进程
    args_list = [(scene_dir, vqvae_model, device) for scene_dir in scene_dirs]
    results = list(tqdm(
        pool.imap(process_scene_wrapper, args_list),
        total=len(scene_dirs),
        desc="处理场景"
    ))
```

## 📖 总结

重新生成 VQ 索引的核心步骤：
1. ✅ 训练你的 VQ-VAE 模型
2. ✅ 使用 `quantize_to_indices()` 方法编码特征
3. ✅ 更新每个场景的 `models_info.pkl` 文件
4. ✅ 验证索引的正确性
5. ✅ 使用更新后的数据集训练后续模型

这样就可以确保你的整个训练流程使用一致的 VQ-VAE 编码！


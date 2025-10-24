# 🎨 手动选择3D模型指南

## 快速回答

### ✅ 不需要完整的3D-FUTURE数据集！

只需要两个文件：
1. **`dataset_directory`**: `dataset/InstructScene/threed_front_bedroom`
2. **`path_to_pickled_3d_futute_models`**: `dataset/InstructScene/threed_future_model_bedroom.pkl`

配置文件已包含这些设置，**不需要设置** `path_to_3d_future_dataset_directory`。

## 🔍 浏览可用模型

### 查看所有类别

```bash
python custom_model_selection.py --room_type bedroom
```

输出示例：
```
double_bed          :  201 个模型
nightstand          :  265 个模型
wardrobe            :  218 个模型
...
```

### 查看特定类别的所有模型

```bash
# 查看所有double_bed模型
python custom_model_selection.py --room_type bedroom --category double_bed --details

# 查看nightstand模型
python custom_model_selection.py --room_type bedroom --category nightstand --details
```

输出：
```
[0] jid: e3560eb3-d4e1-4add-8b51-b3dd5ec6943b
    尺寸 (L×W×H): 1.04 × 0.47 × 1.05 m

[1] jid: a61d62b3-f40c-4099-b55f-0ca94c30e1aa
    尺寸 (L×W×H): 1.01 × 0.51 × 1.10 m
...
```

### 按尺寸搜索模型

```bash
# 搜索尺寸在1.0-1.5m之间的床
python custom_model_selection.py \
    --room_type bedroom \
    --category double_bed \
    --min_size 1.0 0.4 0.8 \
    --max_size 1.5 1.0 1.3 \
    --search
```

## 📝 方式1: 自动选择（当前默认）

系统会根据以下条件自动选择最匹配的模型：
- 物体类别
- 边界框尺寸
- OpenShape特征（相似度）

```python
# 不需要指定jid，系统自动检索
objs = [8, 12, 12]  # double_bed, nightstand, nightstand
```

## 🎯 方式2: 手动指定jid（推荐用于精确控制）

### 步骤1: 查找想要的模型jid

```bash
# 查看double_bed的所有可用模型
python custom_model_selection.py --room_type bedroom --category double_bed --details | head -30

# 查看nightstand的所有可用模型
python custom_model_selection.py --room_type bedroom --category nightstand --details | head -30
```

### 步骤2: 记录选中的jid

例如：
- 床（double_bed）: `f43310eb-270b-49ec-aef9-11103921b224` (1.12 × 1.13 × 1.10 m)
- 床头柜1（nightstand）: `2fd5c449-667b-418e-b53d-3596f954eab7` (0.36 × 0.29 × 0.19 m)
- 床头柜2（nightstand）: `5e4c8a81-79dd-43c2-a117-f3b9bda9e560` (0.40 × 0.26 × 0.23 m)

### 步骤3: 在场景配置中指定jid

在场景数据字典中添加 `model_jids` 字段：

```python
scene_data = {
    'room_type': 'bedroom',
    'template': 'custom',
    'objs': [8, 12, 12],
    'edges': edges,
    'objfeat_vq_indices': objfeat_vq_indices,
    'objfeats': objfeats[0],
    'boxes': boxes_pred[0].numpy(),
    'translations': boxes["translations"][0].numpy(),
    'sizes': boxes["sizes"][0].numpy(),
    'angles': boxes["angles"][0].numpy(),
    
    # 手动指定模型jid（可选）
    'model_jids': [
        'f43310eb-270b-49ec-aef9-11103921b224',  # 床
        '2fd5c449-667b-418e-b53d-3596f954eab7',  # 床头柜1
        '5e4c8a81-79dd-43c2-a117-f3b9bda9e560',  # 床头柜2
    ]
}
```

## 📊 完整示例

### 1. 浏览并选择模型

```bash
# 步骤1: 查看所有double_bed
python custom_model_selection.py --room_type bedroom --category double_bed --details | less

# 步骤2: 选择jid: f43310eb-270b-49ec-aef9-11103921b224

# 步骤3: 查看所有nightstand
python custom_model_selection.py --room_type bedroom --category nightstand --details | less

# 步骤4: 选择两个jid
```

### 2. 创建自定义场景配置

创建文件 `my_bedroom_config.json`:

```json
{
  "room_type": "bedroom",
  "objects": [
    {
      "type": "double_bed",
      "type_id": 8,
      "jid": "f43310eb-270b-49ec-aef9-11103921b224"
    },
    {
      "type": "nightstand",
      "type_id": 12,
      "jid": "2fd5c449-667b-418e-b53d-3596f954eab7"
    },
    {
      "type": "nightstand",
      "type_id": 12,
      "jid": "5e4c8a81-79dd-43c2-a117-f3b9bda9e560"
    }
  ],
  "relations": [
    {"from": 0, "to": 1, "type": "right_of"},
    {"from": 0, "to": 2, "type": "left_of"}
  ]
}
```

### 3. 生成并渲染

```bash
# 使用自定义配置生成场景
python custom_scene_with_visualization.py \
    --room_type bedroom \
    --template simple \
    --draw_graph \
    --render \
    --resolution 1024 \
    --num_views 8 \
    --device 0
```

## 🔧 高级用法

### 按尺寸筛选

```bash
# 找大尺寸的床 (> 1.5m)
python custom_model_selection.py \
    --room_type bedroom \
    --category double_bed \
    --min_size 1.5 0.5 0.8 \
    --search

# 找小尺寸的床头柜 (< 0.3m)
python custom_model_selection.py \
    --room_type bedroom \
    --category nightstand \
    --max_size 0.3 0.3 0.3 \
    --search
```

### 查看其他房间类型

```bash
# 客厅
python custom_model_selection.py --room_type livingroom

# 餐厅
python custom_model_selection.py --room_type diningroom
```

## 💡 实用技巧

### 1. 保存模型列表

```bash
# 导出double_bed列表到文件
python custom_model_selection.py --room_type bedroom --category double_bed --details > double_bed_models.txt
```

### 2. 选择协调的模型

选择尺寸相近的模型，使场景更协调：

```bash
# 找类似尺寸的床
python custom_model_selection.py \
    --room_type bedroom \
    --category double_bed \
    --min_size 1.0 0.9 0.4 \
    --max_size 1.2 1.1 0.6 \
    --search
```

### 3. 批量查看

```bash
# 查看多个类别
for cat in double_bed nightstand wardrobe; do
    echo "=== $cat ==="
    python custom_model_selection.py --room_type bedroom --category $cat --details | head -20
done
```

## 📁 文件位置

```
dataset/InstructScene/
├── threed_future_model_bedroom.pkl      ← 卧室物体 (2398个模型)
├── threed_future_model_livingroom.pkl   ← 客厅物体
├── threed_future_model_diningroom.pkl   ← 餐厅物体
└── threed_front_bedroom/                ← 场景数据
    └── <scene_id>/
        └── models_info.pkl
```

## 🎯 总结

| 方式 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **自动选择** | 简单快速，无需手动选择 | 无法精确控制 | 快速原型，大批量生成 |
| **手动指定jid** | 完全控制，可复现 | 需要预先浏览选择 | 精确设计，特定需求 |

### 推荐流程

1. **原型阶段**: 使用自动选择快速测试
2. **精细调整**: 浏览模型库，手动选择理想的jid
3. **最终输出**: 使用指定jid确保可复现性

## 常见问题

**Q: 如何知道某个jid对应的模型是什么样子？**

A: 可以先渲染一次查看，或者查看模型的size来判断大致形状。

**Q: 可以混用不同房间类型的模型吗？**

A: 不建议，因为不同pkl文件中的模型可能有不同的风格特征。

**Q: 如何保存我的模型选择？**

A: 将选择的jid保存在配置文件或场景数据中，便于复用。

## 下一步

- 📖 查看完整的可视化文档
- 🎨 尝试不同的物体组合
- 🔧 自定义场景模板

---

**提示**: 系统默认使用pickle文件，不需要完整的3D-FUTURE数据集！✅


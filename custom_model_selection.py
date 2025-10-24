"""
手动选择物体模型的辅助脚本

功能：
1. 浏览可用的物体模型
2. 按类别筛选
3. 查看模型详情
4. 生成自定义场景配置

用法:
python custom_model_selection.py --room_type bedroom
"""

import argparse
import pickle
import numpy as np

def list_models_by_category(pkl_file, category=None, show_details=False):
    """列出所有或特定类别的模型"""
    with open(pkl_file, 'rb') as f:
        dataset = pickle.load(f)
    
    # 按类别分组
    models_by_category = {}
    for obj in dataset.objects:
        label = obj.label
        if label not in models_by_category:
            models_by_category[label] = []
        models_by_category[label].append(obj)
    
    # 显示统计
    print("\n" + "=" * 60)
    print(f"数据集统计: {pkl_file}")
    print("=" * 60)
    print(f"总模型数: {len(dataset)}")
    print(f"类别数: {len(models_by_category)}")
    
    if category:
        # 显示特定类别的模型
        if category in models_by_category:
            models = models_by_category[category]
            print(f"\n类别 '{category}' 的模型 ({len(models)} 个):")
            print("-" * 60)
            
            for i, obj in enumerate(models):
                print(f"\n[{i}] jid: {obj.model_jid}")
                if show_details:
                    print(f"    尺寸 (L×W×H): {obj.size[0]:.2f} × {obj.size[1]:.2f} × {obj.size[2]:.2f} m")
                    if hasattr(obj, 'openshape_vitg14_features') and obj.openshape_vitg14_features is not None:
                        print(f"    特征维度: {obj.openshape_vitg14_features.shape}")
        else:
            print(f"\n✗ 类别 '{category}' 不存在")
            print(f"可用类别: {sorted(models_by_category.keys())}")
    else:
        # 显示所有类别的统计
        print("\n所有类别:")
        print("-" * 60)
        for label in sorted(models_by_category.keys()):
            count = len(models_by_category[label])
            print(f"  {label:20s}: {count:4d} 个模型")
    
    return models_by_category


def generate_scene_template(models_by_category):
    """生成场景模板示例"""
    print("\n" + "=" * 60)
    print("场景模板示例（带手动指定的jid）:")
    print("=" * 60)
    
    template = """
# 在 custom_scene_with_visualization.py 中添加:

def bedroom_template_custom_jids():
    '''自定义卧室 - 手动指定模型jid'''
    objs = [8, 12, 12]  # double_bed, nightstand, nightstand
    edges = [
        [10, 6, 1],  # bed
        [1, 10, 10],  # nightstand_left
        [6, 10, 10],  # nightstand_right
    ]
    
    # 手动指定每个物体的模型jid
    model_jids = [
        'f43310eb-270b-49ec-aef9-11103921b224',  # 特定的床模型
        '2fd5c449-667b-418e-b53d-3596f954eab7',  # 特定的床头柜1
        '5e4c8a81-79dd-43c2-a117-f3b9bda9e560',  # 特定的床头柜2
    ]
    
    return objs, edges, model_jids
"""
    
    print(template)
    
    # 显示一些示例jid
    print("\n示例模型jid (从数据集中随机选择):")
    print("-" * 60)
    
    for label in ['double_bed', 'nightstand', 'wardrobe']:
        if label in models_by_category:
            models = models_by_category[label]
            print(f"\n{label}:")
            for i, obj in enumerate(models[:3]):  # 只显示前3个
                print(f"  [{i}] {obj.model_jid}")
                print(f"      尺寸: {obj.size[0]:.2f} × {obj.size[1]:.2f} × {obj.size[2]:.2f} m")


def search_models(pkl_file, label=None, min_size=None, max_size=None):
    """根据条件搜索模型"""
    with open(pkl_file, 'rb') as f:
        dataset = pickle.load(f)
    
    results = []
    for obj in dataset.objects:
        # 类别过滤
        if label and obj.label != label:
            continue
        
        # 尺寸过滤
        if min_size:
            if any(obj.size[i] < min_size[i] for i in range(3)):
                continue
        if max_size:
            if any(obj.size[i] > max_size[i] for i in range(3)):
                continue
        
        results.append(obj)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="浏览和选择3D物体模型"
    )
    
    parser.add_argument(
        "--room_type",
        type=str,
        default="bedroom",
        choices=["bedroom", "livingroom", "diningroom"],
        help="房间类型"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="显示特定类别的模型（如 double_bed, nightstand）"
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="显示详细信息"
    )
    parser.add_argument(
        "--template",
        action="store_true",
        help="生成场景模板示例"
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="搜索模式"
    )
    parser.add_argument(
        "--min_size",
        type=float,
        nargs=3,
        help="最小尺寸 (长 宽 高)"
    )
    parser.add_argument(
        "--max_size",
        type=float,
        nargs=3,
        help="最大尺寸 (长 宽 高)"
    )
    
    args = parser.parse_args()
    
    # 数据集文件
    pkl_files = {
        'bedroom': 'dataset/InstructScene/threed_future_model_bedroom.pkl',
        'livingroom': 'dataset/InstructScene/threed_future_model_livingroom.pkl',
        'diningroom': 'dataset/InstructScene/threed_future_model_diningroom.pkl',
    }
    
    pkl_file = pkl_files[args.room_type]
    
    if args.search:
        # 搜索模式
        results = search_models(
            pkl_file,
            label=args.category,
            min_size=args.min_size,
            max_size=args.max_size
        )
        print(f"\n找到 {len(results)} 个匹配的模型:")
        for i, obj in enumerate(results[:20]):  # 只显示前20个
            print(f"[{i}] {obj.label}: {obj.model_jid}")
            print(f"    尺寸: {obj.size[0]:.2f} × {obj.size[1]:.2f} × {obj.size[2]:.2f} m")
    else:
        # 列表模式
        models_by_category = list_models_by_category(
            pkl_file,
            category=args.category,
            show_details=args.details
        )
        
        if args.template:
            generate_scene_template(models_by_category)


if __name__ == "__main__":
    main()


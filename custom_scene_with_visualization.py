"""
带可视化的自定义场景生成脚本
- 可视化语义图 (Scene Graph)
- Blender 渲染 3D 场景

用法示例:
python custom_scene_with_visualization.py \
    --room_type bedroom \
    --template simple \
    --device 0 \
    --visualize
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from src.utils.util import load_config, load_checkpoints
from src.utils.visualize import (
    draw_scene_graph, 
    blender_render_scene, 
    export_scene,
    get_textured_objects
)
from src.models import model_from_config, ObjectFeatureVQVAE
from src.data import get_dataset_raw_and_encoded
from src.data.threed_future_dataset import ThreedFutureDataset
from diffusers.training_utils import EMAModel


# ========== 预定义的场景模板 ==========

def bedroom_template_simple():
    """简单卧室: 1床 + 2床头柜"""
    objs = [8, 12, 12]  # double_bed(8), nightstand(12), nightstand(12)
    edges = [
        [10, 1, 6],  # bed: none(10), left_of(1) nightstand1, right_of(6) nightstand2
        [6, 10, 10],  # nightstand1: right_of(6) bed, none(10), none(10)
        [1, 10, 10],  # nightstand2: left_of(1) bed, none(10), none(10)
    ]
    return objs, edges


def bedroom_template_standard():
    """标准卧室: 1床 + 2床头柜 + 1衣柜"""
    objs = [8, 12, 12, 20]  # double_bed(8), nightstand(12), nightstand(12), wardrobe(20)
    edges = [
        [10, 1, 6, 2],  # bed: none(10), left_of(1) nightstand1, right_of(6) nightstand2, in_front_of(2) wardrobe
        [6, 10, 10, 10],  # nightstand1: right_of(6) bed, none(10), none(10), none(10)
        [1, 10, 10, 10],  # nightstand2: left_of(1) bed, none(10), none(10), none(10)
        [7, 10, 10, 10],  # wardrobe: behind(7) bed, none(10), none(10), none(10)
    ]
    return objs, edges


def livingroom_template_simple():
    """简单客厅: 1沙发 + 1茶几"""
    objs = [4, 5]  # sofa, coffee_table
    edges = [
        [6, 3],  # sofa: none, behind
        [2, 6],  # coffee_table: front, none
    ]
    return objs, edges


def livingroom_template_standard():
    """标准客厅: 1沙发 + 1茶几 + 1电视柜"""
    objs = [4, 5, 8]  # sofa, coffee_table, tv_stand
    edges = [
        [6, 3, 6],  # sofa
        [2, 6, 3],  # coffee_table: front of sofa, behind tv_stand
        [6, 2, 6],  # tv_stand: front of coffee_table
    ]
    return objs, edges


TEMPLATES = {
    'bedroom': {
        'simple': bedroom_template_simple,
        'standard': bedroom_template_standard,
    },
    'livingroom': {
        'simple': livingroom_template_simple,
        'standard': livingroom_template_standard,
    }
}

OBJECT_NAMES = {
    'bedroom': {
        0: 'armchair', 1: 'bookshelf', 2: 'cabinet', 3: 'ceiling_lamp',
        4: 'chair', 5: 'children_cabinet', 6: 'coffee_table', 7: 'desk',
        8: 'double_bed', 9: 'dressing_chair', 10: 'dressing_table', 11: 'kids_bed',
        12: 'nightstand', 13: 'pendant_lamp', 14: 'shelf', 15: 'single_bed',
        16: 'sofa', 17: 'stool', 18: 'table', 19: 'tv_stand', 20: 'wardrobe'
    },
    'livingroom': {
        # TODO: 需要从实际数据集获取正确的ID映射
        4: 'sofa', 5: 'coffee_table', 6: 'table', 7: 'chair',
        8: 'tv_stand', 9: 'cabinet', 10: 'bookshelf'
    }
}

"""
数字ID	关系名称 (英文)	关系名称 (中文)	说明
0	above	上方	A 在 B 的上方
1	left of	左边	A 在 B 的左边
2	in front of	前面	A 在 B 的前面
3	closely left of	紧邻左边	A 紧邻 B 的左边（距离<1米）
4	closely in front of	紧邻前面	A 紧邻 B 的前面（距离<1米）
5	below	下方	A 在 B 的下方
6	right of	右边	A 在 B 的右边
7	behind	后面	A 在 B 的后面
8	closely right of	紧邻右边	A 紧邻 B 的右边（距离<1米）
9	closely behind	紧邻后面	A 紧邻 B 的后面（距离<1米）
10	none	无关系	用于邻接矩阵表示无关系
"""

RELATION_NAMES = ['left', 'right', 'front', 'behind', 'bigger', 'smaller', 'none']


def edges_matrix_to_triples(edges, n_objs):
    """将邻接矩阵转换为三元组列表 (subject, predicate, object)"""
    triples = []
    for i in range(n_objs):
        for j in range(n_objs):
            if i != j and edges[i][j] != 10:  # 10 是 'none' 关系
                triples.append([i, edges[i][j], j])
    return triples


def draw_custom_scene_graph(objs, triples, output_file, object_types, predicate_types):
    """自定义的语义图绘制函数（修复版）"""
    import tempfile
    
    lines = [
        'digraph{',
        'graph [size="5,3",ratio="compress",dpi="300",bgcolor="transparent"]',
        'rankdir=LR',
        'node [shape="box",style="rounded,filled",fontsize="48",color="none"]',
        'node [fillcolor="lightpink1"]',
    ]
    
    # 添加节点
    for i, obj_id in enumerate(objs):
        obj_name = object_types[obj_id]
        lines.append(f'{i} [label="{obj_name}"]')
    
    # 添加关系
    next_node_id = len(objs)
    lines.append('node [fillcolor="lightblue1"]')
    for s, p, o in triples:
        pred_name = predicate_types[p]
        lines += [
            f'{next_node_id} [label="{pred_name}"]',
            f'{s}->{next_node_id} [penwidth=6,arrowsize=1.5,weight=1.2]',
            f'{next_node_id}->{o} [penwidth=6,arrowsize=1.5,weight=1.2]'
        ]
        next_node_id += 1
    
    lines.append('}')
    
    # 写入临时文件
    ff, dot_filename = tempfile.mkstemp(suffix='.dot')
    with open(dot_filename, 'w') as f:
        f.write('\n'.join(lines))
    os.close(ff)
    
    # 执行 dot 命令
    output_format = os.path.splitext(output_file)[1][1:]
    cmd = f'dot -T{output_format} {dot_filename} > {output_file}'
    ret = os.system(cmd)
    
    # 删除临时文件
    os.remove(dot_filename)
    
    return ret == 0 and os.path.exists(output_file)


def generate_and_visualize_scene(args):
    """生成场景并可视化"""
    
    # 设置随机种子以保证可重复性
    if args.seed is not None:
        print(f"设置随机种子: {args.seed}")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
        print()
    
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}\n")
    
    # 加载场景模板
    if args.template not in TEMPLATES[args.room_type]:
        print(f"错误: 模板 '{args.template}' 不可用")
        return
    
    objs, edges = TEMPLATES[args.room_type][args.template]()
    n_objs = len(objs)
    
    # 打印场景信息
    print("=" * 60)
    print(f"场景配置: {args.room_type.upper()} - {args.template.upper()}")
    print("=" * 60)
    print(f"物体数量: {n_objs}\n")
    
    obj_names = OBJECT_NAMES[args.room_type]
    print("物体列表:")
    for i, obj_id in enumerate(objs):
        obj_name = obj_names.get(obj_id, f"unknown_{obj_id}")
        print(f"  [{i}] {obj_name}")
    print()
    
    # 生成随机物体特征
    objfeat_vq_indices = np.random.randint(0, 64, size=(n_objs, 4))
    
    # 配置文件路径
    config_file = f"configs/{args.room_type}_sg2sc_diffusion_objfeat.yaml"
    if not os.path.exists(config_file):
        print(f"错误: 配置文件不存在: {config_file}")
        return
    
    config = load_config(config_file)
    
    # 加载数据集
    print("加载数据集配置...")
    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=lambda s: s,
        split=["train", "val"]
    )
    
    # 保存目录
    sg2sc_tag = args.sg2sc_tag or f"{args.room_type}_sg2scdiffusion_objfeat"
    save_dir = f"{args.output_dir}/{sg2sc_tag}/custom_scenes/{args.room_type}_{args.template}"
    os.makedirs(save_dir, exist_ok=True)
    
    # ========== 1. 可视化语义图 ==========
    if args.draw_graph:
        print("\n绘制语义图...")
        triples = edges_matrix_to_triples(edges, n_objs)
        print(f"  物体数量: {n_objs}")
        print(f"  关系数量: {len(triples)}")
        
        if len(triples) == 0:
            print("  警告: 没有有效的关系，跳过语义图绘制")
        else:
            graph_output = os.path.join(save_dir, "scene_graph.png")
            try:
                success = draw_custom_scene_graph(
                    objs, triples, graph_output,
                    raw_dataset.object_types,
                    raw_dataset.predicate_types
                )
                if success:
                    print(f"  ✓ 语义图已保存到: {graph_output}")
                else:
                    print(f"  ✗ 语义图生成失败，请检查 GraphViz 是否正确安装")
                    print(f"    尝试运行: sudo apt-get install graphviz")
            except Exception as e:
                print(f"  ✗ 绘制语义图时出错: {e}")
                import traceback
                traceback.print_exc()
    
    # ========== 2. 加载模型并生成布局 ==========
    
    # 转换为张量
    objs_tensor = torch.LongTensor(objs).unsqueeze(0).to(device)
    edges_tensor = torch.LongTensor(edges).unsqueeze(0).to(device)
    objfeat_vq_indices_tensor = torch.LongTensor(objfeat_vq_indices).unsqueeze(0).to(device)
    obj_masks = torch.ones(1, n_objs, dtype=torch.long).to(device)
    
    # 加载 fVQ-VAE
    print("\n加载 fVQ-VAE 模型...")
    fvqvae_tag = args.fvqvae_tag or "threedfront_objfeat_vqvae"
    with open(f"{args.output_dir}/{fvqvae_tag}/objfeat_bounds.pkl", "rb") as f:
        kwargs = pickle.load(f)
    vqvae_model = ObjectFeatureVQVAE("openshape_vitg14", "gumbel", **kwargs)
    ckpt_path = f"{args.output_dir}/{fvqvae_tag}/checkpoints/epoch_01999.pth"
    vqvae_model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
    vqvae_model = vqvae_model.to(device)
    vqvae_model.eval()
    
    # 加载 Layout Decoder
    print("加载 Layout Decoder 模型...")
    model = model_from_config(
        config["model"],
        raw_dataset.n_object_types,
        raw_dataset.n_predicate_types
    ).to(device)
    
    ckpt_dir = f"{args.output_dir}/{sg2sc_tag}/checkpoints"
    ema_config = config["training"]["ema"]
    if ema_config["use_ema"]:
        ema_states = EMAModel(model.parameters())
        ema_states.to(device)
    else:
        ema_states = None
    
    load_epoch = load_checkpoints(model, ckpt_dir, ema_states, epoch=-1, device=device)
    
    if ema_states is not None:
        ema_states.copy_to(model.parameters())
    model.eval()
    
    # 生成场景
    print("\n开始生成 3D 场景布局...")
    with torch.no_grad():
        boxes_pred = model.generate_samples(
            objs_tensor,
            edges_tensor,
            objfeat_vq_indices_tensor,
            obj_masks,
            vqvae_model,
            cfg_scale=args.cfg_scale
        )
    
    # 解码物体特征
    B, N = objfeat_vq_indices_tensor.shape[:2]
    objfeats = vqvae_model.reconstruct_from_indices(
        objfeat_vq_indices_tensor.reshape(B*N, -1)
    ).reshape(B, N, -1)
    objfeats = (objfeats * obj_masks[..., None].float()).cpu().numpy()
    
    boxes_pred = boxes_pred.cpu()
    objs_cpu = objs_tensor.cpu()
    
    # 后处理
    bbox_params = {
        "class_labels": F.one_hot(objs_cpu, num_classes=raw_dataset.n_object_types+1).float(),
        "translations": boxes_pred[..., :3],
        "sizes": boxes_pred[..., 3:6],
        "angles": boxes_pred[..., 6:]
    }
    boxes = dataset.post_process(bbox_params)
    
    bbox_params_t = torch.cat([
        boxes["class_labels"],
        boxes["translations"],
        boxes["sizes"],
        boxes["angles"]
    ], dim=-1).numpy()
    
    # 打印结果
    print("\n" + "=" * 60)
    print("生成的场景布局:")
    print("=" * 60)
    for i in range(n_objs):
        obj_name = obj_names.get(objs[i], f"unknown_{objs[i]}")
        position = boxes["translations"][0, i].numpy()
        size = boxes["sizes"][0, i].numpy()
        angle = boxes["angles"][0, i].numpy()
        # 角度已经是弧度值（post_process 中已转换）
        angle_rad = angle.item() if angle.size == 1 else angle[0]
        angle_deg = angle_rad * 180 / np.pi
        
        print(f"\n[{i}] {obj_name.upper()}")
        print(f"    位置 (x, z): [{position[0]:6.3f}, {position[2]:6.3f}] m")
        print(f"    尺寸 (L×W×H): {size[0]:.2f} × {size[1]:.2f} × {size[2]:.2f} m")
        print(f"    旋转角度: {angle_deg:6.1f}°")
    print("\n" + "=" * 60)
    
    # 保存场景数据
    scene_data = {
        'room_type': args.room_type,
        'template': args.template,
        'objs': objs,
        'edges': edges,
        'objfeat_vq_indices': objfeat_vq_indices,
        'objfeats': objfeats[0],
        'boxes': boxes_pred[0].numpy(),
        'translations': boxes["translations"][0].numpy(),
        'sizes': boxes["sizes"][0].numpy(),
        'angles': boxes["angles"][0].numpy(),
    }
    
    scene_pkl = os.path.join(save_dir, "scene_data.pkl")
    with open(scene_pkl, 'wb') as f:
        pickle.dump(scene_data, f)
    print(f"\n✓ 场景数据已保存到: {scene_pkl}")
    
    # ========== 3. Blender 渲染 ==========
    if args.render:
        print("\n" + "=" * 60)
        print("开始 Blender 渲染...")
        print("=" * 60)
        
        # 加载 3D-FUTURE 物体数据集
        print("\n加载 3D-FUTURE 物体数据集...")
        
        # 优先使用pickle文件（更快，不需要完整数据集）
        path_to_pickled_models = config["data"].get("path_to_pickled_3d_futute_models")
        
        if path_to_pickled_models and os.path.exists(path_to_pickled_models):
            print(f"从pickle文件加载: {path_to_pickled_models}")
            with open(path_to_pickled_models, 'rb') as f:
                objects_dataset = pickle.load(f)
            print(f"✓ 已加载 {len(objects_dataset)} 个物体模型")
        else:
            # 备选：从完整数据集加载
            path_to_3d_future = config["data"].get("path_to_3d_future_dataset_directory")
            path_to_model_info = config["data"].get("path_to_model_info")
            path_to_3d_future_model_info = config["data"].get("path_to_3d_future_model_info")
            
            if not path_to_3d_future or not os.path.exists(path_to_3d_future):
                print(f"✗ 错误: 找不到3D物体数据")
                print(f"  - pickle文件: {path_to_pickled_models}")
                print(f"  - 数据集路径: {path_to_3d_future}")
                print(f"\n请确保配置文件中设置了:")
                print(f"  path_to_pickled_3d_futute_models: 'dataset/InstructScene/threed_future_model_bedroom.pkl'")
                return
            
            print(f"从完整数据集加载: {path_to_3d_future}")
            objects_dataset = ThreedFutureDataset(
                path_to_3d_future,
                path_to_model_info=path_to_model_info,
                path_to_3d_future_model_info=path_to_3d_future_model_info
            )
        
        # 检索物体并导出场景
        print("检索 3D 物体...")
        classes = np.array(dataset.object_types)
        
        trimesh_meshes, bbox_meshes, obj_classes, obj_sizes, obj_ids = get_textured_objects(
            bbox_params_t[0],
            objects_dataset,
            classes,
            obj_features=objfeats[0],
            objfeat_type="openshape_vitg14",
            with_cls=True,
            get_bbox_meshes=True,
            verbose=True
        )
        
        # 导出场景为 OBJ 文件
        scene_export_dir = os.path.join(save_dir, "scene_obj")
        os.makedirs(scene_export_dir, exist_ok=True)
        
        print(f"\n导出场景文件到: {scene_export_dir}")
        export_scene(scene_export_dir, trimesh_meshes, bbox_meshes)
        
        # 使用 Blender 渲染
        render_output_dir = os.path.join(save_dir, "renders")
        os.makedirs(render_output_dir, exist_ok=True)
        
        print(f"\nBlender 渲染中...")
        print(f"输出目录: {render_output_dir}")
        print(f"分辨率: {args.resolution}x{args.resolution}")
        print(f"视角数: {args.num_views}")
        
        try:
            blender_render_scene(
                scene_dir=scene_export_dir,
                output_dir=render_output_dir,
                output_suffix="",
                engine="CYCLES",
                top_down_view=args.top_view,
                num_images=args.num_views,
                resolution_x=args.resolution,
                resolution_y=args.resolution,
                cycle_samples=args.samples,
                verbose=args.verbose
            )
            print(f"\n✓ 渲染完成！图像已保存到: {render_output_dir}")
        except Exception as e:
            print(f"\n✗ Blender 渲染失败: {e}")
            print("请确保已安装 Blender 并正确配置路径")
    
    print("\n" + "=" * 60)
    print("全部完成!")
    print("=" * 60)
    print(f"\n输出目录: {save_dir}")
    if args.draw_graph:
        print(f"  - 语义图: scene_graph.png")
    print(f"  - 场景数据: scene_data.pkl")
    if args.render:
        print(f"  - 3D 模型: scene_obj/")
        print(f"  - 渲染图像: renders/")


def main():
    parser = argparse.ArgumentParser(
        description="自定义场景生成 + 可视化"
    )
    
    parser.add_argument(
        "--room_type",
        type=str,
        required=True,
        choices=["bedroom", "livingroom", "diningroom"],
        help="房间类型"
    )
    parser.add_argument(
        "--template",
        type=str,
        default="simple",
        choices=["simple", "standard", "full"],
        help="场景模板"
    )
    parser.add_argument(
        "--sg2sc_tag",
        type=str,
        default=None,
        help="Layout Decoder 实验标签"
    )
    parser.add_argument(
        "--fvqvae_tag",
        type=str,
        default=None,
        help="fVQ-VAE 实验标签"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out",
        help="输出目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU 设备 ID"
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.5,
        help="CFG 强度"
    )
    parser.add_argument(
        "--draw_graph",
        action="store_true",
        help="绘制语义图"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="使用 Blender 渲染场景"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="渲染分辨率"
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=4,
        help="渲染视角数量"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=32,
        help="Cycles 采样数"
    )
    parser.add_argument(
        "--top_view",
        action="store_true",
        help="俯视图"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（用于可重复生成）"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("InstructScene - 自定义场景生成 + 可视化")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  房间类型: {args.room_type}")
    print(f"  场景模板: {args.template}")
    print(f"  绘制语义图: {'是' if args.draw_graph else '否'}")
    print(f"  Blender 渲染: {'是' if args.render else '否'}")
    if args.render:
        print(f"  渲染分辨率: {args.resolution}x{args.resolution}")
        print(f"  视角数量: {args.num_views}")
    print()
    
    generate_and_visualize_scene(args)


if __name__ == "__main__":
    main()


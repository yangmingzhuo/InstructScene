"""
InstructScene Web Interface - Flask Backend API
提供场景图编辑和3D场景生成的REST API
"""

import sys
import os

# 获取项目根目录并添加到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 设置Blender路径
BLENDER_PATH = os.path.join(PROJECT_ROOT, "blender/blender-3.3.1-linux-x64/blender")
if os.path.exists(BLENDER_PATH):
    os.environ['BLENDER_PATH'] = BLENDER_PATH
    print(f"✓ Blender路径已设置: {BLENDER_PATH}")
else:
    print(f"⚠️  Blender未找到: {BLENDER_PATH}")

from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import traceback
from pathlib import Path
import base64
import time

from src.utils.util import load_config, load_checkpoints
from src.utils.visualize import get_textured_objects, export_scene, blender_render_scene
from src.models import model_from_config, ObjectFeatureVQVAE
from src.data import get_dataset_raw_and_encoded
from diffusers.training_utils import EMAModel

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
CORS(app)

# 全局变量存储已加载的模型
models_cache = {}

# 物体类型和关系类型
OBJECT_TYPES = {
    'bedroom': {
        0: 'armchair', 1: 'bookshelf', 2: 'cabinet', 3: 'ceiling_lamp',
        4: 'chair', 5: 'children_cabinet', 6: 'coffee_table', 7: 'desk',
        8: 'double_bed', 9: 'dressing_chair', 10: 'dressing_table', 11: 'kids_bed',
        12: 'nightstand', 13: 'pendant_lamp', 14: 'shelf', 15: 'single_bed',
        16: 'sofa', 17: 'stool', 18: 'table', 19: 'tv_stand', 20: 'wardrobe'
    }
}

RELATION_TYPES = {
    0: 'above',
    1: 'left of',
    2: 'in front of',
    3: 'closely left of',
    4: 'closely in front of',
    5: 'below',
    6: 'right of',
    7: 'behind',
    8: 'closely right of',
    9: 'closely behind',
    10: 'none'
}

RELATION_TYPES_ZH = {
    0: '上方',
    1: '左边',
    2: '前面',
    3: '紧邻左边',
    4: '紧邻前面',
    5: '下方',
    6: '右边',
    7: '后面',
    8: '紧邻右边',
    9: '紧邻后面',
    10: '无关系'
}


def fix_config_paths(config, base_dir):
    """将配置中的相对路径转换为绝对路径"""
    import copy
    config = copy.deepcopy(config)
    
    # 需要修正的路径字段（绝对路径）
    path_fields = [
        'path_to_bounds',
        'path_to_pickled_3d_futute_models',
        'path_to_3d_future_dataset_directory',
        'path_to_model_info',
        'path_to_3d_future_model_info',
        'path_to_invalid_scene_ids',
        'path_to_invalid_bbox_jids',
        'path_to_floor_plan_textures',
        'annotation_file',
        'dataset_directory',  # 数据集目录
    ]
    
    def process_dict(d):
        """递归处理字典中的路径"""
        for key, value in d.items():
            if isinstance(value, dict):
                process_dict(value)
            elif isinstance(value, str):
                # 检查是否是路径字段
                if key in path_fields:
                    # 如果是相对路径，转换为绝对路径
                    if not os.path.isabs(value):
                        d[key] = os.path.join(base_dir, value)
    
    process_dict(config)
    return config


def load_models(room_type, device='cuda:0'):
    """加载模型（带缓存）"""
    cache_key = f"{room_type}_{device}"
    if cache_key in models_cache:
        return models_cache[cache_key]
    
    print(f"Loading models for {room_type} on {device}...")
    
    config_file = os.path.join(PROJECT_ROOT, f"configs/{room_type}_sg2sc_diffusion_objfeat.yaml")
    config = load_config(config_file)
    
    # 修正配置中的相对路径为绝对路径
    config = fix_config_paths(config, PROJECT_ROOT)
    
    # 加载数据集
    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=lambda s: s,
        split=["train", "val"]
    )
    
    # 加载 fVQ-VAE
    fvqvae_tag = "threedfront_objfeat_vqvae"
    objfeat_bounds_path = os.path.join(PROJECT_ROOT, f"out/{fvqvae_tag}/objfeat_bounds.pkl")
    with open(objfeat_bounds_path, "rb") as f:
        kwargs = pickle.load(f)
    vqvae_model = ObjectFeatureVQVAE("openshape_vitg14", "gumbel", **kwargs)
    ckpt_path = os.path.join(PROJECT_ROOT, f"out/{fvqvae_tag}/checkpoints/epoch_01999.pth")
    vqvae_model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
    vqvae_model = vqvae_model.to(device)
    vqvae_model.eval()
    
    # 加载 Layout Decoder
    sg2sc_tag = f"{room_type}_sg2scdiffusion_objfeat"
    model = model_from_config(
        config["model"],
        raw_dataset.n_object_types,
        raw_dataset.n_predicate_types
    ).to(device)
    
    ckpt_dir = os.path.join(PROJECT_ROOT, f"out/{sg2sc_tag}/checkpoints")
    ema_config = config["training"]["ema"]
    if ema_config["use_ema"]:
        ema_states = EMAModel(model.parameters())
        ema_states.to(device)
    else:
        ema_states = None
    
    load_checkpoints(model, ckpt_dir, ema_states, epoch=-1, device=device)
    
    if ema_states is not None:
        ema_states.copy_to(model.parameters())
    model.eval()
    
    models_dict = {
        'model': model,
        'vqvae_model': vqvae_model,
        'raw_dataset': raw_dataset,
        'dataset': dataset,
        'config': config
    }
    
    models_cache[cache_key] = models_dict
    print(f"Models loaded successfully!")
    return models_dict


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/object_types/<room_type>', methods=['GET'])
def get_object_types(room_type):
    """获取物体类型列表"""
    if room_type not in OBJECT_TYPES:
        return jsonify({'error': 'Invalid room type'}), 400
    
    obj_types = [{'id': k, 'name': v} for k, v in OBJECT_TYPES[room_type].items()]
    return jsonify({'object_types': obj_types})


@app.route('/api/relation_types', methods=['GET'])
def get_relation_types():
    """获取关系类型列表"""
    rel_types = [
        {
            'id': k, 
            'name': v,
            'name_zh': RELATION_TYPES_ZH[k]
        } 
        for k, v in RELATION_TYPES.items()
    ]
    return jsonify({'relation_types': rel_types})


@app.route('/api/generate_scene', methods=['POST'])
def generate_scene():
    """生成3D场景布局"""
    try:
        data = request.json
        room_type = data.get('room_type', 'bedroom')
        objs = data.get('objects', [])
        edges = data.get('edges', [])
        cfg_scale = data.get('cfg_scale', 1.5)
        seed = data.get('seed', None)
        device = data.get('device', 'cuda:0')
        
        # 设置随机种子
        if seed is not None:
            print(f"\n🎲 设置随机种子: {seed}")
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        
        if not objs:
            return jsonify({'error': 'No objects provided'}), 400
        
        n_objs = len(objs)
        
        # 构建邻接矩阵（用10表示none关系）
        edges_matrix = [[10 for _ in range(n_objs)] for _ in range(n_objs)]
        for edge in edges:
            src = edge['source']
            tgt = edge['target']
            rel = edge['relation']
            if 0 <= src < n_objs and 0 <= tgt < n_objs:
                edges_matrix[src][tgt] = rel
        
        print(f"edges_matrix: {edges_matrix}")
        
        # 生成随机物体特征
        objfeat_vq_indices = np.random.randint(0, 64, size=(n_objs, 4))
        
        # 加载模型
        models = load_models(room_type, device)
        
        # 转换为张量
        objs_tensor = torch.LongTensor(objs).unsqueeze(0).to(device)
        edges_tensor = torch.LongTensor(edges_matrix).unsqueeze(0).to(device)
        objfeat_vq_indices_tensor = torch.LongTensor(objfeat_vq_indices).unsqueeze(0).to(device)
        obj_masks = torch.ones(1, n_objs, dtype=torch.long).to(device)
        
        # 生成场景布局
        with torch.no_grad():
            boxes_pred = models['model'].generate_samples(
                objs_tensor,
                edges_tensor,
                objfeat_vq_indices_tensor,
                obj_masks,
                models['vqvae_model'],
                cfg_scale=cfg_scale
            )
        
        # 解码物体特征
        B, N = objfeat_vq_indices_tensor.shape[:2]
        objfeats = models['vqvae_model'].reconstruct_from_indices(
            objfeat_vq_indices_tensor.reshape(B*N, -1)
        ).reshape(B, N, -1)
        objfeats_np = (objfeats * obj_masks[..., None].float()).cpu().numpy()
        
        boxes_pred = boxes_pred.cpu()
        objs_cpu = objs_tensor.cpu()
        
        # 后处理
        bbox_params = {
            "class_labels": F.one_hot(objs_cpu, num_classes=models['raw_dataset'].n_object_types+1).float(),
            "translations": boxes_pred[..., :3],
            "sizes": boxes_pred[..., 3:6],
            "angles": boxes_pred[..., 6:]
        }
        boxes = models['dataset'].post_process(bbox_params)
        
        # 检索3D模型的jid（参考 src/utils/visualize.py 的实现）
        obj_jids = []
        try:
            # 加载3D物体数据集
            path_to_pickled_models = models['config']['data'].get('path_to_pickled_3d_futute_models')
            if path_to_pickled_models and os.path.exists(path_to_pickled_models):
                print(f"📦 加载模型数据集: {path_to_pickled_models}")
                with open(path_to_pickled_models, 'rb') as f:
                    objects_dataset = pickle.load(f)
                
                # 修正数据集中对象的路径（从相对路径转为绝对路径）
                print(f"🔧 修正 {len(objects_dataset.objects)} 个模型的路径...")
                for obj in objects_dataset.objects:
                    if hasattr(obj, 'path_to_models') and not os.path.isabs(obj.path_to_models):
                        obj.path_to_models = os.path.join(PROJECT_ROOT, obj.path_to_models)
                print("✓ 路径修正完成")
                
                # 检索每个物体的jid
                classes = np.array(models['dataset'].object_types)
                print(f"\n🔍 开始检索 {n_objs} 个物体的3D模型...")
                
                for j in range(n_objs):
                    query_label = classes[objs[j]]
                    query_size = boxes["sizes"][0, j].numpy()
                    query_feature = objfeats_np[0, j] if objfeats_np is not None else None
                    
                    try:
                        # 使用特征进行检索（如果有特征的话）
                        if query_feature is not None:
                            furniture, select_gap = objects_dataset.get_closest_furniture_to_objfeat_and_size(
                                query_label, query_size, query_feature, "openshape_vitg14"
                            )
                        else:
                            # 仅使用尺寸检索
                            furniture, select_gap = objects_dataset.get_closest_furniture_to_box(
                                query_label, query_size
                            )
                        
                        obj_jids.append(furniture.model_jid)
                        print(f"  [{j}] {query_label}: {furniture.model_jid} (gap={select_gap:.4f})")
                    except Exception as e:
                        print(f"  [{j}] ⚠️  {query_label} 检索失败: {e}")
                        obj_jids.append(None)
                
                success_count = len([j for j in obj_jids if j])
                print(f"\n✓ 成功检索 {success_count} / {n_objs} 个模型")
            else:
                print(f"⚠️  模型数据集文件不存在: {path_to_pickled_models}")
                obj_jids = [None] * n_objs
        except Exception as e:
            print(f"❌ 模型检索失败: {e}")
            import traceback
            traceback.print_exc()
            obj_jids = [None] * n_objs
        
        # 准备返回结果
        result = {
            'objects': []
        }
        
        # 打印场景生成结果
        print("\n" + "=" * 80)
        print("🎨 场景生成完成！")
        print("=" * 80)
        
        for i in range(n_objs):
            obj_name = OBJECT_TYPES[room_type].get(objs[i], f"unknown_{objs[i]}")
            position = boxes["translations"][0, i].numpy().tolist()
            size = boxes["sizes"][0, i].numpy().tolist()
            angle = boxes["angles"][0, i].item()
            angle_deg = angle * 180 / np.pi
            jid = obj_jids[i] if i < len(obj_jids) else None
            
            # 详细输出每个物体的信息
            print(f"\n[{i}] {obj_name.upper()}")
            print(f"    位置 (x,y,z): {position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}")
            print(f"    尺寸 (L×W×H): {size[0]:.2f} × {size[1]:.2f} × {size[2]:.2f} m")
            print(f"    旋转角度: {angle_deg:.1f}° (弧度: {angle:.4f})")
            print(f"    模型ID: {jid if jid else '无'}")
            
            # 计算边界框
            half_size = [s/2 for s in size]
            print(f"    边界框:")
            print(f"      X范围: [{position[0] - half_size[0]:.3f}, {position[0] + half_size[0]:.3f}]")
            print(f"      Y范围: [{position[1] - half_size[1]:.3f}, {position[1] + half_size[1]:.3f}]")
            print(f"      Z范围: [{position[2] - half_size[2]:.3f}, {position[2] + half_size[2]:.3f}]")
            
            result['objects'].append({
                'id': i,
                'type_id': objs[i],
                'type_name': obj_name,
                'position': position,
                'size': size,
                'angle': angle,
                'angle_deg': angle_deg,
                'jid': jid
            })
        
        print("\n" + "=" * 80)
        
        return jsonify(result)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/<jid>/<filename>')
def serve_model_file(jid, filename):
    """提供3D模型文件"""
    try:
        model_dir = os.path.join(PROJECT_ROOT, f"dataset/3D-FRONT/3D-FUTURE-model/{jid}")
        if not os.path.exists(model_dir):
            return jsonify({'error': 'Model not found'}), 404
        
        file_path = os.path.join(model_dir, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_from_directory(model_dir, filename)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/exported_scenes')
def list_exported_scenes():
    """列出所有已导出的Blender场景"""
    try:
        scenes = []
        output_dir = os.path.join(PROJECT_ROOT, "out/bedroom_sg2scdiffusion_objfeat/custom_scenes")
        
        if os.path.exists(output_dir):
            for scene_name in os.listdir(output_dir):
                scene_path = os.path.join(output_dir, scene_name)
                scene_obj_dir = os.path.join(scene_path, "scene_obj")
                
                if os.path.exists(scene_obj_dir):
                    # 检查是否有OBJ文件
                    obj_files = [f for f in os.listdir(scene_obj_dir) if f.endswith('.obj')]
                    if obj_files:
                        scenes.append({
                            'name': scene_name,
                            'path': f"/api/scene_obj/{scene_name}",
                            'obj_count': len(obj_files)
                        })
        
        return jsonify({'scenes': scenes})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/scene_obj/<scene_name>')
def get_scene_obj_list(scene_name):
    """获取场景的所有文件列表（OBJ, MTL, 纹理等）"""
    try:
        scene_obj_dir = os.path.join(
            PROJECT_ROOT, 
            f"out/bedroom_sg2scdiffusion_objfeat/custom_scenes/{scene_name}/scene_obj"
        )
        
        if not os.path.exists(scene_obj_dir):
            return jsonify({'error': 'Scene not found'}), 404
        
        files = []
        for filename in os.listdir(scene_obj_dir):
            # 包含所有文件：OBJ, MTL, 纹理图片等
            files.append({
                'name': filename,
                'url': f"/api/scene_obj/{scene_name}/{filename}"
            })
        
        return jsonify({'files': files, 'base_url': f"/api/scene_obj/{scene_name}"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/scene_obj/<scene_name>/<filename>')
def serve_scene_obj_file(scene_name, filename):
    """提供场景的OBJ/MTL/纹理文件"""
    try:
        scene_obj_dir = os.path.join(
            PROJECT_ROOT, 
            f"out/bedroom_sg2scdiffusion_objfeat/custom_scenes/{scene_name}/scene_obj"
        )
        
        if not os.path.exists(scene_obj_dir):
            return jsonify({'error': 'Scene not found'}), 404
        
        file_path = os.path.join(scene_obj_dir, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_from_directory(scene_obj_dir, filename)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/render_scene', methods=['POST'])
def render_scene():
    """渲染3D场景（完整版，包括Blender导出和渲染）"""
    try:
        # 用于记录生成过程
        process_log = []
        
        data = request.json
        room_type = data.get('room_type', 'bedroom')
        objs = data.get('objects', [])
        edges = data.get('edges', [])
        cfg_scale = data.get('cfg_scale', 1.5)
        seed = data.get('seed', None)
        scene_name = data.get('scene_name', f'web_{int(time.time())}')
        device = data.get('device', 'cuda:0')
        
        # 设置随机种子
        if seed is not None:
            print(f"\n🎲 设置随机种子: {seed}")
            process_log.append({
                'type': 'info',
                'title': '设置随机种子',
                'content': f'随机种子已设置为: {seed}\n确保生成结果可重现'
            })
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        
        if not objs:
            return jsonify({'error': 'No objects provided'}), 400
        
        n_objs = len(objs)
        process_log.append({
            'type': 'step',
            'title': '加载AI模型',
            'content': f'正在加载{room_type}场景生成模型...\n包括扩散模型和VQ-VAE特征编码器'
        })
        
        # 构建邻接矩阵
        edges_matrix = [[10 for _ in range(n_objs)] for _ in range(n_objs)]
        for edge in edges:
            src = edge['source']
            tgt = edge['target']
            rel = edge['relation']
            if 0 <= src < n_objs and 0 <= tgt < n_objs:
                edges_matrix[src][tgt] = rel
        
        # 生成随机物体特征
        objfeat_vq_indices = np.random.randint(0, 64, size=(n_objs, 4))
        
        # 加载模型
        models = load_models(room_type, device)
        
        # 转换为张量
        objs_tensor = torch.LongTensor(objs).unsqueeze(0).to(device)
        edges_tensor = torch.LongTensor(edges_matrix).unsqueeze(0).to(device)
        objfeat_vq_indices_tensor = torch.LongTensor(objfeat_vq_indices).unsqueeze(0).to(device)
        obj_masks = torch.ones(1, n_objs, dtype=torch.long).to(device)
        
        # 生成场景布局
        with torch.no_grad():
            boxes_pred = models['model'].generate_samples(
                objs_tensor,
                edges_tensor,
                objfeat_vq_indices_tensor,
                obj_masks,
                models['vqvae_model'],
                cfg_scale=cfg_scale
            )
        
        # 解码物体特征
        B, N = objfeat_vq_indices_tensor.shape[:2]
        objfeats = models['vqvae_model'].reconstruct_from_indices(
            objfeat_vq_indices_tensor.reshape(B*N, -1)
        ).reshape(B, N, -1)
        objfeats_np = (objfeats * obj_masks[..., None].float()).cpu().numpy()
        
        boxes_pred = boxes_pred.cpu()
        objs_cpu = objs_tensor.cpu()
        
        # 后处理
        bbox_params = {
            "class_labels": F.one_hot(objs_cpu, num_classes=models['raw_dataset'].n_object_types+1).float(),
            "translations": boxes_pred[..., :3],
            "sizes": boxes_pred[..., 3:6],
            "angles": boxes_pred[..., 6:]
        }
        boxes = models['dataset'].post_process(bbox_params)
        
        # 构建bbox_params_t用于导出
        bbox_params_t = torch.cat([
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"]
        ], dim=-1).numpy()
        
        print("\n" + "=" * 80)
        print("🎨 开始Blender导出和渲染流程...")
        print("=" * 80)
        
        # 加载3D模型数据集
        path_to_pickled_models = models['config']['data'].get('path_to_pickled_3d_futute_models')
        if not path_to_pickled_models or not os.path.exists(path_to_pickled_models):
            return jsonify({'error': '3D模型数据集不存在'}), 500
        
        with open(path_to_pickled_models, 'rb') as f:
            objects_dataset = pickle.load(f)
        
        # 修正路径
        for obj in objects_dataset.objects:
            if hasattr(obj, 'path_to_models') and not os.path.isabs(obj.path_to_models):
                obj.path_to_models = os.path.join(PROJECT_ROOT, obj.path_to_models)
        
        # 导入可视化函数
        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
        from utils.visualize import get_textured_objects, export_scene, blender_render_scene
        
        # 检索物体（需要在项目根目录执行，因为visualize.py使用相对路径）
        process_log.append({
            'type': 'step',
            'title': '3D模型检索',
            'content': f'从数据库中检索{n_objs}个最匹配的3D模型...\n使用OpenShape特征匹配算法'
        })
        
        current_dir = os.getcwd()
        try:
            os.chdir(PROJECT_ROOT)
            classes = np.array(models['dataset'].object_types)
            trimesh_meshes, bbox_meshes, obj_classes, obj_sizes, obj_ids = get_textured_objects(
                bbox_params_t[0],
                objects_dataset,
                classes,
                obj_features=objfeats_np[0],
                objfeat_type="openshape_vitg14",
                with_cls=True,
                get_bbox_meshes=True,
                verbose=True
            )
        finally:
            os.chdir(current_dir)
        
        process_log.append({
            'type': 'success',
            'title': '模型检索完成',
            'content': f'成功检索{len([x for x in obj_ids if x])}个3D模型\n每个模型都包含完整的几何和纹理'
        })
        
        # 导出场景
        save_dir = os.path.join(PROJECT_ROOT, f"out/bedroom_sg2scdiffusion_objfeat/custom_scenes/{scene_name}")
        scene_export_dir = os.path.join(save_dir, "scene_obj")
        os.makedirs(scene_export_dir, exist_ok=True)
        
        process_log.append({
            'type': 'step',
            'title': '导出OBJ文件',
            'content': f'正在导出场景到{scene_export_dir}'
        })
        
        print(f"\n📦 导出场景到: {scene_export_dir}")
        export_scene(scene_export_dir, trimesh_meshes, bbox_meshes)
        
        process_log.append({
            'type': 'success',
            'title': '文件导出完成',
            'content': f'已生成{len(trimesh_meshes)}个OBJ文件及对应的MTL和纹理文件'
        })
        
        # 使用Blender导出合并的场景（包含完整的材质和纹理）
        print(f"\n🎨 使用Blender处理并导出合并场景...")
        process_log.append({
            'type': 'step',
            'title': 'Blender后处理',
            'content': '使用Blender合并所有模型并导出统一场景\n自动复制所有纹理文件'
        })
        
        try:
            blender_render_scene(
                scene_dir=scene_export_dir,
                output_dir=scene_export_dir,  # 输出到同一目录
                output_suffix="",
                engine="CYCLES",
                export_merged=True,  # 导出合并场景
                skip_normalize=True,  # 不进行normalize处理，保持原始输出
                verbose=True
            )
            print(f"✓ Blender导出完成: scene_merged.obj (原始坐标)")
            process_log.append({
                'type': 'success',
                'title': 'Blender处理完成',
                'content': '生成scene_merged.obj\n包含所有物体的合并模型和完整材质'
            })
        except Exception as e:
            print(f"⚠️  Blender导出失败: {e}")
            traceback.print_exc()
            process_log.append({
                'type': 'warning',
                'title': 'Blender导出失败',
                'content': f'将使用独立的OBJ文件\n错误: {str(e)[:100]}'
            })
        
        # 可选：渲染图片
        render_images = data.get('render_images', False)
        if render_images:
            render_output_dir = os.path.join(save_dir, "renders")
            os.makedirs(render_output_dir, exist_ok=True)
            
            print(f"\n🎬 Blender渲染图片中...")
            try:
                blender_render_scene(
                    scene_dir=scene_export_dir,
                    output_dir=render_output_dir,
                    output_suffix="",
                    engine="CYCLES",
                    top_down_view=False,
                    num_images=4,
                    resolution_x=512,
                    resolution_y=512,
                    cycle_samples=32,
                    verbose=False
                )
                print(f"✓ 渲染完成")
            except Exception as e:
                print(f"⚠️  渲染失败: {e}")
        
        print("=" * 80)
        
        # 添加最终日志
        process_log.append({
            'type': 'success',
            'title': 'Blender导出完成',
            'content': f'场景已导出到: {scene_name}/scene_obj/\n包含OBJ文件和纹理'
        })
        
        return jsonify({
            'success': True,
            'scene_name': scene_name,
            'scene_url': f'/api/scene_obj/{scene_name}',
            'export_dir': scene_export_dir,
            'process_log': process_log
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/templates/<room_type>', methods=['GET'])
def get_templates(room_type):
    """获取预定义模板"""
    templates = {
        'bedroom': {
            'simple': {
                'name': '简单卧室',
                'description': '1床 + 2床头柜',
                'objects': [8, 12, 12],
                'edges': [
                    {'source': 0, 'target': 1, 'relation': 1},  # bed left_of nightstand1
                    {'source': 0, 'target': 2, 'relation': 6},  # bed right_of nightstand2
                    {'source': 1, 'target': 0, 'relation': 6},  # nightstand1 right_of bed
                    {'source': 2, 'target': 0, 'relation': 1},  # nightstand2 left_of bed
                ]
            },
            'standard': {
                'name': '标准卧室',
                'description': '1床 + 2床头柜 + 1衣柜',
                'objects': [8, 12, 12, 20],
                'edges': [
                    {'source': 0, 'target': 1, 'relation': 1},  # bed left_of nightstand1
                    {'source': 0, 'target': 2, 'relation': 6},  # bed right_of nightstand2
                    {'source': 0, 'target': 3, 'relation': 2},  # bed in_front_of wardrobe
                    {'source': 1, 'target': 0, 'relation': 6},  # nightstand1 right_of bed
                    {'source': 2, 'target': 0, 'relation': 1},  # nightstand2 left_of bed
                    {'source': 3, 'target': 0, 'relation': 7},  # wardrobe behind bed
                ]
            },
            'full': {
                'name': '完整卧室',
                'description': '1床 + 2床头柜 + 1衣柜 + 1书桌 + 1椅子',
                'objects': [8, 12, 12, 20, 7, 4],  # bed, 2 nightstands, wardrobe, desk, chair
                'edges': [
                    {'source': 0, 'target': 1, 'relation': 1},  # bed left_of nightstand1
                    {'source': 0, 'target': 2, 'relation': 6},  # bed right_of nightstand2
                    {'source': 0, 'target': 3, 'relation': 2},  # bed in_front_of wardrobe
                    {'source': 1, 'target': 0, 'relation': 6},  # nightstand1 right_of bed
                    {'source': 2, 'target': 0, 'relation': 1},  # nightstand2 left_of bed
                    {'source': 3, 'target': 0, 'relation': 7},  # wardrobe behind bed
                    {'source': 4, 'target': 3, 'relation': 1},  # desk left_of wardrobe
                    {'source': 5, 'target': 4, 'relation': 2},  # chair in_front_of desk
                ]
            },
            'study': {
                'name': '学习卧室',
                'description': '1床 + 1床头柜 + 1书桌 + 1椅子 + 1书架',
                'objects': [15, 12, 7, 4, 1],  # single_bed, nightstand, desk, chair, bookshelf
                'edges': [
                    {'source': 0, 'target': 1, 'relation': 6},  # bed right_of nightstand
                    {'source': 1, 'target': 0, 'relation': 1},  # nightstand left_of bed
                    {'source': 2, 'target': 0, 'relation': 1},  # desk left_of bed
                    {'source': 3, 'target': 2, 'relation': 2},  # chair in_front_of desk
                    {'source': 4, 'target': 2, 'relation': 6},  # bookshelf right_of desk
                ]
            },
            'luxury': {
                'name': '豪华卧室',
                'description': '1床 + 2床头柜 + 1衣柜 + 1梳妆台 + 1梳妆椅 + 1扶手椅',
                'objects': [8, 12, 12, 20, 10, 9, 0],  # bed, 2 nightstands, wardrobe, dressing_table, dressing_chair, armchair
                'edges': [
                    {'source': 0, 'target': 1, 'relation': 1},  # bed left_of nightstand1
                    {'source': 0, 'target': 2, 'relation': 6},  # bed right_of nightstand2
                    {'source': 0, 'target': 3, 'relation': 2},  # bed in_front_of wardrobe
                    {'source': 1, 'target': 0, 'relation': 6},  # nightstand1 right_of bed
                    {'source': 2, 'target': 0, 'relation': 1},  # nightstand2 left_of bed
                    {'source': 3, 'target': 0, 'relation': 7},  # wardrobe behind bed
                    {'source': 4, 'target': 3, 'relation': 1},  # dressing_table left_of wardrobe
                    {'source': 5, 'target': 4, 'relation': 2},  # dressing_chair in_front_of dressing_table
                    {'source': 6, 'target': 0, 'relation': 1},  # armchair left_of bed
                ]
            }
        }
    }
    
    if room_type not in templates:
        return jsonify({'templates': {}})
    
    return jsonify({'templates': templates[room_type]})


if __name__ == '__main__':
    # 创建必要的目录
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(templates_dir, exist_ok=True)
    
    print("=" * 60)
    print("InstructScene Web Interface")
    print("=" * 60)
    print(f"项目根目录: {PROJECT_ROOT}")
    print("Starting server at http://localhost:6006")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=6006, debug=True)


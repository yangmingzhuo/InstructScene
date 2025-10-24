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
    
    # 加载对象特征边界值
    exp_dir = os.path.join("out", args.vqvae_tag)
    objfeat_bounds_path = os.path.join(exp_dir, "objfeat_bounds.pkl")
    if not os.path.exists(objfeat_bounds_path):
        raise FileNotFoundError(
            f"找不到对象特征边界文件: {objfeat_bounds_path}\n"
            f"这个文件应该在训练 VQ-VAE 时生成。"
        )
    
    with open(objfeat_bounds_path, "rb") as f:
        objfeat_bounds = pickle.load(f)
    print(f"加载特征边界: min={objfeat_bounds['objfeat_min']:.4f}, max={objfeat_bounds['objfeat_max']:.4f}")
    
    # 创建模型时传入边界参数
    vqvae_model = model_from_config(config["model"], **objfeat_bounds)
    vqvae_model = vqvae_model.to(device)
    vqvae_model.eval()
    
    # 加载检查点
    ckpt_path = os.path.join(exp_dir, "checkpoints")
    loaded_epoch = load_checkpoints(vqvae_model, ckpt_path, None, epoch=args.vqvae_epoch, device=device)
    print(f"已加载 checkpoint (epoch {loaded_epoch})\n")
    
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
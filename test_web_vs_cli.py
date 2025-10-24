#!/usr/bin/env python
"""对比Web界面和命令行脚本的输出"""

import requests
import pickle
import numpy as np

print("=" * 80)
print("Web界面 vs 命令行脚本 - 输出对比测试")
print("=" * 80)

# 1. 从命令行脚本加载数据
print("\n1️⃣  加载命令行脚本生成的场景数据...")
pkl_path = "out/bedroom_sg2scdiffusion_objfeat/custom_scenes/bedroom_simple/scene_data.pkl"
with open(pkl_path, 'rb') as f:
    cli_data = pickle.load(f)

print(f"✓ 已加载: {pkl_path}")
print(f"  物体数量: {len(cli_data['objs'])}")
print(f"  模板: {cli_data['template']}")

# 2. 调用Web API生成场景
print("\n2️⃣  调用Web API生成场景...")
url = "http://localhost:6006/api/generate_scene"
objs_list = cli_data['objs'] if isinstance(cli_data['objs'], list) else cli_data['objs'].tolist()
payload = {
    "room_type": "bedroom",
    "objects": objs_list,
    "edges": [],  # 构建边列表
    "cfg_scale": 1.5,
    "seed": 42
}

# 构建边列表
edges_matrix = cli_data['edges']
n_objs = len(cli_data['objs'])
for i in range(n_objs):
    for j in range(n_objs):
        if i != j and edges_matrix[i][j] != 10:
            payload["edges"].append({
                "source": i,
                "target": j,
                "relation": int(edges_matrix[i][j])
            })

print(f"  物体: {payload['objects']}")
print(f"  边数量: {len(payload['edges'])}")
print(f"  种子: {payload['seed']}")

try:
    response = requests.post(url, json=payload, timeout=120)
    if response.status_code == 200:
        web_data = response.json()
        print(f"✓ API调用成功")
        print(f"  返回物体数量: {len(web_data['objects'])}")
    else:
        print(f"✗ API调用失败: {response.status_code}")
        print(response.text)
        exit(1)
except Exception as e:
    print(f"✗ 连接失败: {e}")
    print("请确保Web服务器正在运行: http://localhost:6006")
    exit(1)

# 3. 对比数据
print("\n3️⃣  对比数据...")
print("=" * 80)
print(f"{'项目':<15} {'命令行脚本':<40} {'Web界面':<40}")
print("=" * 80)

all_match = True
for i in range(n_objs):
    cli_pos = cli_data['translations'][i]
    cli_size = cli_data['sizes'][i]
    cli_angle = cli_data['angles'][i][0]
    
    web_obj = web_data['objects'][i]
    web_pos = np.array(web_obj['position'])
    web_size = np.array(web_obj['size'])
    web_angle = web_obj['angle']
    
    print(f"\n物体 [{i}]")
    
    # 对比位置
    pos_diff = np.linalg.norm(cli_pos - web_pos)
    pos_match = pos_diff < 0.001
    all_match = all_match and pos_match
    print(f"  位置 (x,y,z):  {cli_pos}  |  {web_pos}")
    if not pos_match:
        print(f"    ⚠️  差异: {pos_diff:.6f}米")
    
    # 对比尺寸
    size_diff = np.linalg.norm(cli_size - web_size)
    size_match = size_diff < 0.001
    all_match = all_match and size_match
    print(f"  尺寸 (L×W×H): {cli_size}  |  {web_size}")
    if not size_match:
        print(f"    ⚠️  差异: {size_diff:.6f}米")
    
    # 对比角度
    angle_diff = abs(cli_angle - web_angle)
    angle_match = angle_diff < 0.001
    all_match = all_match and angle_match
    print(f"  角度 (rad):    {cli_angle:.4f}                               |  {web_angle:.4f}")
    if not angle_match:
        print(f"    ⚠️  差异: {angle_diff:.6f}弧度 ({angle_diff * 180 / np.pi:.3f}度)")
    
    # 对比JID
    cli_jid = "（命令行未实现JID检索）"
    web_jid = web_obj.get('jid', 'None')
    print(f"  模型ID:       {cli_jid:<36} |  {web_jid}")

print("\n" + "=" * 80)
if all_match:
    print("✅ 结论: Web界面和命令行脚本输出完全一致！")
    print("   问题不在后处理或模型输入，而是在3D可视化的坐标系统。")
else:
    print("❌ 结论: Web界面和命令行脚本输出存在差异！")
    print("   需要检查后处理或模型输入逻辑。")

print("\n💡 提示:")
print("   - adjustSceneToGround() 函数会调整Y坐标以对齐地面")
print("   - 训练数据的Y范围是 [0.045, 3.625]，不是从0开始")
print("   - 物体悬空是正常的，因为模型输出的是中心位置")
print("=" * 80)


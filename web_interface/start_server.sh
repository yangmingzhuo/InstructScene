#!/bin/bash

# InstructScene Web Interface 启动脚本

echo "=========================================="
echo "InstructScene Web Interface"
echo "=========================================="
echo ""

# 检查是否在正确的目录
if [ ! -f "app.py" ]; then
    echo "❌ 错误: 请在 web_interface 目录下运行此脚本"
    exit 1
fi

# 切换到项目根目录
cd ..

# 检查依赖
echo "📦 检查依赖..."
pip list | grep -q flask || {
    echo "⚠️  Flask 未安装，正在安装依赖..."
    pip install -r web_interface/requirements.txt
}

# 检查模型文件
echo "🔍 检查模型文件..."
if [ ! -d "out/threedfront_objfeat_vqvae" ]; then
    echo "⚠️  警告: 未找到 fVQ-VAE 模型文件"
    echo "   请确保已下载模型到 out/threedfront_objfeat_vqvae/"
fi

if [ ! -d "out/bedroom_sg2scdiffusion_objfeat" ]; then
    echo "⚠️  警告: 未找到 bedroom 模型文件"
    echo "   请确保已下载模型到 out/bedroom_sg2scdiffusion_objfeat/"
fi

echo ""
echo "🚀 启动 Web 服务器..."
echo ""
echo "访问地址: http://localhost:6006"
echo "按 Ctrl+C 停止服务器"
echo ""
echo "=========================================="
echo ""

# 启动服务器
cd web_interface
python app.py


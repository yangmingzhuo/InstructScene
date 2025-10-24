#!/usr/bin/env python
"""
简单的启动脚本 - 用于快速启动 Web 界面
使用方法: python run.py
"""

import os
import sys

# 切换到脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 添加父目录到路径
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

print("=" * 60)
print("Web Interface")
print("=" * 60)
print(f"工作目录: {parent_dir}")
print("启动服务器...")
print("=" * 60)

# 导入并运行 Flask app
from web_interface.app import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006, debug=True)


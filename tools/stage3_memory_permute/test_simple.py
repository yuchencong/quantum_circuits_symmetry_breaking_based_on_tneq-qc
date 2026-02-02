#!/usr/bin/env python3
"""简单测试脚本 - 诊断问题"""

import sys
print("=== 开始测试 ===", flush=True)
print(f"Python 版本: {sys.version}", flush=True)

print("\n导入模块...", flush=True)
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}", flush=True)
except ImportError as e:
    print(f"✗ PyTorch 导入失败: {e}", flush=True)
    sys.exit(1)

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}", flush=True)
except ImportError as e:
    print(f"✗ NumPy 导入失败: {e}", flush=True)
    sys.exit(1)

print("\n测试基本操作...", flush=True)
import time

# 测试 1: 创建张量
print("创建张量...", flush=True)
start = time.time()
x = torch.randn(1000, 1000)
print(f"  创建张量耗时: {(time.time()-start)*1000:.2f} ms", flush=True)

# 测试 2: 复制张量
print("复制张量...", flush=True)
start = time.time()
y = x.clone()
print(f"  复制张量耗时: {(time.time()-start)*1000:.2f} ms", flush=True)

# 测试 3: 矩阵乘法
print("矩阵乘法...", flush=True)
start = time.time()
z = torch.matmul(x, x)
print(f"  矩阵乘法耗时: {(time.time()-start)*1000:.2f} ms", flush=True)

print("\n=== 测试成功完成 ===", flush=True)

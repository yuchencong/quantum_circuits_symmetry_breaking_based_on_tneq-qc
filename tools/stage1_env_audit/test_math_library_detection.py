#!/usr/bin/env python3
"""
第一阶段测试 1: 数学库链接探测
测试 PyTorch 及底层 NumPy/SciPy 链接的 BLAS/LAPACK 库版本
验证是否链接了 Fujitsu SSL2、ArmPL 或支持 SVE 的 OpenBLAS
"""

import sys
import subprocess
import torch
import numpy as np
import json
import os
from pathlib import Path

def check_pytorch_config():
    """检查 PyTorch 配置和链接库"""
    print("=" * 80)
    print("PyTorch 配置信息")
    print("=" * 80)
    
    results = {
        "pytorch_version": torch.__version__,
        "pytorch_config": {},
        "linked_libraries": [],
        "warnings": []
    }
    
    # 显示 PyTorch 配置
    print(f"\nPyTorch 版本: {torch.__version__}")
    print("\nPyTorch 详细配置:")
    print("-" * 80)
    
    try:
        config_str = torch.__config__.show()
        print(config_str)
        results["pytorch_config"]["raw_output"] = str(config_str)
    except Exception as e:
        print(f"无法获取 PyTorch 配置: {e}")
        results["warnings"].append(f"PyTorch config error: {str(e)}")
    
    # 检查是否为 CPU 版本
    print(f"\nCUDA 可用: {torch.cuda.is_available()}")
    results["pytorch_config"]["cuda_available"] = torch.cuda.is_available()
    
    if torch.cuda.is_available():
        results["warnings"].append("警告: 检测到 CUDA，但富岳应该使用 CPU 版本")
    
    return results

def check_numpy_blas():
    """检查 NumPy 链接的 BLAS 库"""
    print("\n" + "=" * 80)
    print("NumPy BLAS/LAPACK 配置")
    print("=" * 80)
    
    results = {
        "numpy_version": np.__version__,
        "blas_info": {}
    }
    
    print(f"\nNumPy 版本: {np.__version__}")
    
    try:
        import numpy.distutils.system_info as sysinfo
        
        # 获取 BLAS 信息
        blas_info = sysinfo.get_info('blas_opt')
        print("\nBLAS 优化库信息:")
        print("-" * 80)
        for key, value in blas_info.items():
            print(f"{key}: {value}")
            results["blas_info"][key] = str(value)
        
        # 获取 LAPACK 信息
        lapack_info = sysinfo.get_info('lapack_opt')
        print("\nLAPACK 优化库信息:")
        print("-" * 80)
        for key, value in lapack_info.items():
            print(f"{key}: {value}")
            results["blas_info"][f"lapack_{key}"] = str(value)
            
    except Exception as e:
        print(f"无法获取 NumPy BLAS 信息: {e}")
        results["warnings"] = [f"NumPy BLAS info error: {str(e)}"]
    
    # 尝试使用 numpy show_config
    try:
        print("\nNumPy 配置详情:")
        print("-" * 80)
        config = np.show_config()
        print(config)
    except Exception as e:
        print(f"无法显示 NumPy 配置: {e}")
    
    return results

def check_linked_libraries():
    """使用 ldd 检查实际链接的动态库"""
    print("\n" + "=" * 80)
    print("动态链接库检查 (ldd)")
    print("=" * 80)
    
    results = {
        "linked_libs": [],
        "blas_library": "unknown",
        "warnings": []
    }
    
    # 获取 Python 可执行文件路径
    python_path = sys.executable
    print(f"\nPython 路径: {python_path}")
    
    # 查找 PyTorch 共享库
    try:
        import torch
        torch_path = Path(torch.__file__).parent
        torch_libs = list(torch_path.glob("**/*.so"))
        
        print(f"\n找到 {len(torch_libs)} 个 PyTorch .so 文件")
        
        # 检查主要的 torch 库
        main_libs = [lib for lib in torch_libs if 'torch' in lib.name and 'python' in lib.name]
        
        if main_libs:
            target_lib = main_libs[0]
            print(f"\n分析库: {target_lib}")
            print("-" * 80)
            
            try:
                result = subprocess.run(
                    ['ldd', str(target_lib)],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                output = result.stdout
                print(output)
                
                results["linked_libs"] = output.split('\n')
                
                # 检查特定的 BLAS 库
                if 'ssl2' in output.lower() or 'fujitsu' in output.lower():
                    results["blas_library"] = "Fujitsu SSL2 (推荐)"
                    print("\n✓ 检测到 Fujitsu SSL2 - 这是富岳的最佳选择!")
                elif 'armpl' in output.lower():
                    results["blas_library"] = "ArmPL (推荐)"
                    print("\n✓ 检测到 ArmPL - ARM 优化库")
                elif 'openblas' in output.lower():
                    if 'sve' in output.lower():
                        results["blas_library"] = "OpenBLAS with SVE (推荐)"
                        print("\n✓ 检测到支持 SVE 的 OpenBLAS")
                    else:
                        results["blas_library"] = "OpenBLAS (可能未优化)"
                        results["warnings"].append("警告: OpenBLAS 可能未针对 ARM SVE 优化")
                        print("\n⚠ 检测到 OpenBLAS，但可能未针对 SVE 优化")
                elif 'blas' in output.lower() or 'lapack' in output.lower():
                    results["blas_library"] = "Generic BLAS (性能较差)"
                    results["warnings"].append("警告: 使用通用 BLAS，性能可能差 5-10 倍")
                    print("\n⚠ 使用通用 BLAS 库 - 性能将严重受限!")
                else:
                    results["blas_library"] = "Unknown"
                    results["warnings"].append("警告: 无法识别 BLAS 库类型")
                    print("\n⚠ 未检测到明确的 BLAS 库链接")
                    
            except subprocess.TimeoutExpired:
                print("ldd 命令超时")
                results["warnings"].append("ldd timeout")
            except FileNotFoundError:
                print("ldd 命令不可用")
                results["warnings"].append("ldd not available")
                
    except Exception as e:
        print(f"检查链接库时出错: {e}")
        results["warnings"].append(f"Library check error: {str(e)}")
    
    return results

def save_results(all_results, output_dir="test_results"):
    """保存测试结果到 JSON 文件"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "stage1_math_library_detection.json")
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n结果已保存到: {output_file}")

def main():
    print("\n" + "=" * 80)
    print("富岳集群 - 第一阶段测试: 数学库链接探测")
    print("=" * 80)
    
    all_results = {
        "test_name": "math_library_detection",
        "stage": 1
    }
    
    # 执行各项检查
    all_results["pytorch"] = check_pytorch_config()
    all_results["numpy"] = check_numpy_blas()
    all_results["linked_libraries"] = check_linked_libraries()
    
    # 汇总警告
    all_warnings = []
    for category in ['pytorch', 'numpy', 'linked_libraries']:
        if 'warnings' in all_results[category]:
            all_warnings.extend(all_results[category]['warnings'])
    
    # 打印总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    blas_lib = all_results["linked_libraries"].get("blas_library", "unknown")
    print(f"\n检测到的 BLAS 库: {blas_lib}")
    
    if all_warnings:
        print("\n警告列表:")
        for i, warning in enumerate(all_warnings, 1):
            print(f"  {i}. {warning}")
    else:
        print("\n✓ 所有检查通过，未发现问题")
    
    # 给出建议
    print("\n建议:")
    if "Fujitsu SSL2" in blas_lib or "ArmPL" in blas_lib or "SVE" in blas_lib:
        print("  ✓ 当前配置适合富岳集群，可以进行下一步测试")
    else:
        print("  ⚠ 建议重新编译 PyTorch 并链接到优化的数学库")
        print("    - 推荐使用 Fujitsu SSL2")
        print("    - 或使用 ARM Performance Libraries (ArmPL)")
        print("    - 或使用支持 SVE 的 OpenBLAS")
    
    # 保存结果
    save_results(all_results)
    
    return 0 if not all_warnings else 1

if __name__ == "__main__":
    sys.exit(main())

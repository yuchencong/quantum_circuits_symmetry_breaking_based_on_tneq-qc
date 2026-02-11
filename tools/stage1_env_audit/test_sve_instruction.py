#!/usr/bin/env python3
"""
第一阶段测试 2: SVE 指令集有效性验证
运行小规模矩阵乘法，观察是否触发 512-bit SVE 向量指令
"""

import sys
import torch
import time
import json
import os
import subprocess

def test_matmul_performance(sizes=[128, 256, 512, 1024], num_trials=10):
    """测试不同规模的矩阵乘法性能"""
    print("=" * 80)
    print("矩阵乘法性能测试")
    print("=" * 80)
    
    results = []
    
    for size in sizes:
        print(f"\n测试矩阵规模: {size}x{size}")
        print("-" * 80)
        
        # 预热
        A = torch.randn(size, size, dtype=torch.float32)
        B = torch.randn(size, size, dtype=torch.float32)
        _ = torch.matmul(A, B)
        
        # 计时测试
        times = []
        for trial in range(num_trials):
            A = torch.randn(size, size, dtype=torch.float32)
            B = torch.randn(size, size, dtype=torch.float32)
            
            start = time.perf_counter()
            C = torch.matmul(A, B)
            end = time.perf_counter()
            
            times.append(end - start)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        
        # 计算 GFLOPS
        # 矩阵乘法的浮点运算次数: 2 * N^3
        flops = 2 * size ** 3
        gflops = flops / (avg_time * 1e9)
        peak_gflops = flops / (min_time * 1e9)
        
        result = {
            "size": size,
            "avg_time_ms": avg_time * 1000,
            "min_time_ms": min_time * 1000,
            "avg_gflops": gflops,
            "peak_gflops": peak_gflops
        }
        results.append(result)
        
        print(f"  平均时间: {avg_time*1000:.4f} ms")
        print(f"  最短时间: {min_time*1000:.4f} ms")
        print(f"  平均性能: {gflops:.2f} GFLOPS")
        print(f"  峰值性能: {peak_gflops:.2f} GFLOPS")
    
    return results

def test_different_dtypes():
    """测试不同数据类型的性能"""
    print("\n" + "=" * 80)
    print("不同精度性能对比")
    print("=" * 80)
    
    size = 1024
    num_trials = 20
    dtypes = [torch.float32, torch.float16]
    dtype_names = ["FP32", "FP16"]
    
    results = []
    
    for dtype, dtype_name in zip(dtypes, dtype_names):
        print(f"\n测试数据类型: {dtype_name}")
        print("-" * 80)
        
        try:
            # 预热
            A = torch.randn(size, size, dtype=dtype)
            B = torch.randn(size, size, dtype=dtype)
            _ = torch.matmul(A, B)
            
            times = []
            for _ in range(num_trials):
                A = torch.randn(size, size, dtype=dtype)
                B = torch.randn(size, size, dtype=dtype)
                
                start = time.perf_counter()
                C = torch.matmul(A, B)
                end = time.perf_counter()
                
                times.append(end - start)
            
            avg_time = sum(times) / len(times)
            flops = 2 * size ** 3
            gflops = flops / (avg_time * 1e9)
            
            result = {
                "dtype": dtype_name,
                "avg_time_ms": avg_time * 1000,
                "gflops": gflops,
                "supported": True
            }
            
            print(f"  平均时间: {avg_time*1000:.4f} ms")
            print(f"  性能: {gflops:.2f} GFLOPS")
            
        except Exception as e:
            print(f"  错误: {e}")
            result = {
                "dtype": dtype_name,
                "supported": False,
                "error": str(e)
            }
        
        results.append(result)
    
    # 计算加速比
    if len(results) >= 2 and results[0]["supported"] and results[1]["supported"]:
        speedup = results[1]["gflops"] / results[0]["gflops"]
        print(f"\nFP16 vs FP32 加速比: {speedup:.2f}x")
        results.append({"fp16_speedup": speedup})
    
    return results

def check_cpu_info():
    """检查 CPU 信息和指令集支持"""
    print("\n" + "=" * 80)
    print("CPU 信息检查")
    print("=" * 80)
    
    results = {
        "cpu_info": {},
        "sve_support": "unknown"
    }
    
    try:
        # 读取 /proc/cpuinfo
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
        
        # 提取关键信息
        for line in cpuinfo.split('\n'):
            if 'model name' in line.lower():
                model = line.split(':')[1].strip()
                results["cpu_info"]["model"] = model
                print(f"CPU 型号: {model}")
                break
        
        # 检查 Features
        features_found = False
        for line in cpuinfo.split('\n'):
            if 'features' in line.lower() or 'flags' in line.lower():
                features = line.split(':')[1].strip()
                results["cpu_info"]["features"] = features
                features_found = True
                
                # 检查 SVE 支持
                if 'sve' in features.lower():
                    results["sve_support"] = "yes"
                    print("\n✓ 检测到 SVE 指令集支持")
                else:
                    results["sve_support"] = "no"
                    print("\n⚠ 未检测到 SVE 指令集")
                
                print(f"\nCPU Features: {features[:200]}...")
                break
        
        if not features_found:
            print("\n无法从 /proc/cpuinfo 获取 Features 信息")
            
    except Exception as e:
        print(f"读取 CPU 信息时出错: {e}")
        results["error"] = str(e)
    
    # 尝试使用 lscpu
    try:
        result = subprocess.run(['lscpu'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("\n" + "-" * 80)
            print("lscpu 输出:")
            print("-" * 80)
            print(result.stdout)
            results["lscpu"] = result.stdout
    except Exception as e:
        print(f"无法运行 lscpu: {e}")
    
    return results

def analyze_performance(matmul_results, cpu_info):
    """分析性能并给出评估"""
    print("\n" + "=" * 80)
    print("性能分析与评估")
    print("=" * 80)
    
    analysis = {
        "verdict": "unknown",
        "recommendations": []
    }
    
    # 获取最大矩阵的性能
    if matmul_results:
        largest_test = matmul_results[-1]
        gflops = largest_test["avg_gflops"]
        
        print(f"\n最大规模矩阵 ({largest_test['size']}x{largest_test['size']}) 性能: {gflops:.2f} GFLOPS")
        
        # 富岳单 CMG 理论峰值约 768 GFLOPS @ 2.0GHz (FP32)
        # 考虑单线程，预期 50-100 GFLOPS
        # 如果使用 OpenMP，单 CMG 预期 300-600 GFLOPS
        
        if gflops > 200:
            analysis["verdict"] = "excellent"
            print("\n✓ 性能优秀 - SVE 指令集可能已启用")
            analysis["recommendations"].append("性能已达到较高水平，可以进行后续测试")
        elif gflops > 50:
            analysis["verdict"] = "good"
            print("\n✓ 性能良好 - 建议检查是否充分利用多核")
            analysis["recommendations"].append("考虑启用 OpenMP 或增加线程数")
        elif gflops > 10:
            analysis["verdict"] = "poor"
            print("\n⚠ 性能较差 - 可能未使用 SVE 指令")
            analysis["recommendations"].append("检查编译选项，确保启用 SVE 支持")
            analysis["recommendations"].append("验证是否链接了优化的 BLAS 库")
        else:
            analysis["verdict"] = "very_poor"
            print("\n✗ 性能极差 - 严重性能问题")
            analysis["recommendations"].append("重新编译 PyTorch 并启用 ARM 优化")
            analysis["recommendations"].append("使用 Fujitsu 编译器和 SSL2 库")
        
        analysis["gflops"] = gflops
        analysis["theoretical_peak_single_cmg"] = 768
        analysis["utilization_percent"] = (gflops / 768) * 100
    
    # SVE 支持检查
    sve_support = cpu_info.get("sve_support", "unknown")
    if sve_support == "yes":
        print("\n硬件支持 SVE 指令集")
        if analysis.get("verdict") in ["poor", "very_poor"]:
            analysis["recommendations"].append("硬件支持 SVE 但性能未达标，可能是软件配置问题")
    elif sve_support == "no":
        print("\n⚠ 硬件不支持 SVE - 这不是富岳 A64FX 处理器")
        analysis["recommendations"].append("当前处理器不支持 SVE，测试结果仅供参考")
    
    return analysis

def save_results(all_results, output_dir="test_results"):
    """保存测试结果"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "stage1_sve_instruction.json")
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n结果已保存到: {output_file}")

def main():
    print("\n" + "=" * 80)
    print("富岳集群 - 第一阶段测试: SVE 指令集验证")
    print("=" * 80)
    
    all_results = {
        "test_name": "sve_instruction_validation",
        "stage": 1
    }
    
    # CPU 信息检查
    all_results["cpu_info"] = check_cpu_info()
    
    # 矩阵乘法性能测试
    all_results["matmul_fp32"] = test_matmul_performance()
    
    # 不同精度测试
    all_results["dtype_comparison"] = test_different_dtypes()
    
    # 性能分析
    all_results["analysis"] = analyze_performance(
        all_results["matmul_fp32"],
        all_results["cpu_info"]
    )
    
    # 打印建议
    print("\n" + "=" * 80)
    print("测试建议")
    print("=" * 80)
    recommendations = all_results["analysis"].get("recommendations", [])
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    # 保存结果
    save_results(all_results)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

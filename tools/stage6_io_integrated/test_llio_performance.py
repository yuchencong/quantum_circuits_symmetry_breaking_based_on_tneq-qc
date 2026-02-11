#!/usr/bin/env python3
"""
第六阶段测试 1: LLIO (Local Layered I/O) 性能测试
测试模型 Checkpoint 的保存和加载速度
"""

import sys
import torch
import time
import json
import os
import numpy as np
import tempfile
import shutil

def create_model_state_dict(num_tensors=10, tensor_size_mb=10):
    """创建模拟的模型状态字典"""
    state_dict = {}
    
    elements_per_tensor = int(tensor_size_mb * 1024 * 1024 / 4)  # FP32
    
    for i in range(num_tensors):
        # 模拟不同形状的张量
        if i % 3 == 0:
            # 2D 张量 (权重矩阵)
            size = int(np.sqrt(elements_per_tensor))
            tensor = torch.randn(size, size)
        elif i % 3 == 1:
            # 4D 张量 (卷积核)
            size = int((elements_per_tensor / 16) ** (1/4))
            tensor = torch.randn(16, 16, size, size)
        else:
            # 1D 张量 (偏置)
            tensor = torch.randn(elements_per_tensor)
        
        state_dict[f'layer_{i}.weight'] = tensor
    
    return state_dict

def test_torch_save_load():
    """测试 torch.save 和 torch.load 性能"""
    print("=" * 80)
    print("Torch Save/Load 性能测试")
    print("=" * 80)
    
    results = []
    
    # 测试不同的模型大小
    model_configs = [
        (10, 10),   # 100MB
        (20, 10),   # 200MB
        (50, 10),   # 500MB
        (100, 10),  # 1GB
    ]
    
    # 使用临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        for num_tensors, tensor_size_mb in model_configs:
            total_size_mb = num_tensors * tensor_size_mb
            print(f"\n测试模型大小: {total_size_mb} MB ({num_tensors} 个张量)")
            print("-" * 80)
            
            # 创建模型
            state_dict = create_model_state_dict(num_tensors, tensor_size_mb)
            
            # 测试保存
            save_path = os.path.join(tmpdir, f'checkpoint_{total_size_mb}mb.pt')
            
            save_times = []
            for trial in range(5):
                start = time.perf_counter()
                torch.save(state_dict, save_path)
                end = time.perf_counter()
                save_times.append(end - start)
                
                # 清除文件系统缓存的影响（如果可能）
                if trial < 4:
                    os.sync()
            
            avg_save_time = np.mean(save_times)
            save_bandwidth = (total_size_mb / 1024) / avg_save_time  # GB/s
            
            # 测试加载
            load_times = []
            for _ in range(5):
                start = time.perf_counter()
                loaded_state = torch.load(save_path)
                end = time.perf_counter()
                load_times.append(end - start)
            
            avg_load_time = np.mean(load_times)
            load_bandwidth = (total_size_mb / 1024) / avg_load_time  # GB/s
            
            # 获取实际文件大小
            file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
            
            result = {
                "model_size_mb": total_size_mb,
                "file_size_mb": file_size_mb,
                "num_tensors": num_tensors,
                "save_time_s": avg_save_time,
                "load_time_s": avg_load_time,
                "save_bandwidth_gbps": save_bandwidth,
                "load_bandwidth_gbps": load_bandwidth
            }
            results.append(result)
            
            print(f"  文件大小: {file_size_mb:.2f} MB")
            print(f"  保存时间: {avg_save_time:.3f} s ({save_bandwidth:.2f} GB/s)")
            print(f"  加载时间: {avg_load_time:.3f} s ({load_bandwidth:.2f} GB/s)")
            
            # 清理
            os.remove(save_path)
    
    return results

def test_safetensors_vs_pickle():
    """对比 safetensors 和 pickle 格式的性能"""
    print("\n" + "=" * 80)
    print("Safetensors vs Pickle 格式对比")
    print("=" * 80)
    
    # 检查 safetensors 是否可用
    try:
        from safetensors.torch import save_file, load_file
        has_safetensors = True
    except ImportError:
        print("未安装 safetensors，跳过此测试")
        print("安装命令: pip install safetensors")
        has_safetensors = False
        return {"error": "safetensors not available"}
    
    results = []
    model_size_mb = 500
    num_tensors = 50
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n测试模型大小: {model_size_mb} MB")
        print("-" * 80)
        
        state_dict = create_model_state_dict(num_tensors, model_size_mb // num_tensors)
        
        # 测试 1: Pickle (torch.save)
        pickle_path = os.path.join(tmpdir, 'model_pickle.pt')
        
        start = time.perf_counter()
        torch.save(state_dict, pickle_path)
        pickle_save_time = time.perf_counter() - start
        
        start = time.perf_counter()
        _ = torch.load(pickle_path)
        pickle_load_time = time.perf_counter() - start
        
        pickle_size_mb = os.path.getsize(pickle_path) / (1024 * 1024)
        
        print(f"\nPickle 格式:")
        print(f"  文件大小: {pickle_size_mb:.2f} MB")
        print(f"  保存时间: {pickle_save_time:.3f} s")
        print(f"  加载时间: {pickle_load_time:.3f} s")
        
        # 测试 2: Safetensors
        safe_path = os.path.join(tmpdir, 'model_safe.safetensors')
        
        start = time.perf_counter()
        save_file(state_dict, safe_path)
        safe_save_time = time.perf_counter() - start
        
        start = time.perf_counter()
        _ = load_file(safe_path)
        safe_load_time = time.perf_counter() - start
        
        safe_size_mb = os.path.getsize(safe_path) / (1024 * 1024)
        
        print(f"\nSafetensors 格式:")
        print(f"  文件大小: {safe_size_mb:.2f} MB")
        print(f"  保存时间: {safe_save_time:.3f} s")
        print(f"  加载时间: {safe_load_time:.3f} s")
        
        # 对比
        print(f"\n对比:")
        print(f"  文件大小比: {safe_size_mb/pickle_size_mb:.2f}x")
        print(f"  保存速度比: {pickle_save_time/safe_save_time:.2f}x")
        print(f"  加载速度比: {pickle_load_time/safe_load_time:.2f}x")
        
        result = {
            "pickle": {
                "size_mb": pickle_size_mb,
                "save_time_s": pickle_save_time,
                "load_time_s": pickle_load_time
            },
            "safetensors": {
                "size_mb": safe_size_mb,
                "save_time_s": safe_save_time,
                "load_time_s": safe_load_time
            },
            "comparison": {
                "size_ratio": safe_size_mb / pickle_size_mb,
                "save_speedup": pickle_save_time / safe_save_time,
                "load_speedup": pickle_load_time / safe_load_time
            }
        }
        results.append(result)
    
    return results

def test_checkpoint_strategies():
    """测试不同的 checkpoint 策略"""
    print("\n" + "=" * 80)
    print("Checkpoint 策略对比")
    print("=" * 80)
    
    results = {}
    
    num_tensors = 20
    tensor_size_mb = 10
    state_dict = create_model_state_dict(num_tensors, tensor_size_mb)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 策略 1: 完整保存
        print("\n策略 1: 保存完整模型")
        print("-" * 80)
        
        full_path = os.path.join(tmpdir, 'full_checkpoint.pt')
        
        start = time.perf_counter()
        torch.save({
            'model_state_dict': state_dict,
            'optimizer_state_dict': state_dict,  # 模拟 optimizer
            'epoch': 100,
            'loss': 0.1234
        }, full_path)
        full_time = time.perf_counter() - start
        full_size_mb = os.path.getsize(full_path) / (1024 * 1024)
        
        print(f"  大小: {full_size_mb:.2f} MB")
        print(f"  时间: {full_time:.3f} s")
        
        results['full'] = {
            "size_mb": full_size_mb,
            "time_s": full_time
        }
        
        # 策略 2: 只保存模型
        print("\n策略 2: 只保存模型权重")
        print("-" * 80)
        
        model_only_path = os.path.join(tmpdir, 'model_only.pt')
        
        start = time.perf_counter()
        torch.save(state_dict, model_only_path)
        model_time = time.perf_counter() - start
        model_size_mb = os.path.getsize(model_only_path) / (1024 * 1024)
        
        print(f"  大小: {model_size_mb:.2f} MB")
        print(f"  时间: {model_time:.3f} s")
        print(f"  vs 完整: {full_size_mb/model_size_mb:.2f}x 大小")
        
        results['model_only'] = {
            "size_mb": model_size_mb,
            "time_s": model_time,
            "vs_full_size": full_size_mb / model_size_mb
        }
    
    return results

def analyze_io_performance(all_results):
    """分析 IO 性能"""
    print("\n" + "=" * 80)
    print("IO 性能分析")
    print("=" * 80)
    
    analysis = {
        "recommendations": []
    }
    
    # 分析保存带宽
    if "torch_save_load" in all_results and all_results["torch_save_load"]:
        # 取最大模型的性能
        largest = all_results["torch_save_load"][-1]
        save_bw = largest["save_bandwidth_gbps"]
        load_bw = largest["load_bandwidth_gbps"]
        
        print(f"\n最大模型 ({largest['model_size_mb']} MB) IO 性能:")
        print(f"  保存带宽: {save_bw:.2f} GB/s")
        print(f"  加载带宽: {load_bw:.2f} GB/s")
        
        # 富岳的本地 NVMe 预期 > 2 GB/s
        if save_bw > 2.0:
            print("\n✓ 保存性能优秀，可能使用了 LLIO")
            analysis["io_verdict"] = "excellent"
        elif save_bw > 0.5:
            print("\n✓ 保存性能良好")
            analysis["io_verdict"] = "good"
            analysis["recommendations"].append("考虑使用 LLIO 以进一步提升性能")
        else:
            print("\n⚠ 保存性能较低")
            analysis["io_verdict"] = "poor"
            analysis["recommendations"].append("检查是否使用共享文件系统而非本地存储")
            analysis["recommendations"].append("富岳建议使用 LLIO (Local Layered I/O)")
    
    # 分析格式选择
    if "safetensors_vs_pickle" in all_results and not isinstance(all_results["safetensors_vs_pickle"], dict):
        comparison = all_results["safetensors_vs_pickle"][0]["comparison"]
        
        if comparison["load_speedup"] > 1.2:
            print(f"\n✓ Safetensors 加载速度快 {comparison['load_speedup']:.1f}x")
            analysis["recommendations"].append("推荐使用 safetensors 格式保存模型")
    
    return analysis

def save_results(all_results, output_dir="../test_results/stage6"):
    """保存测试结果"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "llio_performance.json")
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n结果已保存到: {output_file}")

def main():
    print("\n" + "=" * 80)
    print("富岳集群 - 第六阶段测试: LLIO 性能")
    print("=" * 80)
    
    all_results = {
        "test_name": "llio_performance",
        "stage": 6
    }
    
    # 执行测试
    all_results["torch_save_load"] = test_torch_save_load()
    all_results["safetensors_vs_pickle"] = test_safetensors_vs_pickle()
    all_results["checkpoint_strategies"] = test_checkpoint_strategies()
    
    # 分析结果
    all_results["analysis"] = analyze_io_performance(all_results)
    
    # 打印建议
    print("\n" + "=" * 80)
    print("优化建议")
    print("=" * 80)
    recommendations = all_results["analysis"].get("recommendations", [])
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    print("\n富岳 LLIO 使用提示:")
    print("  1. 将 checkpoint 保存到本地 /local 目录而非共享文件系统")
    print("  2. 训练结束后将重要 checkpoint 移到持久化存储")
    print("  3. 使用 safetensors 格式可以提升加载速度")
    
    # 保存结果
    save_results(all_results)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

#!/bin/bash
#PJM -L "rscgrp=small"
#PJM -L "node=4"
#PJM -L "elapse=00:30:00"
#PJM -j
#PJM -S

# 富岳作业提交脚本示例
# 使用方法: pjsub submit_fugaku.sh

echo "================================================================================"
echo "富岳集群 - 第五阶段通信测试 (多节点)"
echo "================================================================================"
echo "节点数: 4"
echo "开始时间: $(date)"
echo ""

# 加载必要的模块
module load python/3.x  # 根据实际情况修改
module load fujitsu-mpi  # 加载 Fujitsu MPI

# 设置环境变量
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12

# 创建结果目录
RESULT_DIR="./test_results/stage5/fugaku_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULT_DIR"

echo "结果将保存到: $RESULT_DIR"
echo ""

# 测试 1: 2 节点测试
echo "--------------------------------------------------------------------------------"
echo "测试: 2 节点 MPI 通信"
echo "--------------------------------------------------------------------------------"
mpirun -n 2 --map-by ppr:1:node python3 test_mpi_baseline.py > "$RESULT_DIR/2nodes.log" 2>&1
echo "✓ 2 节点测试完成"
echo ""

# 测试 2: 4 节点测试
echo "--------------------------------------------------------------------------------"
echo "测试: 4 节点 MPI 通信"
echo "--------------------------------------------------------------------------------"
mpirun -n 4 --map-by ppr:1:node python3 test_mpi_baseline.py > "$RESULT_DIR/4nodes.log" 2>&1
echo "✓ 4 节点测试完成"
echo ""

# 测试 3: 每节点多进程 (4 节点 x 4 进程/节点 = 16 进程)
echo "--------------------------------------------------------------------------------"
echo "测试: 4 节点 x 4 进程/节点"
echo "--------------------------------------------------------------------------------"
mpirun -n 16 --map-by ppr:4:node python3 test_mpi_baseline.py > "$RESULT_DIR/16procs.log" 2>&1
echo "✓ 16 进程测试完成"
echo ""

# 测试 4: PyTorch Distributed MPI 后端
echo "--------------------------------------------------------------------------------"
echo "测试: PyTorch Distributed (MPI 后端, 4 节点)"
echo "--------------------------------------------------------------------------------"
mpirun -n 4 --map-by ppr:1:node python3 test_torch_distributed.py mpi > "$RESULT_DIR/torch_mpi_4nodes.log" 2>&1
echo "✓ PyTorch MPI 测试完成"
echo ""

# 测试完成
echo "================================================================================"
echo "所有测试完成"
echo "================================================================================"
echo "结束时间: $(date)"
echo "结果保存在: $RESULT_DIR"
echo ""

# 显示关键结果摘要
echo "结果摘要:"
echo "--------"
for logfile in "$RESULT_DIR"/*.log; do
    echo "文件: $(basename $logfile)"
    # 提取延迟信息（如果有）
    grep -A 1 "延迟" "$logfile" | head -2
    echo ""
done

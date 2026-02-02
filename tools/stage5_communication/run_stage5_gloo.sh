#!/bin/bash
# 第五阶段 Gloo 测试启动脚本

# 默认使用 2 个进程
NUM_PROCS=${1:-2}

echo "================================================================================"
echo "富岳集群测试 - 第五阶段：通信与互联 (Gloo 测试)"
echo "================================================================================"
echo "使用 $NUM_PROCS 个进程"
echo ""

# 创建结果目录
RESULT_DIR="../test_results/stage5"
mkdir -p "$RESULT_DIR"

# 设置日志文件
LOG_FILE="$RESULT_DIR/stage5_gloo_$(date +%Y%m%d_%H%M%S).log"

echo "测试结果将保存到: $RESULT_DIR"
echo "日志文件: $LOG_FILE"
echo ""

# 设置 PyTorch distributed 环境变量
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=$NUM_PROCS

echo "启动 $NUM_PROCS 个进程进行 Gloo 测试..."
echo ""

# 启动多个进程
pids=()
for ((rank=0; rank<NUM_PROCS; rank++)); do
    export RANK=$rank
    python3 test_torch_distributed.py gloo > "$RESULT_DIR/gloo_rank${rank}.log" 2>&1 &
    pids+=($!)
done

# 等待所有进程完成
echo "等待所有进程完成..."
all_success=true
for pid in "${pids[@]}"; do
    wait $pid
    if [ $? -ne 0 ]; then
        all_success=false
    fi
done

# 合并日志
cat "$RESULT_DIR"/gloo_rank*.log >> "$LOG_FILE"

# 汇总结果
echo ""
echo "================================================================================"
echo "第五阶段 Gloo 测试完成"
echo "================================================================================"
echo ""

if $all_success; then
    echo "✓ 测试通过"
    echo ""
    echo "详细结果请查看: $RESULT_DIR"
    echo "完整日志: $LOG_FILE"
    echo ""
    echo "提示:"
    echo "  - 运行 MPI 测试进行对比: bash run_stage5_mpi.sh $NUM_PROCS"
    echo "  - Gloo 在 CPU 上的性能通常不如 MPI"
    exit 0
else
    echo "✗ 测试失败"
    echo "请查看日志文件: $LOG_FILE"
    exit 1
fi

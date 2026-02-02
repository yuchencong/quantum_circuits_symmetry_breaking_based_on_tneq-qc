#!/bin/bash
# 第四阶段测试启动脚本：节点内并行架构

echo "================================================================================"
echo "富岳集群测试 - 第四阶段：节点内并行架构"
echo "================================================================================"
echo ""

# 创建结果目录
RESULT_DIR="../test_results/stage4"
mkdir -p "$RESULT_DIR"

# 设置日志文件
LOG_FILE="$RESULT_DIR/stage4_$(date +%Y%m%d_%H%M%S).log"

echo "测试结果将保存到: $RESULT_DIR"
echo "日志文件: $LOG_FILE"
echo ""

# 执行测试 1: NUMA 惩罚测试
echo "--------------------------------------------------------------------------------"
echo "测试 1/2: NUMA 惩罚测试"
echo "--------------------------------------------------------------------------------"
python3 test_numa_penalty.py 2>&1 | tee -a "$LOG_FILE"
TEST1_STATUS=$?

echo ""
echo "--------------------------------------------------------------------------------"
echo "测试 2/2: OpenMP vs MPI 对比"
echo "--------------------------------------------------------------------------------"
python3 test_openmp_vs_mpi.py 2>&1 | tee -a "$LOG_FILE"
TEST2_STATUS=$?

# 汇总结果
echo ""
echo "================================================================================"
echo "第四阶段测试完成"
echo "================================================================================"
echo ""
echo "测试结果摘要:"
echo "  测试 1 (NUMA 惩罚): $([ $TEST1_STATUS -eq 0 ] && echo '✓ 通过' || echo '✗ 失败')"
echo "  测试 2 (OpenMP vs MPI): $([ $TEST2_STATUS -eq 0 ] && echo '✓ 通过' || echo '✗ 失败')"
echo ""
echo "详细结果请查看: $RESULT_DIR"
echo "完整日志: $LOG_FILE"
echo ""

# 关键建议
echo "关键建议:"
echo "  - 在富岳上，推荐使用 MPI 多进程 + OpenMP 混合模式"
echo "  - 每个 MPI 进程绑定到一个 CMG (12 核)"
echo "  - 使用 numactl 进行内存和 CPU 绑定"
echo "  - 避免跨 CMG 的内存访问和线程迁移"
echo ""

# 返回状态
if [ $TEST1_STATUS -eq 0 ] && [ $TEST2_STATUS -eq 0 ]; then
    echo "✓ 所有测试通过，可以继续第五阶段测试（通信测试需要多节点）"
    exit 0
else
    echo "✗ 部分测试失败，请检查配置"
    exit 1
fi

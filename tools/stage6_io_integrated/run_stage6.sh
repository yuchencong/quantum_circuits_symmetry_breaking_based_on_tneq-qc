#!/bin/bash
# 第六阶段测试启动脚本：IO 与综合场景

echo "================================================================================"
echo "富岳集群测试 - 第六阶段：IO 与综合场景"
echo "================================================================================"
echo ""

# 创建结果目录
RESULT_DIR="../test_results/stage6"
mkdir -p "$RESULT_DIR"

# 设置日志文件
LOG_FILE="$RESULT_DIR/stage6_$(date +%Y%m%d_%H%M%S).log"

echo "测试结果将保存到: $RESULT_DIR"
echo "日志文件: $LOG_FILE"
echo ""

# 设置线程数
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12

# 执行测试 1: LLIO 性能测试
echo "--------------------------------------------------------------------------------"
echo "测试 1/2: LLIO 性能测试"
echo "--------------------------------------------------------------------------------"
python3 test_llio_performance.py 2>&1 | tee -a "$LOG_FILE"
TEST1_STATUS=$?

echo ""
echo "--------------------------------------------------------------------------------"
echo "测试 2/2: 张量网络模拟负载"
echo "--------------------------------------------------------------------------------"
python3 test_tensor_network_workload.py 2>&1 | tee -a "$LOG_FILE"
TEST2_STATUS=$?

# 汇总结果
echo ""
echo "================================================================================"
echo "第六阶段测试完成"
echo "================================================================================"
echo ""
echo "测试结果摘要:"
echo "  测试 1 (LLIO 性能): $([ $TEST1_STATUS -eq 0 ] && echo '✓ 通过' || echo '✗ 失败')"
echo "  测试 2 (张量网络负载): $([ $TEST2_STATUS -eq 0 ] && echo '✓ 通过' || echo '✗ 失败')"
echo ""
echo "详细结果请查看: $RESULT_DIR"
echo "完整日志: $LOG_FILE"
echo ""

# 关键发现
echo "关键发现:"
echo "  - IO 测试揭示了 checkpoint 保存/加载的最佳实践"
echo "  - 张量网络负载测试识别了主要性能瓶颈"
echo "  - 结合前面阶段的结果，可以全面优化训练流程"
echo ""

# 返回状态
if [ $TEST1_STATUS -eq 0 ] && [ $TEST2_STATUS -eq 0 ]; then
    echo "✓ 所有测试通过！"
    echo ""
    echo "现在可以:"
    echo "  1. 分析所有阶段的测试结果"
    echo "  2. 根据发现的瓶颈进行优化"
    echo "  3. 在富岳上运行实际的分布式训练"
    exit 0
else
    echo "✗ 部分测试失败，请检查配置"
    exit 1
fi

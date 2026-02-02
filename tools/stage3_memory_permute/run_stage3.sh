#!/bin/bash
# 第三阶段测试启动脚本：访存与转置

echo "================================================================================"
echo "富岳集群测试 - 第三阶段：访存与转置"
echo "================================================================================"
echo ""

# 创建结果目录
RESULT_DIR="../test_results/stage3"
mkdir -p "$RESULT_DIR"

# 设置日志文件
LOG_FILE="$RESULT_DIR/stage3_$(date +%Y%m%d_%H%M%S).log"

echo "测试结果将保存到: $RESULT_DIR"
echo "日志文件: $LOG_FILE"
echo ""

# 设置线程数
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12

echo "线程设置:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo ""

# 执行测试 1: HBM2 带宽测试
echo "--------------------------------------------------------------------------------"
echo "测试 1/3: HBM2 峰值带宽"
echo "--------------------------------------------------------------------------------"
python3 test_hbm2_bandwidth.py 2>&1 | tee -a "$LOG_FILE"
TEST1_STATUS=$?

echo ""
echo "--------------------------------------------------------------------------------"
echo "测试 2/3: 张量转置代价"
echo "--------------------------------------------------------------------------------"
python3 test_transpose_cost.py 2>&1 | tee -a "$LOG_FILE"
TEST2_STATUS=$?

echo ""
echo "--------------------------------------------------------------------------------"
echo "测试 3/3: L2 Cache 敏感度"
echo "--------------------------------------------------------------------------------"
python3 test_cache_sensitivity.py 2>&1 | tee -a "$LOG_FILE"
TEST3_STATUS=$?

# 汇总结果
echo ""
echo "================================================================================"
echo "第三阶段测试完成"
echo "================================================================================"
echo ""
echo "测试结果摘要:"
echo "  测试 1 (HBM2 带宽): $([ $TEST1_STATUS -eq 0 ] && echo '✓ 通过' || echo '✗ 失败')"
echo "  测试 2 (转置代价): $([ $TEST2_STATUS -eq 0 ] && echo '✓ 通过' || echo '✗ 失败')"
echo "  测试 3 (Cache 敏感度): $([ $TEST3_STATUS -eq 0 ] && echo '✓ 通过' || echo '✗ 失败')"
echo ""
echo "详细结果请查看: $RESULT_DIR"
echo "完整日志: $LOG_FILE"
echo ""

# 关键提示
echo "关键发现:"
echo "  - HBM2 带宽测试揭示了内存访问的最大吞吐量"
echo "  - 转置测试显示了维度重排的代价，这在张量网络中非常关键"
echo "  - Cache 测试识别了性能悬崖，避免这些尺寸可以提升性能"
echo ""

# 返回状态
if [ $TEST1_STATUS -eq 0 ] && [ $TEST2_STATUS -eq 0 ] && [ $TEST3_STATUS -eq 0 ]; then
    echo "✓ 所有测试通过，可以继续第四阶段测试"
    exit 0
else
    echo "✗ 部分测试失败，请检查配置"
    exit 1
fi

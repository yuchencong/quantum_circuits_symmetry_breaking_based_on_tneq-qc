#!/bin/bash
# 第二阶段测试启动脚本：计算微基准测试

echo "================================================================================"
echo "富岳集群测试 - 第二阶段：计算微基准测试"
echo "================================================================================"
echo ""

# 激活 conda 环境
if command -v conda &> /dev/null; then
    echo "激活 conda 环境: py311"
    eval "$(conda shell.bash hook)"
    conda activate py311
    if [ $? -eq 0 ]; then
        echo "✓ conda 环境 py311 已激活"
        echo "Python 版本: $(python --version)"
    else
        echo "⚠ 无法激活 py311 环境，使用系统默认 Python"
    fi
else
    echo "⚠ conda 未安装，使用系统默认 Python"
fi
echo ""

# 创建结果目录
RESULT_DIR="../test_results/stage2"
mkdir -p "$RESULT_DIR"

# 设置日志文件
LOG_FILE="$RESULT_DIR/stage2_$(date +%Y%m%d_%H%M%S).log"

echo "测试结果将保存到: $RESULT_DIR"
echo "日志文件: $LOG_FILE"
echo ""

# 设置线程数（单个 CMG 12 核）
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12

echo "线程设置:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo ""

# 可选：使用 numactl 绑定到特定 CMG
# 在富岳上，可以使用以下命令绑定到 CMG 0:
# numactl --cpunodebind=0 --membind=0 python3 test_single_cmg_peak.py

# 执行测试 1: 单 CMG 算力极限
echo "--------------------------------------------------------------------------------"
echo "测试 1/2: 单 CMG 算力极限"
echo "--------------------------------------------------------------------------------"
python3 test_single_cmg_peak.py 2>&1 | tee -a "$LOG_FILE"
TEST1_STATUS=$?

echo ""
echo "--------------------------------------------------------------------------------"
echo "测试 2/2: 算子融合与开销"
echo "--------------------------------------------------------------------------------"
python3 test_op_fusion.py 2>&1 | tee -a "$LOG_FILE"
TEST2_STATUS=$?

# 汇总结果
echo ""
echo "================================================================================"
echo "第二阶段测试完成"
echo "================================================================================"
echo ""
echo "测试结果摘要:"
echo "  测试 1 (单 CMG 算力): $([ $TEST1_STATUS -eq 0 ] && echo '✓ 通过' || echo '✗ 失败')"
echo "  测试 2 (算子融合): $([ $TEST2_STATUS -eq 0 ] && echo '✓ 通过' || echo '✗ 失败')"
echo ""
echo "详细结果请查看: $RESULT_DIR"
echo "完整日志: $LOG_FILE"
echo ""

# 使用 numactl 的建议
echo "提示: 在富岳上可以使用 numactl 绑定进程到特定 CMG 以获得更好性能:"
echo "  numactl --cpunodebind=0 --membind=0 bash run_stage2.sh"
echo ""

# 返回状态
if [ $TEST1_STATUS -eq 0 ] && [ $TEST2_STATUS -eq 0 ]; then
    echo "✓ 所有测试通过，可以继续第三阶段测试"
    exit 0
else
    echo "✗ 部分测试失败，请检查配置"
    exit 1
fi

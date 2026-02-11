#!/bin/bash
# 第一阶段测试启动脚本：环境与基础库审计

echo "================================================================================"
echo "富岳集群测试 - 第一阶段：环境与基础库审计"
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
RESULT_DIR="../test_results/stage1"
mkdir -p "$RESULT_DIR"

# 设置日志文件
LOG_FILE="$RESULT_DIR/stage1_$(date +%Y%m%d_%H%M%S).log"

echo "测试结果将保存到: $RESULT_DIR"
echo "日志文件: $LOG_FILE"
echo ""

# 执行测试 1: 数学库链接探测
echo "--------------------------------------------------------------------------------"
echo "测试 1/2: 数学库链接探测"
echo "--------------------------------------------------------------------------------"
python3 test_math_library_detection.py 2>&1 | tee -a "$LOG_FILE"
TEST1_STATUS=$?

echo ""
echo "--------------------------------------------------------------------------------"
echo "测试 2/2: SVE 指令集验证"
echo "--------------------------------------------------------------------------------"
python3 test_sve_instruction.py 2>&1 | tee -a "$LOG_FILE"
TEST2_STATUS=$?

# 汇总结果
echo ""
echo "================================================================================"
echo "第一阶段测试完成"
echo "================================================================================"
echo ""
echo "测试结果摘要:"
echo "  测试 1 (数学库探测): $([ $TEST1_STATUS -eq 0 ] && echo '✓ 通过' || echo '⚠ 有警告')"
echo "  测试 2 (SVE 指令集): $([ $TEST2_STATUS -eq 0 ] && echo '✓ 通过' || echo '⚠ 有警告')"
echo ""
echo "详细结果请查看: $RESULT_DIR"
echo "完整日志: $LOG_FILE"
echo ""

# 返回状态
if [ $TEST1_STATUS -eq 0 ] && [ $TEST2_STATUS -eq 0 ]; then
    echo "✓ 所有测试通过，可以继续第二阶段测试"
    exit 0
else
    echo "⚠ 部分测试有警告，建议检查配置后再继续"
    exit 1
fi

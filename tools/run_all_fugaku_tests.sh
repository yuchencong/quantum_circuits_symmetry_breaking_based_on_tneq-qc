#!/bin/bash
# 富岳集群完整测试套件启动脚本

echo "================================================================================"
echo "富岳集群完整测试套件"
echo "================================================================================"
echo "开始时间: $(date)"
echo ""

# 创建总结果目录
MAIN_RESULT_DIR="test_results"
mkdir -p "$MAIN_RESULT_DIR"

# 主日志文件
MAIN_LOG="$MAIN_RESULT_DIR/all_tests_$(date +%Y%m%d_%H%M%S).log"

echo "测试结果将保存到: $MAIN_RESULT_DIR"
echo "主日志文件: $MAIN_LOG"
echo ""

# 测试状态跟踪
STAGE1_STATUS=0
STAGE2_STATUS=0
STAGE3_STATUS=0
STAGE4_STATUS=0
STAGE5_STATUS=0
STAGE6_STATUS=0

# 询问用户要运行哪些阶段
echo "选择要运行的测试阶段:"
echo "  1. 全部运行"
echo "  2. 只运行单节点测试 (阶段 1-4, 6)"
echo "  3. 自定义选择"
echo ""
read -p "请选择 (1/2/3): " choice

run_stage1=false
run_stage2=false
run_stage3=false
run_stage4=false
run_stage5=false
run_stage6=false

case $choice in
    1)
        run_stage1=true
        run_stage2=true
        run_stage3=true
        run_stage4=true
        run_stage5=true
        run_stage6=true
        echo "将运行所有阶段"
        ;;
    2)
        run_stage1=true
        run_stage2=true
        run_stage3=true
        run_stage4=true
        run_stage6=true
        echo "将运行单节点测试 (阶段 1-4, 6)"
        echo "注意: 阶段 5 需要多节点环境，已跳过"
        ;;
    3)
        read -p "运行阶段 1 (环境审计)? (y/n): " ans
        [[ $ans == "y" ]] && run_stage1=true
        
        read -p "运行阶段 2 (计算基准)? (y/n): " ans
        [[ $ans == "y" ]] && run_stage2=true
        
        read -p "运行阶段 3 (访存与转置)? (y/n): " ans
        [[ $ans == "y" ]] && run_stage3=true
        
        read -p "运行阶段 4 (节点内并行)? (y/n): " ans
        [[ $ans == "y" ]] && run_stage4=true
        
        read -p "运行阶段 5 (通信测试, 需要多进程)? (y/n): " ans
        [[ $ans == "y" ]] && run_stage5=true
        
        read -p "运行阶段 6 (IO与综合)? (y/n): " ans
        [[ $ans == "y" ]] && run_stage6=true
        ;;
    *)
        echo "无效选择，退出"
        exit 1
        ;;
esac

echo ""
echo "开始测试..."
echo ""

# 阶段 1: 环境与基础库审计
if $run_stage1; then
    echo "================================================================================"
    echo "第一阶段: 环境与基础库审计"
    echo "================================================================================"
    cd stage1_env_audit
    bash run_stage1.sh 2>&1 | tee -a "../$MAIN_LOG"
    STAGE1_STATUS=$?
    cd ..
    echo ""
    
    if [ $STAGE1_STATUS -ne 0 ]; then
        echo "⚠ 第一阶段有警告，建议查看但可以继续"
    fi
fi

# 阶段 2: 计算微基准测试
if $run_stage2; then
    echo "================================================================================"
    echo "第二阶段: 计算微基准测试"
    echo "================================================================================"
    cd stage2_compute_benchmark
    bash run_stage2.sh 2>&1 | tee -a "../$MAIN_LOG"
    STAGE2_STATUS=$?
    cd ..
    echo ""
fi

# 阶段 3: 访存与转置
if $run_stage3; then
    echo "================================================================================"
    echo "第三阶段: 访存与转置"
    echo "================================================================================"
    cd stage3_memory_permute
    bash run_stage3.sh 2>&1 | tee -a "../$MAIN_LOG"
    STAGE3_STATUS=$?
    cd ..
    echo ""
fi

# 阶段 4: 节点内并行架构
if $run_stage4; then
    echo "================================================================================"
    echo "第四阶段: 节点内并行架构"
    echo "================================================================================"
    cd stage4_intranode_parallel
    bash run_stage4.sh 2>&1 | tee -a "../$MAIN_LOG"
    STAGE4_STATUS=$?
    cd ..
    echo ""
fi

# 阶段 5: 通信与互联
if $run_stage5; then
    echo "================================================================================"
    echo "第五阶段: 通信与互联"
    echo "================================================================================"
    echo "注意: 此阶段需要 MPI 环境"
    echo ""
    
    read -p "使用多少个进程进行测试? (默认 2): " num_procs
    num_procs=${num_procs:-2}
    
    cd stage5_communication
    bash run_stage5_mpi.sh $num_procs 2>&1 | tee -a "../$MAIN_LOG"
    STAGE5_STATUS=$?
    cd ..
    echo ""
fi

# 阶段 6: IO 与综合场景
if $run_stage6; then
    echo "================================================================================"
    echo "第六阶段: IO 与综合场景"
    echo "================================================================================"
    cd stage6_io_integrated
    bash run_stage6.sh 2>&1 | tee -a "../$MAIN_LOG"
    STAGE6_STATUS=$?
    cd ..
    echo ""
fi

# 生成测试报告
echo "================================================================================"
echo "测试完成摘要"
echo "================================================================================"
echo "结束时间: $(date)"
echo ""
echo "各阶段状态:"

$run_stage1 && echo "  阶段 1 (环境审计): $([ $STAGE1_STATUS -eq 0 ] && echo '✓ 通过' || echo '⚠ 有警告')"
$run_stage2 && echo "  阶段 2 (计算基准): $([ $STAGE2_STATUS -eq 0 ] && echo '✓ 通过' || echo '✗ 失败')"
$run_stage3 && echo "  阶段 3 (访存转置): $([ $STAGE3_STATUS -eq 0 ] && echo '✓ 通过' || echo '✗ 失败')"
$run_stage4 && echo "  阶段 4 (节点并行): $([ $STAGE4_STATUS -eq 0 ] && echo '✓ 通过' || echo '✗ 失败')"
$run_stage5 && echo "  阶段 5 (通信测试): $([ $STAGE5_STATUS -eq 0 ] && echo '✓ 通过' || echo '✗ 失败')"
$run_stage6 && echo "  阶段 6 (IO综合): $([ $STAGE6_STATUS -eq 0 ] && echo '✓ 通过' || echo '✗ 失败')"

echo ""
echo "详细结果位置:"
echo "  - 各阶段结果: test_results/stage*/"
echo "  - 主日志: $MAIN_LOG"
echo ""

# 生成 JSON 格式的总结报告
SUMMARY_FILE="$MAIN_RESULT_DIR/test_summary.json"
cat > "$SUMMARY_FILE" << EOF
{
  "test_date": "$(date -Iseconds)",
  "stages": {
    "stage1": {"run": $run_stage1, "status": $STAGE1_STATUS},
    "stage2": {"run": $run_stage2, "status": $STAGE2_STATUS},
    "stage3": {"run": $run_stage3, "status": $STAGE3_STATUS},
    "stage4": {"run": $run_stage4, "status": $STAGE4_STATUS},
    "stage5": {"run": $run_stage5, "status": $STAGE5_STATUS},
    "stage6": {"run": $run_stage6, "status": $STAGE6_STATUS}
  }
}
EOF

echo "测试摘要已保存: $SUMMARY_FILE"
echo ""

# 计算总体状态
total_failed=0
$run_stage1 && [ $STAGE1_STATUS -ne 0 ] && total_failed=$((total_failed + 1))
$run_stage2 && [ $STAGE2_STATUS -ne 0 ] && total_failed=$((total_failed + 1))
$run_stage3 && [ $STAGE3_STATUS -ne 0 ] && total_failed=$((total_failed + 1))
$run_stage4 && [ $STAGE4_STATUS -ne 0 ] && total_failed=$((total_failed + 1))
$run_stage5 && [ $STAGE5_STATUS -ne 0 ] && total_failed=$((total_failed + 1))
$run_stage6 && [ $STAGE6_STATUS -ne 0 ] && total_failed=$((total_failed + 1))

if [ $total_failed -eq 0 ]; then
    echo "✓ 所有测试通过！富岳环境配置良好"
    echo ""
    echo "后续步骤:"
    echo "  1. 查看各阶段的性能分析报告"
    echo "  2. 根据识别的瓶颈优化代码"
    echo "  3. 在富岳上运行实际的分布式张量网络训练"
    exit 0
else
    echo "⚠ 有 $total_failed 个阶段失败或有警告"
    echo "请查看详细日志进行排查"
    exit 1
fi

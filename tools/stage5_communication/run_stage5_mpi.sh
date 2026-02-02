#!/bin/bash
# 第五阶段 MPI 测试启动脚本

# 默认使用 2 个进程
NUM_PROCS=${1:-2}

echo "================================================================================"
echo "富岳集群测试 - 第五阶段：通信与互联 (MPI 测试)"
echo "================================================================================"
echo "使用 $NUM_PROCS 个 MPI 进程"
echo ""

# 创建结果目录
RESULT_DIR="../test_results/stage5"
mkdir -p "$RESULT_DIR"

# 设置日志文件
LOG_FILE="$RESULT_DIR/stage5_mpi_$(date +%Y%m%d_%H%M%S).log"

echo "测试结果将保存到: $RESULT_DIR"
echo "日志文件: $LOG_FILE"
echo ""

# 检查 mpi4py 是否安装
python3 -c "import mpi4py" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "错误: 未安装 mpi4py"
    echo "安装命令: pip install mpi4py"
    exit 1
fi

# 检查是否有 mpirun
if ! command -v mpirun &> /dev/null; then
    echo "错误: 找不到 mpirun 命令"
    echo "请确保已安装 MPI (OpenMPI, MPICH, 或 Fujitsu MPI)"
    exit 1
fi

echo "MPI 环境检查通过"
echo ""

# 执行测试 1: MPI 基线性能
echo "--------------------------------------------------------------------------------"
echo "测试 1/2: MPI 基线性能 (mpi4py)"
echo "--------------------------------------------------------------------------------"
mpirun -n $NUM_PROCS python3 test_mpi_baseline.py 2>&1 | tee -a "$LOG_FILE"
TEST1_STATUS=$?

echo ""
echo "--------------------------------------------------------------------------------"
echo "测试 2/2: PyTorch Distributed (MPI 后端)"
echo "--------------------------------------------------------------------------------"

# 设置 PyTorch distributed 环境变量
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# 使用 MPI 启动 PyTorch distributed
mpirun -n $NUM_PROCS python3 test_torch_distributed.py mpi 2>&1 | tee -a "$LOG_FILE"
TEST2_STATUS=$?

# 汇总结果
echo ""
echo "================================================================================"
echo "第五阶段 MPI 测试完成"
echo "================================================================================"
echo ""
echo "测试结果摘要:"
echo "  测试 1 (MPI 基线): $([ $TEST1_STATUS -eq 0 ] && echo '✓ 通过' || echo '✗ 失败')"
echo "  测试 2 (PyTorch MPI): $([ $TEST2_STATUS -eq 0 ] && echo '✓ 通过' || echo '✗ 失败')"
echo ""
echo "详细结果请查看: $RESULT_DIR"
echo "完整日志: $LOG_FILE"
echo ""

# 提示
echo "提示:"
echo "  - 在富岳上使用 pjsub 提交多节点作业进行完整测试"
echo "  - 对比 MPI 和 Gloo 的性能差异"
echo "  - 预期 MPI 在富岳 Tofu 互联上性能更优"
echo ""

# 返回状态
if [ $TEST1_STATUS -eq 0 ] && [ $TEST2_STATUS -eq 0 ]; then
    echo "✓ 所有测试通过"
    exit 0
else
    echo "✗ 部分测试失败"
    exit 1
fi

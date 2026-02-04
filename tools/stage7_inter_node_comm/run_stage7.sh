#!/bin/bash
# 第七阶段：机器间通信效率测试（两机两进程，PyTorch）
#
# 目的：两台机器各 1 个进程，测量跨节点通信的延迟与带宽。
# 仍使用 torch.distributed（Gloo 或 MPI 后端）。
#
# 使用方式：
#   1) 两机两进程（推荐 MPI）：
#      export MASTER_ADDR=第一台机器IP
#      export MASTER_PORT=29500
#      mpirun -np 2 -host node1,node2 python3 test_inter_node_torch.py mpi
#      或使用 hostfile：mpirun -np 2 --hostfile hosts.txt python3 test_inter_node_torch.py mpi
#
#   2) 单机两进程（仅作连通性/脚本检查）：
#      bash run_stage7.sh
#      → 在本机起 2 个进程，结果主要反映本机通信，非跨节点效率。

set -e
NUM_PROCS=2
BACKEND="${1:-mpi}"

echo "================================================================================"
echo "第七阶段：机器间通信效率（两台机器各 1 进程，PyTorch）"
echo "================================================================================"
echo "进程数: $NUM_PROCS (固定)"
echo "后端: $BACKEND"
echo ""

RESULT_DIR="../test_results/stage7"
mkdir -p "$RESULT_DIR"
LOG_FILE="$RESULT_DIR/stage7_$(date +%Y%m%d_%H%M%S).log"
echo "结果目录: $RESULT_DIR"
echo "日志文件: $LOG_FILE"
echo ""

# 单机时默认用 localhost；两机时需在运行前设置 MASTER_ADDR=第一台机器 IP
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-29500}"

if [ "$BACKEND" = "mpi" ]; then
    python3 -c "import mpi4py" 2>/dev/null || { echo "错误: 需要 mpi4py"; exit 1; }
    command -v mpirun &>/dev/null || { echo "错误: 找不到 mpirun"; exit 1; }
    echo "使用 MPI 启动 $NUM_PROCS 个进程..."
    mpirun -n $NUM_PROCS python3 test_inter_node_torch.py mpi 2>&1 | tee -a "$LOG_FILE"
else
    echo "使用 Gloo 时请在两台机器上分别启动 1 个进程，或单机用以下方式："
    echo "  机器1: MASTER_ADDR=本机IP RANK=0 WORLD_SIZE=2 python3 test_inter_node_torch.py gloo"
    echo "  机器2: MASTER_ADDR=机器1IP RANK=1 WORLD_SIZE=2 python3 test_inter_node_torch.py gloo"
    echo "单机自测（2 进程）："
    export WORLD_SIZE=$NUM_PROCS
    pids=()
    for ((rank=0; rank<NUM_PROCS; rank++)); do
        RANK=$rank python3 test_inter_node_torch.py gloo >> "$LOG_FILE" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait $pid; done
fi

STATUS=$?
echo ""
echo "================================================================================"
echo "第七阶段完成"
echo "================================================================================"
echo "结果: $([ $STATUS -eq 0 ] && echo '✓ 通过' || echo '✗ 失败')"
echo "详细结果: $RESULT_DIR/inter_node_torch.json"
echo "日志: $LOG_FILE"
echo ""
echo "两机真实带宽测试请在两台节点上使用 MPI："
echo "  mpirun -np 2 -host <节点1>,<节点2> python3 test_inter_node_torch.py mpi"
echo ""
exit $STATUS

export MASTER_ADDR=192.168.100.118
export MASTER_PORT=29500
export GLOO_SOCKET_IFNAME=eno1
export NCCL_SOCKET_IFNAME=eno1
export TP_SOCKET_IFNAME=eno1

export RANK=1
export WORLD_SIZE=2

echo "node1"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "GLOO_SOCKET_IFNAME: $GLOO_SOCKET_IFNAME"
echo "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
echo "TP_SOCKET_IFNAME: $TP_SOCKET_IFNAME"
echo "RANK: $RANK"
echo "WORLD_SIZE: $WORLD_SIZE"

echo "start node1"
python3 test_inter_node_torch.py

echo "start node1 with mpi"
mpirun -np 2 python3 test_inter_node_torch.py mpi

echo "done node1"

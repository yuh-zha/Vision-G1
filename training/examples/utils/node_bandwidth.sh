#!/bin/bash
#SBATCH --job-name=rl-reasoning
#SBATCH --partition=main
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=112
#SBATCH --mem=1512G
#SBATCH --output=slurm_log/node-test-%j.out
#SBATCH --error=slurm_log/node-test-%j.err
#SBATCH --exclusive


# module load perftest
#sSBATCH --exclude=g42-odin-h100-[001-100]

# get node list
nodes=($(scontrol show hostnames $SLURM_NODELIST))
num_nodes=${#nodes[@]}

# Print node information
echo "Number of nodes: $num_nodes"
echo "Node list: ${nodes[@]}"

# Array to store IP addresses
declare -a node_ips

# Get IP addresses for each node
echo "Fetching IP addresses for each node..."
for ((i=0; i<$num_nodes; i++)); do
    node=${nodes[$i]}
    # Get the IP address using hostname command
    ip=$(srun -N 1 -n 1 -w $node hostname -i | awk '{print $1}')
    node_ips[$i]=$ip
    echo "Node $i: $node - IP: $ip"
done

echo "All node IPs: ${node_ips[@]}"

# mpirun -np `python3 -c "print($node_count*8)"` -host "$IP_LIST" --oversubscribe -x NCCL_DEBUG=WARN -map-by slot ./build/all_reduce_perf -b 8 -e 1024M -f 8 -g 1
# Calculate total number of GPUs
total_gpus=$((num_nodes * 8))

# Create host list for mpirun using node hostnames instead of IPs
host_list=""
for node in "${nodes[@]}"; do
    if [ -z "$host_list" ]; then
        host_list="$node"
    else
        host_list="$host_list,$node"
    fi
done

echo "Starting NCCL bandwidth test across $num_nodes nodes with $total_gpus GPUs"
echo "Host list: $host_list"

# Create directory for logs if it doesn't exist
mkdir -p nccl_test_logs

# Run the NCCL all_reduce_perf test
mpirun -np $total_gpus \
    -host $host_list \
    --oversubscribe \
    -x NCCL_DEBUG=INFO \
    -map-by slot \
    $HOME/nccl-tests/build/all_reduce_perf \
    -b 8 \
    -e 1024M \
    -f 2 \
    -g 1 \
    | tee nccl_test_logs/nccl_bandwidth_test_$(date +%Y%m%d_%H%M%S).log

# Check if the mpirun command succeeded
if [ $? -eq 0 ]; then
    echo "NCCL bandwidth test was successful"
else
    echo "NCCL bandwidth test failed with exit code $?"
fi
echo "NCCL bandwidth test completed"


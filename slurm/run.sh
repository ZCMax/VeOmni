#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

# --- Style and Color Configuration ---
# Use tput to get terminal control sequences for better portability and safety.
# This will also check if the current terminal supports colors.
if tput setaf 1 >&/dev/null; then
    BOLD=$(tput bold)
    UNDERLINE=$(tput smul) # Start underline mode
    CYAN=$(tput setaf 6)
    GREEN=$(tput setaf 2)
    YELLOW=$(tput setaf 3)
    RED=$(tput setaf 1)
    RESET=$(tput sgr0)   # Reset all attributes
else
    # If tput fails, disable colors by setting variables to empty strings.
    BOLD=""
    UNDERLINE=""
    CYAN=""
    GREEN=""
    YELLOW=""
    RED=""
    RESET=""
fi

# Helper functions for formatted printing
print_header() {
    local title="$1"
    local underline
    underline=$(printf '%*s' "${#title}" '' | tr ' ' '─')
    printf "\n${BOLD}${CYAN}${UNDERLINE}%s${RESET}\n" "${title}"
    printf "${CYAN}%s${RESET}\n" "${underline}"
}

print_success() {
    printf "✅ ${GREEN}%s${RESET}\n" "$1"
}

print_info() {
    printf "ℹ️ %s\n" "$1"
}

print_warning() {
    printf "⚠️ ${YELLOW}%s${RESET}\n" "$1"
}

print_error() {
    printf "❌ ${RED}${BOLD}%s${RESET}\n" "$1" >&2
}
# --- End Style Configuration ---

export SHARE_ROOT="/mnt/inspurfs/eb3d_t/share"

# --- CFS Configuration ---
: "${USE_CFS:=false}"
export CFSCTL_PATH="${SHARE_ROOT}/cfs/bin/cfsctl"
export CFS_CONFIG_FILE="cfsd.cfg"
: "${CFS_PRELOAD:=false}"
# --- End CFS Configuration ---

if [ "$#" -eq 0 ]; then
    print_error "No command provided to execute."
    echo "Usage: $0 your_command --and --your --args"
    exit 1
fi

export TRITON_CACHE_DIR="${SHARE_ROOT}/.cache/${USER}_triton"
mkdir -p "$TRITON_CACHE_DIR"

# Safely check for GPUs
GPUS_PER_NODE=$(scontrol show job $SLURM_JOBID | grep 'TresPerNode=gpu:' | cut -d':' -f2 || echo 0)

if [ "$GPUS_PER_NODE" -gt 0 ]; then
    print_success "GPU job detected. Using $GPUS_PER_NODE processes per node."
    export NPROC_PER_NODE=$GPUS_PER_NODE
else
    print_info "CPU-only job detected. Using SLURM tasks per node."
    export NPROC_PER_NODE=${SLURM_NTASKS_PER_NODE:-1}
fi

export MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n 1`
export MASTER_PORT=$((RANDOM % 101 + 20000))
export WORLD_SIZE=$((NPROC_PER_NODE * SLURM_NNODES))
export HF_UPDATE_DOWNLOAD_COUNTS=false
export OMP_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=false
export FORCE_QWENVL_VIDEO_READER="torchcodec"
export no_proxy="localhost,127.0.0.1,10.0.0.0/8,100.0.0.0/8,35.220.164.252/32,.pjlab.org.cn"

if [ -n "${DEBUG}" ]; then
    print_warning "DEBUG mode is on. Enabling verbose NCCL logging."
    export NCCL_DEBUG=INFO
    export NCCL_IB_DISABLE=0
    export HYDRA_FULL_ERROR=1
fi

print_header "ENVIRONMENT VARIABLES (DEBUG)"
env | sort

# --- NCCL Configuration ---
print_header "NETWORK CONFIGURATION (NCCL)"
if [[ "$MASTER_ADDR" =~ SH-IDC1-10-140-0-[0-9]+ || "$MASTER_ADDR" =~ SH-IDC1-10-140-1-[0-9]+ ]]; then
    export NCCL_IB_HCA=mlx5_0
    export NCCL_SOCKET_IFNAME=eth0
    print_info "Server S1 detected. Set NCCL_IB_HCA=mlx5_0, NCCL_SOCKET_IFNAME=eth0"
elif [[ "$MASTER_ADDR" =~ SH-IDC1-10-140-24-[0-9]+ ]]; then
    export NCCL_IB_HCA=mlx5_0
    export NCCL_SOCKET_IFNAME=bond0
    print_info "Server S2 detected. Set NCCL_IB_HCA=mlx5_0, NCCL_SOCKET_IFNAME=bond0"
elif [[ "$MASTER_ADDR" =~ SH-IDC1-10-140-37-[0-9]+ ]]; then
    export NCCL_IB_HCA=mlx5_2,mlx5_3
    export NCCL_SOCKET_IFNAME=bond0
    print_success "Server P detected. Set NCCL_IB_HCA=mlx5_2,mlx5_3, NCCL_SOCKET_IFNAME=bond0"
elif [[ "$MASTER_ADDR" =~ SH-IDCA1404-10-140-54-[0-9]+ ]]; then
    export NCCL_IB_HCA=mlx5_2,mlx5_3
    export NCCL_SOCKET_IFNAME=bond0
    print_success "Server P2 detected. Set NCCL_IB_HCA=mlx5_2,mlx5_3, NCCL_SOCKET_IFNAME=bond0"
elif [[ "$MASTER_ADDR" =~ ^HOST-10-140-([0-9]{1,3})-([0-9]{1,3})$ ]]; then
    export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5
    export NCCL_SOCKET_IFNAME=bond0
    export NCCL_ALGO=Tree
    print_success "Server T detected. Set NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5, NCCL_SOCKET_IFNAME=bond0"
else
    print_warning "Unknown server type for IP: $MASTER_ADDR. No custom NCCL settings applied."
fi

# --- Job Summary ---
print_header "JOB CONFIGURATION"
printf "   %-20s : %s\n" "MASTER_ADDR" "$MASTER_ADDR"
printf "   %-20s : %s\n" "MASTER_PORT" "$MASTER_PORT"
printf "   %-20s : %s\n" "SLURM_NNODES" "$SLURM_NNODES"
printf "   %-20s : %s\n" "NPROC_PER_NODE" "$NPROC_PER_NODE"
printf "   %-20s : %s\n" "WORLD_SIZE" "$WORLD_SIZE"
printf "   %-20s : %s\n" "SLURM_JOB_ID" "$SLURM_JOB_ID"

# Construct the final command
TRAINING_CMD=(
    torchrun
    --nnodes=$SLURM_NNODES
    --nproc_per_node=$NPROC_PER_NODE
    --rdzv_id=$SLURM_JOB_ID
    --rdzv_backend=c10d
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT
)
TRAINING_CMD+=("$@")

# --- Main Execution Logic with CFS ---
if [ "$USE_CFS" = "true" ]; then
    print_header "CFS (CUSTOM FILE SYSTEM) - ENABLED"

    CFS_CMD_BASE=("$CFSCTL_PATH" -p "$SLURM_JOB_PARTITION" -n "$SLURM_NNODES" -X "$MASTER_ADDR" -s "$CFS_CONFIG_FILE")

    function cleanup {
        print_info "Performing CFS cleanup..."
        srun "${CFS_CMD_BASE[@]}" stop
        print_success "CFS cleanup finished."
    }
    trap cleanup EXIT SIGHUP SIGINT SIGTERM

    print_info "Running pre-emptive CFS stop for a clean state..."
    srun "${CFS_CMD_BASE[@]}" stop

    print_info "Starting CFS and mounting buckets..."
    srun "${CFS_CMD_BASE[@]}" start
    if [ $? -ne 0 ]; then
        print_error "CFS mount command failed. Aborting."
        exit 1
    fi
    print_success "CFS mounted successfully."

    if [ "$CFS_PRELOAD" = "true" ]; then
        print_info "Starting asynchronous data preload..."
        srun "${CFS_CMD_BASE[@]}" -a preload
    fi

    print_header "🚀 EXECUTING COMMAND"
    printf "${YELLOW}%s${RESET}\n" "${TRAINING_CMD[@]}"
    srun "${TRAINING_CMD[@]}"

else
    print_header "CFS (CUSTOM FILE SYSTEM) - DISABLED"
    print_header "🚀 EXECUTING COMMAND"
    printf "${YELLOW}%s${RESET}\n" "${TRAINING_CMD[@]}"
    exec srun "${TRAINING_CMD[@]}"
fi

print_header "SCRIPT FINISHED"

wandb login 9e37f762624801dfc332b03f2ecefbf87153ed8f # chenming's wandb
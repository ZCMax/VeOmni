#!/bin/bash

if [ -d "$HOME/anaconda3" ]; then
    CONDA_ROOT="$HOME/anaconda3"
    echo "Detected Anaconda at $CONDA_ROOT"
elif [ -d "$HOME/miniconda3" ]; then
    CONDA_ROOT="$HOME/miniconda3"
    echo "Detected Miniconda at $CONDA_ROOT"
else
    echo "Error: Neither Anaconda (anaconda3) nor Miniconda (miniconda3) found in your home directory ($HOME)."
    echo "Please ensure one is installed or manually set CONDA_ROOT."
    exit 1
fi

ENV_NAME=veomni

GCC_ROOT=/mnt/petrelfs/share/gcc/gcc-11.2.0
MPC_ROOT=/mnt/petrelfs/share/gcc/mpc-0.8.1
MPFR_ROOT=/mnt/petrelfs/share/gcc/mpfr-4.1.0
GMP_ROOT=/mnt/petrelfs/share/gcc/gmp-6.2.0
CUDA_ROOT=/mnt/petrelfs/share/cuda-12.4
CMAKE_ROOT=/mnt/petrelfs/share/cmake-3.30.9-linux-x86_64

# --- CUDA ---
export PATH=${CUDA_ROOT}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=${CUDA_ROOT}

# --- GCC & Libraries ---
export PATH=${GCC_ROOT}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${MPC_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=${MPFR_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_LIBRARY_PATH=${GMP_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# 强制使用新 GCC 作为默认 C/C++ 编译器
export CC=${GCC_ROOT}/bin/gcc
export CXX=${GCC_ROOT}/bin/g++

# --- CMAKE ---
export PATH=${CMAKE_ROOT}/bin${PATH:+:${PATH}}

# --- Conda ---
export PATH=${CONDA_ROOT}/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=${CONDA_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# --- Python User Base ---
export PYTHONUSERBASE=${HOME}/.local/${ENV_NAME}
export PATH=${PYTHONUSERBASE}/bin${PATH:+:${PATH}}


# --- Activate Conda Environment ---
source activate
conda deactivate
conda activate ${CONDA_ROOT}/envs/${ENV_NAME}

export LD_PRELOAD=${GCC_ROOT}/lib64/libstdc++.so.6:/mnt/lustre/share/glibc-2.27/lib/libm-2.27.so
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
# apptainer
export APPTAINER_CACHEDIR="/tmp"

echo "✅ Activated conda environment: ${ENV_NAME}"
echo "🛠️  Using: $(${CC} --version | head -n1)"
echo "🐍 Python: $(python --version 2>&1)"
echo "📦 CONDA_PREFIX: $CONDA_PREFIX"

#!/bin/bash
#SBATCH -J ML-torch
#SBATCH -e out_%j
#SBATCH -o out_%j

### select gpu
#SBATCH -p cas_v100_2  # queue name
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
###allocate node
#SBATCH -N 1
#SBATCH --nodes=1 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --comment pytorch

source /scratch/x3100a06/miniconda3/etc/profile.d/conda.sh
conda activate cignn_env

module load cmake/3.26.2
module load gcc/10.2.0 mpi/openmpi-4.1.1
module load cuda/12.1
module load conda/pytorch_2.5.0

export VIRTUAL_ENV=/scratch/x3100a06/miniconda3/envs/cignn_env
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH="$VIRTUAL_ENV"
export OMPI_CXX="$(which g++)"

echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "CUDA_HOME: $CUDA_HOME"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA_VERSION: $CUDA_VERSION"

cd $SLURM_SUBMIT_DIR

CODE_DIR=.

mpirun -np 1 /scratch/x3100a06/MLP_Paper/CIGNN/cnmp/lammps/build/lmp -in /scratch/x3100a06/MLP_Paper/CIGNN/cnmp/example/run.lammps

conda deactivate
exit 0
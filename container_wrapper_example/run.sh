#!/bin/bash -l
#SBATCH --job-name=pytorch_example     # Job name
#SBATCH --output=logs/log_test.out      # Name of stdout output file
#SBATCH --error=logs/log_test.err       # Name of stderr error file
#SBATCH --partition=dev-g               # partition name
#SBATCH --nodes=2                       # Total number of nodes 
#SBATCH --ntasks-per-node=1             # MPI ranks per node
#SBATCH --gpus-per-node=1               # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=7               # CPU cores per task
#SBATCH --mem=480G                      # Total memory for job
#SBATCH --time=0-02:00:00               # Run time (d-hh:mm:ss)


# Path to the environment. Same as INSTALL_DIR in create_environment.sh
ENV_DIR=/projappl/$SLURM_JOB_ACCOUNT/pytorch_example_tykky
source ~/.bashrc

# Load the required modules
module purge
module load LUMI

export PATH="$ENV_DIR/bin:$PATH"

srun python -m pytorch_example.run \
        --num_gpus=$SLURM_GPUS_ON_NODE \
        --num_nodes=$SLURM_JOB_NUM_NODES \


#!/bin/bash -l
#SBATCH --job-name=lumi_env_build     # Job name
#SBATCH --output=logs/log_build.out      # Name of stdout output file
#SBATCH --error=logs/log_build.err       # Name of stderr error file
#SBATCH --partition=small               # partition name
#SBATCH --nodes=1                       # Total number of nodes 
#SBATCH --ntasks-per-node=1             # MPI ranks per node
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G                      # Total memory for job
#SBATCH --time=0-00:20:00               # Run time (d-hh:mm:ss)

# Do not use this script to build environments - this is just an example.
# Use slurm_tools/create_environment.sh instead.

module purge
module load LUMI
module load lumi-container-wrapper

# Where to store the containerized environment
INSTALL_DIR=/projappl/$SLURM_JOB_ACCOUNT/pytorch_example_tykky

# Path to the package being developed
PKG_DIR=/scratch/$SLURM_JOB_ACCOUNT/LUMI-Project-Template

# Path to the conda environment file
ENV_FILE_PATH=$PKG_DIR/environment.yml

conda-containerize new \
    --mamba \
    --prefix $INSTALL_DIR \
    --post-install $PKG_DIR/container_wrapper_example/post_install.sh \
    $ENV_FILE_PATH

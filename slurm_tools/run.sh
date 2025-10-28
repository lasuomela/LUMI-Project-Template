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

# Reference:
# https://github.com/Lumi-supercomputer/Getting_Started_with_AI_workshop/blob/main/07_Extending_containers_with_virtual_environments_for_faster_testing/examples/extending_containers_with_venv.md

# Path to the environment. Same as INSTALL_DIR in create_environment.sh
ENV_DIR=/projappl/$SLURM_JOB_ACCOUNT/pytorch_example

# Name of the image to to use
IMAGE_NAME=pytorch_example.sif

# Load the required modules
module purge
module load LUMI
module use  /appl/local/containers/ai-modules
module load singularity-AI-bindings

source ~/.bashrc
export SINGULARITYENV_PREPEND_PATH=/user-software/bin # gives access to packages inside the container

# Tell RCCL to use only Slingshot interfaces and GPU RDMA
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

# Run the training script:
# We use the Singularity container from 'create_environment.sh'
# with the --bind option to mount the virtual environment in $ENV_DIR/myenv.sqsh
# into the container at /user-software.
srun singularity exec \
   -B $ENV_DIR/myenv.sqsh:/user-software:image-src=/ $ENV_DIR/$IMAGE_NAME \
    python -m pytorch_example.run \
        --num_gpus=$SLURM_GPUS_ON_NODE \
        --num_nodes=$SLURM_JOB_NUM_NODES \


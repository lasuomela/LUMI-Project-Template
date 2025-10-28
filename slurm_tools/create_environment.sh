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

# Reference:
# https://github.com/Lumi-supercomputer/Getting_Started_with_AI_workshop/blob/main/07_Extending_containers_with_virtual_environments_for_faster_testing/examples/extending_containers_with_venv.md

# Name of the image to create
IMAGE_NAME=pytorch_example.sif

# Path to the package being developed
PKG_DIR=/scratch/$SLURM_JOB_ACCOUNT/LUMI-Project-Template

# Where to store the image
INSTALL_DIR=/projappl/$SLURM_JOB_ACCOUNT/pytorch_example

# Path to your conda environment file
ENV_FILE_PATH=$PKG_DIR/environment.yml

# Path to the environment base image. Choose a ROCm version that matches your needs.
# On the base images, RCCL is properly configured to use the high-speed Slingshot-11 interconnect
# between nodes. This ensures optimal performance when training across multiple nodes.
BASE_IMAGE_PATH=/appl/local/containers/sif-images/lumi-rocm-rocm-6.2.4.sif

# Remove the old environment
if [ -d "$INSTALL_DIR" ]; then
    rm -rf $INSTALL_DIR
fi
mkdir -p $INSTALL_DIR

# Purge modules and load cotainr module
module purge
module load LUMI/24.03 cotainr

# Install the conda/pip dependencies specified in the .yml file
srun cotainr build $INSTALL_DIR/$IMAGE_NAME \
    --base-image=$BASE_IMAGE_PATH \
    --conda-env=$ENV_FILE_PATH \
    --accept-license

# Stuff beyond here is optional but useful

# Load modules needed for running Singularity containers
module use  /appl/local/containers/ai-modules
module load singularity-AI-bindings

# Create a virtual environment to install stuff that cannot be installed via conda
# such as an editable install of the package being developed
# or packages you want to install with the --no-deps flag
singularity exec $INSTALL_DIR/$IMAGE_NAME bash -c "
  python -m venv $INSTALL_DIR/myenv --system-site-packages &&
  source $INSTALL_DIR/myenv/bin/activate &&
  pip install git+https://github.com/bdaiinstitute/theia.git --no-deps &&
  pip install -e $PKG_DIR &&
  deactivate
"

# Create a SquashFS image of the virtual environment
mksquashfs $INSTALL_DIR/myenv $INSTALL_DIR/myenv.sqsh
rm -rf $INSTALL_DIR/myenv
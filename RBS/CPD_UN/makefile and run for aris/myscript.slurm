#!/bin/bash

####################################
#     ARIS slurm script template   #
#                                  #
# Submit script: sbatch filename   #
#                                  #
####################################

#SBATCH --job-name=ntf    # Job name
#SBATCH --output=ntf.%j.out # Stdout (%j expands to jobId)
#SBATCH --error=ntf.%j.err # Stderr (%j expands to jobId)
#SBATCH --ntasks=171     # Number of processor cores (i.e. tasks)
#SBATCH --tasks-per-node=40
#SBATCH --cpus-per-task=1     # Threads per task
#SBATCH --time=72:00:00   # walltime
#SBATCH --partition=fat    # Partition
#SBATCH --account=pa170403    # Accounting project


export I_MPI_FABRICS=shm:dapl

if [ x$SLURM_CPUS_PER_TASK == x ]; then
  export OMP_NUM_THREADS=1
else
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi


## LOAD MODULES ##
module purge		# clean up loaded modules 

# load necessary modules
module load gnu/4.9.2
module load intel/15.0.3
module load intelmpi/5.0.3
module load cuda/7.5.18

## RUN YOUR PROGRAM ##
srun ntf

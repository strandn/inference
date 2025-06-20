#!/bin/bash
 
#SBATCH --job-name=inference_exp_restart
#SBATCH --output=%x.out
#SBATCH --error=%x.err
 
#SBATCH --time=2-0:00:00
 
#SBATCH --partition=dinner
#SBATCH --account=pi-dinner
 
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
 
#SBATCH --export=NONE

export JULIA_NUM_THREADS=1

echo $SLURM_JOB_NAME
echo $SLURM_JOB_NODELIST

mpiexecjl -n 48 julia exp_restart.jl

#!/bin/bash

#SBATCH --job-name=multibond
#SBATCH --account=pi-depablo
#SBATCH --output=run.out
#SBATCH --error=run.err
#SBATCH --time=35:59:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=48G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gm4-pmext
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --qos=gm4

module load openmpi/4.1.1+gcc-10.1.0 cuda/11.2

ulimit -l unlimited
ulimit -s unlimited

cp ../$(( $(basename "$(pwd)") - 1 ))/restart2.pkl ./restart.pkl

PYTHON_EXECUTABLE=/project/depablo/acqin2/environments/steered_pysages/bin/python
mpirun -np 1 ${PYTHON_EXECUTABLE} steered.py

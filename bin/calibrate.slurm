#!/bin/bash
#SBATCH --job-name=mktest_redmapper_calibrate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --account=des
#SBATCH --qos=interactive
#SBATCH --partition=cpu
#SBATCH --licenses=cfs,SCRATCH
#SBATCH --output=/pscratch/sd/m/mkwiecie/redmapper_try1/logs/redmapper_calibrate.log
#SBATCH --error=/pscratch/sd/m/mkwiecie/redmapper_try1/logs/redmapper_calibrate.err

# Queue with sbatch calibrate.slurm

module load shifter

# Redmapper shifter image
SHIFTER=docker:ghcr.io/erykoff/redmapper:0.8.5.5

echo '>>>> Updating and loading the Shifter image '$SHIFTER
shifterimg pull $SHIFTER

echo '>>>> Entering Shifter container...'
shifter --module=mpich --image $SHIFTER /bin/bash

echo '>>>> Activating the conda environment...'
source /opt/redmapper/startup.sh

echo '>>>> Running Redmapper calibration...'
python3 /global/homes/m/mkwiecie/des/repos/redmapper/bin/redmapper_calibrate.py -c cal.yml

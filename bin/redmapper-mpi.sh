#! /bin/bash

<<comment
Shell script for running redmapper using MPI+shifter at NERSC. 

ALLOCATE
salloc -N 1 -C cpu -A des -L cfs,SCRATCH -t 02:00:00 --qos interactive --image=ghcr.io/erykoff/redmapper:0.8.5.5

CALIBRATION
srun -n 1 -c 256 shifter --module=mpich $REDMAPPER_DIR/bin/redmapper-mpi.sh calibrate

comment
# Setup redmapper install dir
export REDMAPPER_DIR=/global/homes/m/mkwiecie/des/repos/redmapper

# Run redmapper
export stage=$1

if [ $stage = "calibrate" ]; then
    time python $REDMAPPER_DIR/bin/redmapper_calibrate.py -c cal.yml
else
    echo "Unrecognized stage "$stage
fi
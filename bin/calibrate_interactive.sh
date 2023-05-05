#!/bin/bash

<<comment
Shell script for running redmapper using MPI+shifter at NERSC. 

ALLOCATE 1 node (256 available cores)
salloc -N 1 -C cpu -A des -L cfs,SCRATCH -t 02:00:00 --qos interactive --image=ghcr.io/erykoff/redmapper:0.8.5.5
comment

SHIFTER=docker:ghcr.io/erykoff/redmapper:0.8.5.5

echo 'Updating and loading the shifter image '$SHIFTER
shifterimg pull $SHIFTER

echo 'Entering shifter container...'
shifter --module=mpich --image $SHIFTER /bin/bash

echo 'Activating the conda environment...'
source /opt/redmapper/startup.sh

echo 'Running redmapper calibration...'
python3 /global/homes/m/mkwiecie/des/repos/redmapper/bin/redmapper_calibrate.py -c cal.yml

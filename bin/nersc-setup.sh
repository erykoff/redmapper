module load shifter

SHIFTER=docker:ghcr.io/erykoff/redmapper:0.8.5.5

shifterimg pull $SHIFTER

shifter --module=mpich --image $SHIFTER /bin/bash

# Source the redmapper conda environment
source /opt/redmapper/startup.sh


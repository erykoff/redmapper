SHIFTER=docker:ghcr.io/erykoff/redmapper:0.8.5.5

# salloc -N 1 -C cpu -A des -L cfs,SCRATCH -t 02:00:00 --qos interactive --image=$SHIFTER

# Calibrate
shifter --module=mpich --image $SHIFTER /bin/bash -c ". /opt/redmapper/startup.sh && /global/homes/m/mkwiecie/des/repos/redmapper/bin/redmapper_calibrate.py -c cal.yml"

# ZRed
# Edit $HOME/.redmapper_batch.yaml then run below
shifter --module=mpich --image $SHIFTER /bin/bash -c ". /opt/redmapper/startup.sh && /global/homes/m/mkwiecie/des/repos/redmapper/bin/redmapper_batch.py -c run.yml -b NERSC -N 1 -r 1"

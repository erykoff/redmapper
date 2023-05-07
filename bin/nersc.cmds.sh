<<comment
    On nersc, redmapper works through a docker container.  The workflow is instantiate the docker container, and pass
    the command for redmapper to the shell immediately after initialization.  That is what is done below.

    Commands which call redmapper_batch.py generate a job file to submit to slurm.

    To test batch scripts, use interactive mode:
    salloc -N 1 -C cpu -A des -L cfs,SCRATCH -t 02:00:00 --qos interactive --image=docker:ghcr.io/erykoff/redmapper:0.8.5.5

    NOTE: redmapper on the container does not support NERSC/slurm.  Reference personal repo with this implementation.
comment
# 1. Calibrate red sequence
shifter --module=mpich --image docker:ghcr.io/erykoff/redmapper:0.8.5.5 /bin/bash -c ". /opt/redmapper/startup.sh && /global/homes/m/mkwiecie/des/repos/redmapper/bin/redmapper_calibrate.py -c cal.yml"

# 2. Run zred 
# Edit $HOME/.redmapper_batch.yaml then run below
shifter --module=mpich --image docker:ghcr.io/erykoff/redmapper:0.8.5.5 /bin/bash -c ". /opt/redmapper/startup.sh && /global/homes/m/mkwiecie/des/repos/redmapper/bin/redmapper_batch.py -c run.yml -b NERSC -N 1 -r 1"

# 3. Recalc background after zred
shifter --module=mpich --image docker:ghcr.io/erykoff/redmapper:0.8.5.5 /bin/bash -c "source /opt/redmapper/startup.sh && redmapper_make_zred_bkg.py -c run.yml"

# 4. RUN REDMAPPER!
shifter --module=mpich --image docker:ghcr.io/erykoff/redmapper:0.8.5.5 /bin/bash -c "source /opt/redmapper/startup.sh && /global/homes/m/mkwiecie/des/repos/redmapper/bin/redmapper_batch.py -c run.yml -r 0 -b NERSC -N 1"

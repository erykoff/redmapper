#!/usr/bin/env python
"""
Create a batch configuration script to submit to a cluster.
"""

from __future__ import division, absolute_import, print_function

import os
import argparse
import yaml
import healpy as hp
import numpy as np
import glob

import redmapper
import redmapper.parsl_templates as parsl_templates

def create_batchconfig(filename):
    with open(filename, 'w') as f:
        f.write("""
batchname:
   setup: ''
   batch: 'lsf'
   requirements: ''
""")

def load_batchconfig(filename):
    """
    Load a batch configuration file.

    Parameters
    ----------
    filename: `str`
       Filename of batch configuration file

    Returns
    -------
    yaml_data: `dict`
       Dict of parameters from configuration file.
    """
    with open(filename) as f:
        yaml_data = yaml.load(f, Loader=yaml.SafeLoader)

    for key in yaml_data.keys():
        if 'batch' not in yaml_data[key]:
            raise ValueError("Missing 'batch' key for %s section in %s." % (key, filename))
        if 'setup' not in yaml_data[key]:
            yaml_data[key]['setup'] = ''
        if 'requirements' not in yaml_data[key]:
            yaml_data[key]['requirements'] = ''
        if 'parsl_provider' not in yaml_data[key]:
            yaml_data[key]['parsl_provider'] = 'local'
        if 'image' not in yaml_data[key]:
            yaml_data[key]['image'] = ''
        if 'constraint' not in yaml_data[key]:
            yaml_data[key]['constraint'] = ''
        if 'qos' not in yaml_data[key]:
            yaml_data[key]['qos'] = ''

    return yaml_data


batchconfigfile = os.path.join(os.environ['HOME'], '.redmapper_batch.yml')
if not os.path.isfile(batchconfigfile):
    create_batchconfig(batchconfigfile)
    print("Please edit %s with batch configuration and rerun." % (batchconfigfile))

batchconfig = load_batchconfig(batchconfigfile)

if len(batchconfig) > 1:
    mode_required = True
else:
    mode_required = False

parser = argparse.ArgumentParser(description="Create a batch file for running redmapper codes")

parser.add_argument('-c', '--configfile', action='store', type=str, required=True,
                    help='YAML config file')
parser.add_argument('-r', '--runmode', action='store', type=int, required=True,
                    help='Run mode.  0 is full finder run.  1 is zred run.')
parser.add_argument('-b', '--batchmode', action='store', type=str, required=mode_required,
                    help='Batch mode, defined in ~/.redmapper_batch.yml')
parser.add_argument('-w', '--walltime', action='store', type=int, required=False,
                    help='Wall time (override default)')
parser.add_argument('-n', '--nside', action='store', type=int, required=False,
                    help='Parallelization nside (optional, can use default)')
parser.add_argument('-N', '--nodes', action='store', type=int, required=False,
                    default=2, help='Number of nodes to run (for nersc)')

args = parser.parse_args()

if not mode_required and args.batchmode is None:
    batchmode = list(batchconfig.keys())[0]
else:
    batchmode = args.batchmode

# Read in the config file

config = redmapper.Configuration(args.configfile)

if len(config.hpix) != 0:
    raise ValueError("Cannot run redmapper in batch mode with hpix not an empty list (full sky)")

# Check the nside

nside = args.nside

if args.runmode == 0:
    # This is a full run
    if nside is None:
        nside = 8
    jobtype = 'run'
    default_walltime = 72*60
    memory = 6000
elif args.runmode == 1:
    # This is a zred run
    if nside is None:
        nside = 8
    jobtype = 'zred'
    default_walltime = 5*60
    memory = 2000
elif args.runmode == 2:
    # This is a runcat run
    if nside is None:
        nside = 8
    jobtype = 'runcat'
    default_walltime = 10*60
    memory = 4000
elif args.runmode == 3:
    # This is a random/zmask run
    if nside is None:
        nside = 8
    jobtype = 'zmask'
    default_walltime = 10*60
    memory = 4000
elif args.runmode == 4:
    # This is a zscan run
    if nside is None:
        nside = 8
    jobtype = 'zscan'
    default_walltime = 10*60
    memory = 4000
else:
    raise RuntimeError("Unsupported runmode: %d" % (args.runmode))

if args.walltime is None:
    walltime = default_walltime
else:
    walltime = args.walltime

jobname = '%s_%s' % (config.outbase, jobtype)

# Determine which pixels overlap the galaxy file...

tab = redmapper.Entry.from_fits_file(config.galfile)

theta, phi = hp.pix2ang(tab.nside, tab.hpix)
hpix_run = np.unique(hp.ang2pix(nside, theta, phi))

# Make the batch script in a "jobs" directory

cwd = os.getcwd()
jobpath = os.path.join(cwd, 'jobs')

if not os.path.isdir(jobpath):
    os.makedirs(jobpath)

# Will want to check for previous (failed) jobs
if batchconfig[batchmode]['batch'] == 'parsl':
    test = glob.glob(os.path.join(jobpath, '%s_?.py' % (jobname)))
else:
    test = glob.glob(os.path.join(jobpath, '%s_?.job' % (jobname)))
index = len(test)

need_maskgals = False

if args.runmode == 0:
    # Run in the directory where the config file is, by default
    run_command = 'redmapper_run_redmapper_pixel.py -c %s -p %%s -n %d -d %s' % (
        (os.path.abspath(args.configfile),
         nside,
         os.path.dirname(os.path.abspath(args.configfile))))
    need_maskgals = True
elif args.runmode == 1:
    run_command = 'redmapper_run_zred_pixel.py -c %s -p %%s -n %d -d %s' % (
        (os.path.abspath(args.configfile),
         nside,
         os.path.dirname(os.path.abspath(args.configfile))))
elif args.runmode == 2:
    run_command = 'redmapper_runcat_pixel.py -c %s -p %%s -n %d -d %s' % (
        (os.path.abspath(args.configfile),
         nside,
         os.path.dirname(os.path.abspath(args.configfile))))
    need_maskgals = True
elif args.runmode == 3:
    run_command = 'redmapper_run_zmask_pixel.py -c %s -p %%s -n %d -d %s' % (
        (os.path.abspath(args.configfile),
         nside,
         os.path.dirname(os.path.abspath(args.configfile))))
    need_maskgals = True
elif args.runmode == 4:
    run_command = 'redmapper_run_zscan_pixel.py -c %s -p %%s -n %d -d %s' % (
        (os.path.abspath(args.configfile),
         nside,
         os.path.dirname(os.path.abspath(args.configfile))))
    need_maskgals = True

if need_maskgals:
    # Check to see if maskgals are there, and generate them if not.
    if not os.path.isfile(config.maskgalfile):
        print("Did not find maskgalfile %s.  Generating now." % (config.maskgalfile))
        mask = redmapper.mask.get_mask(config, include_maskgals=False)
        mask.gen_maskgals(config.maskgalfile)

if batchconfig[batchmode]['batch'] == 'parsl':
    jobfile = os.path.join(jobpath, '%s_%d.py' % (jobname, index + 1))
elif batchconfig[batchmode]['batch'] == 'slurm':
    jobfile = os.path.join(jobpath, '%s_%d.slurm' % (jobname, index + 1))
else:
    jobfile = os.path.join(jobpath, '%s_%d.job' % (jobname, index + 1))

with open(jobfile, 'w') as jf:
    write_jobarray = True
    if (batchconfig[batchmode]['batch'] == 'lsf'):
        # LSF mode
        jf.write("#BSUB -R '%s'\n" % (batchconfig[batchmode]['requirements']))
        jf.write("#BSUB -R 'rusage[mem=%d]'\n" % (memory))
        jf.write("#BSUB -J %s[1-%d]\n" % (jobname, hpix_run.size))
        jf.write("#BSUB -oo %s\n" % (os.path.join(jobpath, '%s_%%J_%%I.log' % (jobname))))
        jf.write("#BSUB -n 1\n")
        jf.write("#BSUB -W %d\n\n" % (walltime))

        index_string = '${pixarr[LSB_JOBINDEX-1]}'

    elif (batchconfig[batchmode]['batch'] == 'pbs'):
        # PBS mode
        ppn = batchconfig[batchmode]['ppn']
        n_nodes = int(np.ceil(float(hpix_run.size) / float(ppn)))
        jf.write("#PBS -q %s\n" % (batchconfig[batchmode]['queue']))
        jf.write("#PBS -l nodes=%d:ppn=%d\n" % (n_nodes, ppn))
        jf.write("#PBS -l walltime=%d:00:00\n" % (int(walltime / 60)))
        jf.write("#PBS -l mem=%dmb\n" % (memory))
        jf.write("#PBS -j oe\n")
        jf.write('N_CPU=%d\n' % (n_nodes * batchconfig[batchmode]['ppn']))
    elif (batchconfig[batchmode]['batch'] == 'parsl'):
        write_jobarray = False

        if batchconfig[batchmode]['parsl_provider'] == 'local':
            parsl_config = parsl_templates.PARSL_LOCAL_CONFIG_TEMPLATE
        elif batchconfig[batchmode]['parsl_provider'] == 'slurm':
            parsl_config = parsl_templates.PARSL_SLURM_CONFIG_TEMPLATE.format(
                nodes=args.nodes,
                constraint=batchconfig[batchmode]['constraint'],
                qos=batchconfig[batchmode]['qos'],
                walltime=walltime
            )
        else:
            raise RuntimeError("Invalid parsl_provider (requires either local or slurm).")

        cmd = run_command % ('{pixel}')
        if batchconfig[batchmode]['image'] != '':
            # We are using a shifter image
            image = batchconfig[batchmode]['image']
            parsl_command = f'shifter --image={image} /bin/bash -c ". /opt/redmapper/startup.sh && {cmd}"'
        else:
            # No shifter image
            parsl_command = cmd

        hpix_run_str = [str(hpix) for hpix in hpix_run]
        hpix_list_str = "[" + ', '.join(hpix_run_str) + "]"

        parsl_script = parsl_templates.PARSL_RUN_TEMPLATE.format(
            parsl_config=parsl_config,
            parsl_command=parsl_command,
            memory=memory,
            hpix_list_str=hpix_list_str,
            jobname=jobname
        )

        jf.write(parsl_script)

    elif (batchconfig[batchmode]['batch'] == 'slurm'):

        mem_per_node = int(batchconfig[batchmode]['mpn'])
        pixels_per_node = int(np.floor(mem_per_node / memory))
        total_pixels = hpix_run.size - 1
        total_tasks = int(np.floor(total_pixels / pixels_per_node))

        jf.write("#!/bin/sh\n")
        jf.write("#SBATCH --job-name=%s[0-%d]\n"    % (jobname, total_tasks))
        jf.write("#SBATCH --mem=%d\n"               % (mem_per_node))
        jf.write("#SBATCH --array=0-%d\n"           % (total_tasks))
        # Since redmapper doesn't use MPI, we can only run 1 process per node, and even if we add more nodes
        # we can't allocate more threads to that process, so we only ever allocate 1 node.
        jf.write("#SBATCH --nodes=1\n")
        jf.write("#SBATCH --ntasks-per-node=1\n")   
        jf.write("#SBATCH --cpus-per-task=%d\n"     % (int(batchconfig[batchmode]['ncpu'])))   
        jf.write("#SBATCH --time=%d:00:00\n"        % (int(walltime / 60)))
        jf.write("#SBATCH --account=%s\n"           % (batchconfig[batchmode]['account']))
        jf.write("#SBATCH --qos=%s\n"               % (batchconfig[batchmode]['qos']))
        jf.write("#SBATCH --constraint=cpu\n")
        jf.write("#SBATCH --licenses=%s\n"          % (batchconfig[batchmode]['licenses']))
        jf.write("#SBATCH --image=%s\n"             % (batchconfig[batchmode]['image']))
        jf.write(f"#SBATCH --output=%s\n\n"         % (str(os.path.join(jobpath, '%s-%%A-%%a.log' % (jobname)))))

        jf.write("module load parallel\n")

        # In parallel on each node, spin up a docker container for each pixel and execute the requested redmapper script.
        run_command = f'parallel -j {pixels_per_node:d} shifter --image={batchconfig[batchmode]["image"]} "/bin/bash -c \'source /opt/redmapper/startup.sh && {run_command}\'" ::: "${{TASK_PIX[@]}}"'

        # We need these commands to be run AFTER pixarr is added, so we add them to run_command (in reverse!) instead of jobfile
        run_command=('TASK_PIX=("${pixarr[@]:$TASK_START:$NUM_PIX_PER_NODE}")\n')+run_command
        run_command=("TASK_END=$(($TASK_START + $NUM_PIX_PER_NODE))\n")+run_command
        run_command=("TASK_START=$(($SLURM_ARRAY_TASK_ID * $NUM_PIX_PER_NODE))\n")+run_command
        run_command=("NUM_PIX_PER_NODE=%d\n" % pixels_per_node)+run_command
    
        index_string = '{}'

    else:
        # Nothing else supported
        raise RuntimeError("Only LSF, PBS, SLURM, parsl/slurm, and parsl/local supported at this time.")

    if write_jobarray:
        jf.write("pixarr=(")
        for hpix in hpix_run:
            jf.write("%d " % (hpix))
        jf.write(")\n\n")

        jf.write("%s\n\n" % (batchconfig[batchmode]['setup']))

        cmd = run_command % (index_string)
        jf.write("%s\n" % (cmd))

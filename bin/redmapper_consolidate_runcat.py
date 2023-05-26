#!/usr/bin/env python

from __future__ import division, absolute_import, print_function

import os
import sys
import argparse
import redmapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Consolidate a redMaPPer parallel run')

    parser.add_argument('-c', '--configfile', action='store', type=str, required=True,
                        help='YAML config file')
    parser.add_argument('-r', '--randommode', action='store_true', help='Consolidate randoms')
    parser.add_argument('-t', '--cattype', action='store', type=str, required=False, default='',
                        help='File type used in runcat')

    args = parser.parse_args()

    if args.randommode:
        do_plots = False
        match_spec = False
        consolidate_members = False
        cattype = 'randoms_zmask' if args.cattype == '' else args.cattype
    else:
        do_plots = True
        match_spec = True
        consolidate_members = True
        cattype = 'runcat' if args.cattype == '' else args.cattype

    consolidate = redmapper.pipeline.RuncatConsolidateTask(args.configfile)
    consolidate.run(do_plots=do_plots, match_spec=match_spec, consolidate_members=consolidate_members, cattype=cattype)

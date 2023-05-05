#!/usr/bin/env python

from __future__ import division, absolute_import, print_function

import os
import sys
import argparse
import redmapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute zreds for a single pixel')

    parser.add_argument('-c', '--configfile', action='store', type=str, required=True,
                        help='YAML config file')
    parser.add_argument('-p', '--pixel', action='store', type=int, required=True,
                        help='Healpix pixel to run')
    parser.add_argument('-n', '--nside', action='store', type=int, required=True,
                        help='Healpix nside for pixel')
    parser.add_argument('-d', '--path', action='store', type=str, required=False, default='./',
                        help='Path to set for outputs')

    args = parser.parse_args()

    runZredPixelTask = redmapper.pipeline.RunZredPixelTask(args.configfile, args.pixel, args.nside, path=args.path)
    runZredPixelTask.run()


#!/usr/bin/env python
"""
Predict memory usage on parallelized runs.
"""

import argparse

import redmapper


parser = argparse.ArgumentParser(description="Predict memory usage when running in parallel.")

parser.add_argument('-c', '--configfile', action='store', type=str, required=True,
                    help='YAML config file')
parser.add_argument('-n', '--no_zred', action='store_true',
                    help='Do not include zred.')
parser.add_argument('-b', '--border_factor', action='store', type=float, required=False,
                    default=2.0, help='Approximate factor for border mem usage.')

args = parser.parse_args()

mem_predictor = redmapper.pipeline.MemPredict(args.configfile)

mem_predictor.predict_memory(include_zreds=not args.no_zred, border_factor=args.border_factor)


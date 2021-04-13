#!/usr/bin/env python

import subprocess
import argparse

import redmapper


parser = argparse.ArgumentParser(description="Create a docker image for redmapper.")
parser.add_argument('-v', '--versiontag', action='store', type=str, required=False, default='',
                    help='Tag; optional, overriding the tag generated from current __version__')

args = parser.parse_args()

if args.versiontag == '':
    versiontag = redmapper.__version__
else:
    versiontag = args.versiontag

print("Building docker for redmapper %s" % (versiontag))

subprocess.call(['docker', 'build',
                 '-t', 'erykoff/redmapper:%s' % (versiontag),
                 '--build-arg', 'TAG=%s' % (versiontag),
                 '.'])




#!/bin/bash
# Load the redmapper container using shifter
SHIFTER=docker:ghcr.io/erykoff/redmapper:0.8.5.5
echo 'Updating and loading the shifter image '$SHIFTER
shifterimg pull $SHIFTER
shifter --module=mpich --image $SHIFTER bash
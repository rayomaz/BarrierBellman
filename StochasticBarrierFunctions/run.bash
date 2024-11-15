#!/bin/bash

set -e

case "$1" in
    contraction) 
        echo "Running $1"
        julia --project=/StochasticBarrierFunctions -e 'yaml_file="benchmarks/linear/systems/contraction/sos.yaml"; include("barrier_synthesis.jl")'
        ;;
esac
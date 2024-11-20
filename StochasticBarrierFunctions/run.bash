#!/bin/bash

set -e

case "$1" in
    contraction_sos) 
        echo "Running $1"
        julia --project=/StochasticBarrierFunctions -e 'yaml_file="benchmarks/linear/systems/contraction/sos.yaml"; include("benchmarks/barrier_synthesis.jl")'
        ;;

    contraction_pwc_dual) 
        echo "Running $1"
        julia --project=/StochasticBarrierFunctions -e 'yaml_file="benchmarks/linear/systems/contraction/pwc.yaml"; include("benchmarks/barrier_synthesis.jl")'
        ;;

    population_sos) 
        echo "Running $1"
        julia --project=/StochasticBarrierFunctions -e 'yaml_file="benchmarks/linear/systems/population/sos.yaml"; include("benchmarks/barrier_synthesis.jl")'
        ;;

    oscillator_sos) 
        echo "Running $1"
        julia --project=/StochasticBarrierFunctions -e 'yaml_file="benchmarks/polynomial/systems/oscillator/sos.yaml"; include("benchmarks/barrier_synthesis.jl")'
        ;;

    roomtemperature_sos) 
        echo "Running $1"
        julia --project=/StochasticBarrierFunctions -e 'yaml_file="benchmarks/polynomial/systems/roomtemperature/sos.yaml"; include("benchmarks/barrier_synthesis.jl")'
        ;;

    pendulum_sos) 
        echo "Running $1"
        julia --project=/StochasticBarrierFunctions -e 'yaml_file="benchmarks/nonlinear/systems/pendulum/sos.yaml"; include("benchmarks/barrier_synthesis.jl")'
        ;;

    pendulum_pwc) 
        echo "Running $1"
        julia --project=/StochasticBarrierFunctions -e 'yaml_file="benchmarks/nonlinear/systems/pendulum/pwc.yaml"; include("benchmarks/barrier_synthesis.jl")'
        ;;
esac
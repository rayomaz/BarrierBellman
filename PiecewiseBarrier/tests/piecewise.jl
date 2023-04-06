""" Piecewise Barrier Functions based on Bellman's Equation

    © Rayan Mazouz

"""

# Import packages
using Revise, BenchmarkTools

using PiecewiseBarrier
using DelimitedFiles

# State partitions
system_dimension = 1
state_partitions = readdlm("partitions/test/state_partitions.txt", ',')
state_space = state_space_generation(state_partitions)

# Optimization flags
σ_noise = 0.1
initial_state_partition = 3

# Optimize
@time piecewise_barrier(system_dimension, state_space, state_partitions, initial_state_partition)
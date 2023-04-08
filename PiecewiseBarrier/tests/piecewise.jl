""" Piecewise Barrier Functions based on Bellman's Equation

    © Rayan Mazouz

"""

# Import packages
using Revise, BenchmarkTools

using PiecewiseBarrier
using LazySets
using DelimitedFiles

# State partitions
system_dimension = 1
state_partitions = readdlm("partitions/test/state_partitions.txt", ' ')
state_partitions = [Hyperrectangle(low=[low], high=[high]) for (low, high) in eachrow(state_partitions)]
state_space = state_space_generation(state_partitions)

# Optimization flags
initial_state_partition = 3

# # Optimize
@time piecewise_barrier(system_dimension, state_space, state_partitions, initial_state_partition)
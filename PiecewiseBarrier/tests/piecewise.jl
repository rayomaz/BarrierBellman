""" Piecewise Barrier Functions based on Bellman's Equation

    © Rayan Mazouz

"""

cd(@__DIR__)

# Import packages
using Revise
using PiecewiseBarrier

# State partitions
system_dimension = 1
state_partitions = readdlm("partitions/test/state_partitions.txt", ',')
state_space = PiecewiseBarrier.state_space_generation(state_partitions)

# Optimization flags
σ_noise = 0.1
initial_state_partition = Int(3)

# Optimize
@time barrier_bellman_sos(system_dimension, state_space, state_partitions, initial_state_partition)
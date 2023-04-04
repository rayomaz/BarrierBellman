""" Piecewise Barrier Functions based on Bellman's Equation

    © Rayan Mazouz

"""

cd(@__DIR__)

# Import packages
using Revise
using PiecewiseBarrier

# Parameters
system_dimension = 1
partitions_eps = 0.5
state_space = ([-1.0, 1.0])

# Optimize
@time barrier_bellman_sos(system_dimension, partitions_eps, state_space)
""" Piecewise Barrier Functions based on Bellman's Equation

    © Rayan Mazouz

"""

# Import packages
using Revise, BenchmarkTools
using PiecewiseBarrier
using MAT

# Bounds
bounds_file = "partitions/test/linearsystem_100.mat"
bounds = matopen(bounds_file)

# Optimization flags
initial_state_partition = 50

# Optimize
@time b, beta = constant_barrier(bounds, initial_state_partition)

# data = Dict("b" => b, "beta" => beta)

# # Save the dictionary in a .mat file
# matwrite("barrier_100.mat", data)

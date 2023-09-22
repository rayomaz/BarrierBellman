""" Piecewise Barrier Function: Harrier Model

    © Rayan Mazouz

"""

# Import packages
using Revise, BenchmarkTools
using StochasticBarrierFunctions, LazySets
using YAXArrays, NetCDF

# System
system_flag = "harrier"
number_hypercubes = 25920
σ = [0.1, 0.1, 0.01, 0.1, 0.1, 0.05]

filename = "models/probability_data_$(number_hypercubes)_sigma_$σ.nc"
dataset = open_dataset(joinpath(@__DIR__, filename))

probabilities = load_probabilities(dataset)

initial_region = Hyperrectangle([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, deg2rad(4), 0.5, 0.5, 0.1])
obstacle_region = EmptySet(6)

# Optimize: method 1 (revise beta values)
# @time B_ub, beta_ub = synthesize_barrier(UpperBoundAlgorithm(), probabilities, initial_region, obstacle_region)

# Optimize: method 2 (dual approach)
# @time B_dual, beta_dual = synthesize_barrier(DualAlgorithm(), probabilities, initial_region, obstacle_region)

# Optimize: method 3 (iterative approach)
# @time B_it, beta_it = synthesize_barrier(IterativeUpperBoundAlgorithm(), probabilities, initial_region, obstacle_region)

# Optimize: method 4 (project gradient descent approach)
@time B_pgd, beta_pgd = synthesize_barrier(GradientDescentAlgorithm(), probabilities, initial_region, obstacle_region)

println("Harrier model verified.")

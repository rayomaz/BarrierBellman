""" Piecewise Barrier Function: Linear Dynamic Model [Single Mode]

    © Rayan Mazouz

"""

# Import packages
using Revise, BenchmarkTools
using StochasticBarrierFunctions, LazySets, LinearAlgebra
using YAXArrays, NetCDF

# System
F = 0.50
dim = 2

A = F*I(dim)
b = zeros(dim)
σ = 0.1*ones(dim)

num_regions = 1444
filename = "models/single_mode/probability_data_$(num_regions)_f_$(F)_sigma_$σ.nc"
dataset = open_dataset(joinpath(@__DIR__, filename))

probabilities = load_probabilities(dataset)

# Initial range and obstacle space
initial_range, obstacle_range = 0.10, 0.015
initial_region = Hyperrectangle([-0.70, -0.10], initial_range*ones(dim))

obstacle1 = Hyperrectangle([-0.55, 0.30], obstacle_range*ones(dim))
obstacle2 = Hyperrectangle([-0.55, -0.15], obstacle_range*ones(dim))
obstacle_region = UnionSet(obstacle1, obstacle2)
# obstacle_region = EmptySet(dim)

obstacle1 = Hyperrectangle([-0.55, 0.30], obstacle_range*ones(dim))
obstacle2 = Hyperrectangle([-0.55, -0.15], obstacle_range*ones(dim))
obstacle_region = UnionSet(obstacle1, obstacle2)
# obstacle_region = EmptySet(dim)

# Set horizon
N = 10

# Optimize: method 1 (revise beta values)
@time B_ub, beta_ub = synthesize_barrier(UpperBoundAlgorithm(), probabilities, initial_region, obstacle_region; time_horizon=N)

# Optimize: method 2 (dual approach)
@time B_dual, beta_dual = synthesize_barrier(DualAlgorithm(), probabilities, initial_region, obstacle_region; time_horizon=N)

# Optimize: method 3 (iterative approach)
@time B_it, beta_it = synthesize_barrier(IterativeUpperBoundAlgorithm(), probabilities, initial_region, obstacle_region; time_horizon=N)

# Optimize: method 4 (project gradient descent approach)
@time B_pgd, beta_pgd = synthesize_barrier(GradientDescentAlgorithm(), probabilities, initial_region, obstacle_region; time_horizon=N)

println("Single mode grid verified.")

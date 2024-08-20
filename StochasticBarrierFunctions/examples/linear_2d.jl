""" Piecewise Barrier Function: Neural Network Dynamic Model [Pendulum]

    © Rayan Mazouz

"""

# Import packages
using Revise, BenchmarkTools
using StochasticBarrierFunctions, LazySets
using YAXArrays, NetCDF

# System
system_flag = "linear_2d"
σ = [0.01, 0.01]

num_regions = 400
filename = "models/linear_2d/probability_data_$(num_regions)_sigma_$σ.nc"
dataset = open_dataset(joinpath(@__DIR__, filename))

probabilities = load_probabilities(dataset)

initial_region = Ball2([0.0, 0.0], 1.5)
obstacle_region = Complement(Ball2([0.0, 0.0], 2.0))

# Optimize: method 1 (revise beta values)
@time res_ub = synthesize_barrier(UpperBoundAlgorithm(), probabilities, initial_region, obstacle_region)

# Optimize: method 2 (dual approach)
@time res_dual = synthesize_barrier(DualAlgorithm(), probabilities, initial_region, obstacle_region)

# Optimize: method 3 (iterative approach)
@time res_it = synthesize_barrier(IterativeUpperBoundAlgorithm(), probabilities, initial_region, obstacle_region)

# Optimize: method 4 (project gradient descent approach)
@time res_pgd = synthesize_barrier(GradientDescentAlgorithm(), probabilities, initial_region, obstacle_region)

println("Linear 2d model verified.")
using Revise, BenchmarkTools
using StochasticBarrierFunctions, LazySets
using YAXArrays, NetCDF

# System
system_flag = "pendulum"
number_hypercubes = 120
σ = [0.01, 0.01]

filename = "models/$system_flag/probability_data_$(number_hypercubes)_sigma_$σ.nc"
dataset = open_dataset(joinpath(@__DIR__, filename))

probabilities = load_probabilities(dataset)

initial_region = Hyperrectangle([0.0, 0.0], [0.01, 0.01])
obstacle_region = EmptySet(2)

# Optimize: method 1 (revise beta values)
@time res_ub = synthesize_barrier(UpperBoundAlgorithm(), probabilities, initial_region, obstacle_region)

# Optimize: method 2 (dual approach)
@time res_dual = synthesize_barrier(DualAlgorithm(), probabilities, initial_region, obstacle_region)

# Optimize: method 3 (iterative approach)
@time res_it = synthesize_barrier(IterativeUpperBoundAlgorithm(), probabilities, initial_region, obstacle_region)

# Optimize: method 4 (project gradient descent approach)
@time res_pgd = synthesize_barrier(GradientDescentAlgorithm(), probabilities, initial_region, obstacle_region)

println("Pendulum model verified.")

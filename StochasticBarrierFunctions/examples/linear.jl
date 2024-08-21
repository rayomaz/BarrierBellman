using Revise, BenchmarkTools
using StochasticBarrierFunctions, LazySets
using YAXArrays, NetCDF

# System
system_flag = "linear"
f = 1.05
σ = [0.01]

num_regions = 5
filename = "models/linear/probability_data_$(num_regions)_f_$(f)_sigma_$σ.nc"
dataset = open_dataset(joinpath(@__DIR__, filename))

probabilities = load_probabilities(dataset)

initial_region = Hyperrectangle(low=[-0.05], high=[0.05])
obstacle_region = EmptySet(1)

# Optimize: method 1 (revise beta values)
@time res_ub = synthesize_barrier(UpperBoundAlgorithm(), probabilities, initial_region, obstacle_region)

# Optimize: method 2 (dual approach)
@time res_dual = synthesize_barrier(DualAlgorithm(), probabilities, initial_region, obstacle_region)

# Optimize: method 3 (iterative approach)
@time res_it = synthesize_barrier(IterativeUpperBoundAlgorithm(), probabilities, initial_region, obstacle_region)

# Optimize: method 4 (project gradient descent approach)
@time res_pgd = synthesize_barrier(GradientDescentAlgorithm(), probabilities, initial_region, obstacle_region)

println("Linear model verified.")
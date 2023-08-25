""" Piecewise Barrier Function: Neural Network Dynamic Model [Pendulum]

    © Rayan Mazouz

"""

# Import packages
using Revise, BenchmarkTools
using PiecewiseConstant, LazySets
using YAXArrays, NetCDF

# System
system_flag = "cartpole"
number_hypercubes = 960
σ = [0.1, 0.1, 0.1, 0.1]

filename = "models/$system_flag/probability_data_$(number_hypercubes)_sigma_$σ.nc"
dataset = open_dataset(joinpath(@__DIR__, filename))

probabilities = load_probabilities(dataset)

initial_region = Hyperrectangle([0.0, 0.0, 0.0, 0.0], [0, 0, 0.01, 0])
obstacle_region = EmptySet(4)

# Optimize: method 1 (revise beta values)
@time B, beta = constant_barrier(probabilities, initial_region, obstacle_region)
@time beta_updated, p_distribution = post_compute_beta(B, probabilities)
# @btime beta_updated = accelerated_post_compute_beta(B, regions)

# Optimize: method 2 (dual approach)
@time B_dual, beta_dual = dual_constant_barrier(probabilities, initial_region, obstacle_region)
# @time beta_dual_updated, p_distribution = post_compute_beta(B_dual, probabilities)

# Optimize: method 3 (iterative approach)
@time B_iterative, beta_iterative = iterative_barrier(probabilities, initial_region, obstacle_region)

println("Cartpole model verified.")
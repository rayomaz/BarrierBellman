""" Piecewise Barrier Function: Neural Network Dynamic Model [Pendulum]

    © Rayan Mazouz

"""

# Import packages
using Revise, BenchmarkTools
using PiecewiseConstant, LazySets
using MAT

# System
system_flag = "pendulum"
number_hypercubes = 120
σ = 0.10
probabilities = "models/$system_flag/probability_data_$(number_hypercubes)_sigma_$σ.mat"
probabilities = matopen(probabilities)
regions = "models/$system_flag/partition_data_$number_hypercubes.mat"
regions = matopen(regions)
regions = read_regions(regions, probabilities)

initial_region = Hyperrectangle([0.0, 0.0], [0.01, 0.01])
obstacle_region = EmptySet(2)

# Optimize: method 1 (revise beta values)
@time B, beta = constant_barrier(regions, initial_region, obstacle_region)
@time beta_updated, p_distribution = post_compute_beta(B, regions)
# @btime beta_updated = accelerated_post_compute_beta(B, regions)

# Optimize: method 2 (dual approach)
@time B_dual, beta_dual = dual_constant_barrier(regions, initial_region, obstacle_region)

println("\n Pendulum model verified.")
""" Piecewise Barrier Function: Neural Network Dynamic Model [Pendulum]

    © Rayan Mazouz

"""

# Import packages
using Revise, BenchmarkTools
using PiecewiseConstant, LazySets
using MAT

# System
system_flag = "linear"
number_hypercubes = 5
σ = 0.01
probabilities = "models/$system_flag/probability_data_$(number_hypercubes)_sigma_$σ.mat"
probabilities = matopen(probabilities)
regions = [Interval(-0.25, -0.15), Interval(-0.15, -0.05), Interval(-0.05, 0.05), Interval(0.05, 0.15), Interval(0.15, 0.25)]
regions = read_regions(regions, probabilities)

initial_region = Interval(-0.05, 0.05)
obstacle_region = EmptySet(1)

# Optimize: method 1 (revise beta values)
@time B, beta = constant_barrier(regions, initial_region, obstacle_region)
@time beta_updated, p_distribution = post_compute_beta(B, regions)
# @btime beta_updated = accelerated_post_compute_beta(B, regions)

# Optimize: method 2 (dual approach)
@time b_dual, beta_dual = dual_constant_barrier(regions, initial_region, obstacle_region)

println("\n Linear model verified.")
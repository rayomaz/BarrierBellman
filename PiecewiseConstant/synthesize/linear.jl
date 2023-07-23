""" Piecewise Barrier Function: Neural Network Dynamic Model [Pendulum]

    © Rayan Mazouz

"""

# Import packages
using Revise, BenchmarkTools
using PiecewiseConstant, LazySets
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
@time B, beta = constant_barrier(probabilities, initial_region, obstacle_region)
@time beta_updated, p_distribution = post_compute_beta(B, probabilities)
# println(beta_updated)
# @time beta_updated = accelerated_post_compute_beta(B, probabilities)

# Optimize: method 2 (dual approach)
@time B_dual, beta_dual = dual_constant_barrier(probabilities, initial_region, obstacle_region)
@time beta_dual_updated, p_distribution = post_compute_beta(B_dual, probabilities)
# println(beta_dual_updated)

println("Linear model verified.")
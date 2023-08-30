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
@time B, beta = constant_barrier(probabilities, initial_region, obstacle_region)
@time beta_updated, p_distribution = post_compute_beta(B, probabilities)
# println(beta_updated)
# @time beta_updated = accelerated_post_compute_beta(B, probabilities)

# Optimize: method 2 (dual approach)
@time B_dual, beta_dual = dual_constant_barrier(probabilities, initial_region, obstacle_region)
@time beta_dual_updated, p_distribution = post_compute_beta(B_dual, probabilities)
# println(beta_dual_updated)

println("Linear 2d model verified.")
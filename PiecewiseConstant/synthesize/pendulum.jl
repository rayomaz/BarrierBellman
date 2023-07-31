""" Piecewise Barrier Function: Neural Network Dynamic Model [Pendulum]

    © Rayan Mazouz

"""

# Import packages
using Revise, BenchmarkTools
using PiecewiseConstant, LazySets
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
@time B, beta = constant_barrier(probabilities, initial_region, obstacle_region)
# @time beta_updated, p_distribution = post_compute_beta(B, probabilities)
# # @btime beta_updated = accelerated_post_compute_beta(B, regions)

# # Optimize: method 2 (dual approach)
# @time B_dual, beta_dual = dual_constant_barrier(probabilities, initial_region, obstacle_region)
# @time beta_dual_updated, p_distribution = post_compute_beta(B_dual, probabilities)

# Plotenvironment/barrier
plot_environment(probabilities, initial_region, obstacle_region, B)

println("Pendulum model verified.")
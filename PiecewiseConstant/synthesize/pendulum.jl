""" Piecewise Barrier Function: Neural Network Dynamic Model [Pendulum]

    © Rayan Mazouz

"""

# Import packages
using Revise, BenchmarkTools
using PiecewiseConstant
using MAT

# System
system_flag = "pendulum"
number_hypercubes = 120
σ = 0.01
probabilities = "models/" * system_flag * "/probability_data_"  * string(number_hypercubes) * "_sigma_" *string(σ) * ".mat"
probabilities = matopen(probabilities)

# Optimize: method 1 (revise beta values)
@time b, beta = constant_barrier(probabilities)
@time beta_updated, p_distribution = post_compute_beta(b, probabilities)

@time beta_updated_centralized, p_distribution_central = post_compute_beta_centralized(b, probabilities)

# Optimize: method 2 (dual approach)
# @time b_dual, beta_dual = dual_constant_barrier(probabilities)

# Sanity checks
# for jj = 1:120
#     sum_probabilities(jj, probabilities)
#     # sum_barrier_probabilities(jj, b, beta, probabilities)
# end

println("\n Pendulum model verified.")
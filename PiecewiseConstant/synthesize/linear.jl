""" Piecewise Barrier Functions based on Bellman's Equation

    © Rayan Mazouz

"""

# Import packages
using Revise, BenchmarkTools
using PiecewiseConstant

using LazySets
using MultivariatePolynomials, DynamicPolynomials

using DelimitedFiles
using MAT

# System
@polyvar x
fx = 0.95 * x
σ = 0.1

system = AdditiveGaussianPolynomialSystem{Float64, 1}(x, fx, σ)

# State partitions
state_partitions = readdlm("models/linear/state_partitions.txt", ' ')
state_partitions = [Hyperrectangle(low=[low], high=[high]) for (low, high) in eachrow(state_partitions)]

# Optimization flags
initial_state_partition = Int(round(length(state_partitions)/2))

# Optimize
# @time b, beta = constant_barrier(system, state_partitions, initial_state_partition)

# Optimize: method 1 (revise beta values)
# @time b, beta = constant_barrier(probabilities)
# @time beta_updated = post_compute_beta(b, probabilities)

# # Optimize: method 2 (dual approach)
# @time b_dual, beta_dual = dual_constant_barrier(probabilities)

# data = Dict("b" => b, "beta" => beta)

# # Save the dictionary in a .mat file
# matwrite("barrier_1000.mat", data)
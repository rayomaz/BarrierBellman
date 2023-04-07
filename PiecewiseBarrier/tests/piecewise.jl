""" Piecewise Barrier Functions based on Bellman's Equation

    © Rayan Mazouz

"""

# Import packages
using Revise, BenchmarkTools

using PiecewiseBarrier
using LazySets
using MultivariatePolynomials, DynamicPolynomials

using DelimitedFiles

# System
@polyvar x
fx = 0.5 * x^2
σ = 0.01

system = AdditiveGaussianPolynomialSystem{Float64, 1}(x, fx, σ)

# State partitions
state_partitions = readdlm("partitions/test/state_partitions.txt", ' ')
state_partitions = [Interval(low, high) for (low, high) in eachrow(state_partitions)]
state_space = state_space_generation(state_partitions)

# Optimization flags
σ_noise = 0.1
initial_state_partition = 3

# # Optimize
@time piecewise_barrier(system, state_space, state_partitions, initial_state_partition)
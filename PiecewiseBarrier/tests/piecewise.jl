""" Piecewise Barrier Functions based on Bellman's Equation

    © Rayan Mazouz

"""

# Import packages
using Revise, BenchmarkTools

using PiecewiseBarrier
using LazySets
using MultivariatePolynomials, DynamicPolynomials

using DelimitedFiles
using MAT

# System
@polyvar x
fx = 0.95 * x
σ = false           # comment: noise not needed in piecewise, embedded in bounds on P(x) and E(x)

system = AdditiveGaussianPolynomialSystem{Float64, 1}(x, fx, σ)

# State partitions
state_partitions = readdlm("partitions/test/state_partitions.txt", ' ')
state_partitions = [Hyperrectangle(low=[low], high=[high]) for (low, high) in eachrow(state_partitions)]

# Bounds
bounds_file = "partitions/test/linearsystem_5.mat"
bounds = matopen(bounds_file)

# Optimization flags
initial_state_partition = 3

# Optimize
@time piecewise_barrier(system, bounds, state_partitions, initial_state_partition)
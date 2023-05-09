"""
    - Generation of Stochastic Barrier Functions

    © Rayan Mazouz

"""

# Stochastic Barrier Verification
using Revise, BenchmarkTools

using PiecewiseBarrier
using LazySets
using MultivariatePolynomials, DynamicPolynomials

using DelimitedFiles

# System
@polyvar x
fx = 0.95 * x
σ = 0.1

system = AdditiveGaussianPolynomialSystem{Float64, 1}(x, fx, σ)

# State partitions
state_partitions = readdlm("partitions/test/state_partitions.txt", ' ')
state_partitions = [Hyperrectangle(low=[low], high=[high]) for (low, high) in eachrow(state_partitions)]
state_space = state_space_generation(state_partitions)

# Optimization flags
initial_state_partition = Int(round(length(state_partitions)/2))

# Optimization
eta, beta = @time sos_barrier(system, state_space, state_partitions, initial_state_partition)
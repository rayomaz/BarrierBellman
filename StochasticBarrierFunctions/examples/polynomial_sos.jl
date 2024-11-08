using Revise, BenchmarkTools
using StochasticBarrierFunctions, LazySets, LinearAlgebra
using DynamicPolynomials
using Mosek, MosekTools
using YAXArrays, NetCDF


# System
dim = 2
@polyvar x[1:dim]

f = [x[1]^2 - 3*x[2]; x[2]^2]
σ = [0.1, 0.1]

state_space = Hyperrectangle(low  = [-1.0, -2.0], high = [1.0, 2.0])

system = AdditiveGaussianPolySystem(f, σ, state_space)

# Initial range and obstacle space
initial_region = Hyperrectangle([0.0, 0.0], 0.01*ones(dim))
obstacle_region = EmptySet(dim)

# Set horizon
N = 10

# Optimize: baseline 1 (sos)
@time res_sos = synthesize_barrier(SumOfSquaresAlgorithm(barrier_degree = 4), system, initial_region, obstacle_region; time_horizon = N)

println("Polynomial model verified.")
using Revise, BenchmarkTools
using StochasticBarrierFunctions, LazySets, LinearAlgebra
using DynamicPolynomials
using Mosek, MosekTools
using YAXArrays, NetCDF


# System
dim = 2
@polyvar x[1:dim]

τ = 0.1
f = [x[1] + τ*x[2]; x[2] + τ*(-x[1] + (1 - x[1])^2)]
σ = [0.02, 0.02]

state_space = Hyperrectangle(low  = [-6.0, -6.0], high = [6.0, 6.0])

system = AdditiveGaussianPolySystem(f, σ, state_space)

# Initial range and obstacle space
initial_region = Hyperrectangle(low = [-5.0, -5.0], high = [5.0, 5.0])
obstacle_region = EmptySet(dim)

# Set horizon
N = 5

# Optimize: baseline 1 (sos)
@time res_sos = synthesize_barrier(SumOfSquaresAlgorithm(barrier_degree = 6), system, initial_region, obstacle_region; time_horizon = N)

println("Van der Pol Oscillator model verified.")
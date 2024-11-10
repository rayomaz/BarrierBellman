using Revise, BenchmarkTools
using StochasticBarrierFunctions, LazySets, LinearAlgebra
using DynamicPolynomials
using Mosek, MosekTools
using YAXArrays, NetCDF


# System
dim = 2
@polyvar x[1:dim]

τ = 0.1
α₁ = 1
α₂ = 1
q₀ = -3
q₁ = 4.5

f = [(1 - τ*α₁)*x[1] + τ*α₁*q₁;  τ*α₁*x[1] + (1 - τ*α₂)*x[2] + τ*α₂*q₀]
σ = [1e-4, 1e-4]

state_space = Hyperrectangle(low  = [1.0, 1.0], high = [10.0, 10.0])

system = AdditiveGaussianPolySystem(f, σ, state_space)

# Initial range and obstacle space
initial_region = Hyperrectangle(low = [4.9, 4.9], high = [5.1, 5.1])
obstacle_region = EmptySet(dim)

# Set horizon
N = 10

# Optimize: baseline 1 (sos)
@time res_sos = synthesize_barrier(SumOfSquaresAlgorithm(barrier_degree = 4), system, initial_region, obstacle_region; time_horizon = N)

println("Two Tank problems model verified.")
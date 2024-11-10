using Revise, BenchmarkTools
using StochasticBarrierFunctions, LazySets, LinearAlgebra
using DynamicPolynomials
using Mosek, MosekTools
using YAXArrays, NetCDF


# System
dim = 3
@polyvar x[1:dim]

# Room temperature problem
αₑ = 8e-3
α  = 6.2e-3
τ  = 5
Te = 15
R  = 0.10

f = [1 - τ*(α + αₑ)*x[1] + τ*α*x[2] + τ*αₑ*Te;
     1 - τ*(2*α + αₑ)*x[1] + τ*α*(x[1] + x[3]) + τ*αₑ*Te;
     1 - τ*(α + αₑ)*x[3] + τ*α*x[2]  + τ*αₑ*Te
    ]
σ = R*ones(dim)

state_space = Hyperrectangle(low = 17.0*ones(dim), high = 29.0*ones(dim))

system = AdditiveGaussianPolySystem(f, σ, state_space)

# Initial range and obstacle space
initial_region = Hyperrectangle(low = 17.0*ones(dim), high = 18*ones(dim))
obstacle_region = EmptySet(dim)

# Set horizon
N = 3

# # Optimize: baseline 1 (sos)
@time res_sos = synthesize_barrier(SumOfSquaresAlgorithm(barrier_degree = 6), system, initial_region, obstacle_region; time_horizon = N)

println("Room temperature model verified.")
using Revise, BenchmarkTools
using StochasticBarrierFunctions, LazySets, LinearAlgebra
using DynamicPolynomials
using Mosek, MosekTools
using YAXArrays, NetCDF


# System
dim = 1
@polyvar x[1:dim]

# Room temperature problem
Th = 45
Te = -15
β  = 0.6
θ  = 0.145
R  = 0.10
v  = -0.0120155*x[1] + 0.8

f = [(1 - β - θ*v)*x[1] + θ*Th*v + β*Te]
σ = [R*1]

state_space = Hyperrectangle(low = [1.0], high = [50.0])

system = AdditiveGaussianPolySystem(f, σ, state_space)

# Initial range and obstacle space
initial_region = Hyperrectangle(low = [19.5], high = [20.0])
obstacle1 = Hyperrectangle(low = [1.0], high = [17.0])
obstacle2 = Hyperrectangle(low = [23.0], high = [50.0])
obstacle_region = UnionSet(obstacle1, obstacle2)

# Set horizon
N = 10

# # Optimize: baseline 1 (sos)
@time res_sos = synthesize_barrier(SumOfSquaresAlgorithm(barrier_degree = 8), system, initial_region, obstacle_region; time_horizon = N)

println("Room temperature model verified.")
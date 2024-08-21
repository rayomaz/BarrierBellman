using Revise, BenchmarkTools
using StochasticBarrierFunctions, LazySets, LinearAlgebra
using Mosek, MosekTools
using YAXArrays, NetCDF

# System
F = 0.50
dim = 2

A = F*I(dim)
b = zeros(dim)
σ = 0.1*ones(dim)

state_space = Hyperrectangle(low=-1.0*ones(dim), high=0.5*ones(dim))

system = AdditiveGaussianLinearSystem(A, b, σ, state_space)

# Initial range and obstacle space
initial_range, obstacle_range = 0.10, 0.015
initial_region = Hyperrectangle([-0.70, -0.10], initial_range*ones(dim))

obstacle1 = Hyperrectangle([-0.55, 0.30], obstacle_range*ones(dim))
obstacle2 = Hyperrectangle([-0.55, -0.15], obstacle_range*ones(dim))
obstacle_region = UnionSet(obstacle1, obstacle2)
# obstacle_region = EmptySet(dim)

# Set horizon
N = 10

# Optimize: baseline 1 (sos)
@time res_sos = synthesize_barrier(SumOfSquaresAlgorithm(barrier_degree = 12, sdp_solver = Mosek.Optimizer), system, initial_region, obstacle_region; time_horizon = N)

println("Single mode model verified.")

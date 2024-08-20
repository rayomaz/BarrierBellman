using Revise, BenchmarkTools
using StochasticBarrierFunctions, LazySets
using Mosek, MosekTools
using YAXArrays, NetCDF

# System
system_flag = "linear"
f = 1.05
A = [f][:, :]
b = [0.0]
σ = [0.01]

state_space = Hyperrectangle(low=[-0.25], high=[0.25])

system = AdditiveGaussianLinearSystem(A, b, σ, state_space)

initial_region = Hyperrectangle(low=[-0.05], high=[0.05])
obstacle_region = EmptySet(2)

# Optimize: baseline 1 (sos)
@time res_sos = synthesize_barrier(SumOfSquaresAlgorithm(sdp_solver=Mosek.Optimizer), system, initial_region, obstacle_region)

println("Linear model verified.")

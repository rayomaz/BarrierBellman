""" Piecewise Barrier Function: Neural Network Dynamic Model [Pendulum]

    © Rayan Mazouz

"""

# Import packages
using Revise, BenchmarkTools
using StochasticBarrierFunctions, LazySets
using Mosek, MosekTools
using YAXArrays, NetCDF

# System
system_flag = "linear"
f = 1.00
A = [f][:, :]
b = [0.0]
σ = [0.1]

state_space = Hyperrectangle(low=[-1.0], high=[1.0])

system = AdditiveGaussianLinearSystem(A, b, σ, state_space)

initial_region = Hyperrectangle(low=[-0.25], high=[0.25])
obstacle_region = EmptySet(2)

# Optimize: baseline 1 (sos)
@time res_sos = synthesize_barrier(SumOfSquaresAlgorithm(sdp_solver=Mosek.Optimizer), system, initial_region, obstacle_region)

println("Linear model verified.")

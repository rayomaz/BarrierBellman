""" Piecewise Barrier Function: Neural Network Dynamic Model [Pendulum]

    © Rayan Mazouz

"""

# Import packages
using Revise, BenchmarkTools
using StochasticBarrierFunctions, LazySets
using YAXArrays, NetCDF, MAT

# System
system_flag = "pendulum"
number_hypercubes = 120
number_layers = 1
σ = [0.01, 0.01]

filename = "../data/nndm/$system_flag/$(number_layers)_layer/dynamics_$number_hypercubes.nc"
dataset = open_dataset(joinpath(@__DIR__, filename))
Xs = load_dynamics(dataset)

# filename = "models/$system_flag/partition_data_$number_hypercubes.mat"
# file = matopen(filename)
# Xs = load_dynamics(file)
# close(file)

system = AdditiveGaussianUncertainPWASystem(Xs, σ)

initial_region = Hyperrectangle([0.0, 0.0], [0.01, 0.01])
obstacle_region = EmptySet(2)

# Optimize: baseline 1 (sos)
@time res_sos = synthesize_barrier(SumOfSquaresAlgorithm(), system, initial_region, obstacle_region)

println("Pendulum model verified.")

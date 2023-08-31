# Import packages
using Revise, BenchmarkTools
using StochasticBarrierFunctions
using YAXArrays, NetCDF

# System
system_flag = "harrier"
number_hypercubes = 36000
σ = [0.1, 0.1, 0.01, 0.1, 0.1, 0.01]

filename = "../data/$system_flag/dynamics_$number_hypercubes.nc"
dataset = open_dataset(joinpath(@__DIR__, filename))

Xs = load_dynamics(dataset)
system = AdditiveGaussianUncertainPWASystem(Xs, σ)

# Extract probability data
@time probability_bounds = transition_probabilities(system)

# Save to a .nc file
filename = "models/$system_flag/probability_data_$(number_hypercubes)_sigma_$σ.nc"
savedataset(probability_bounds; path=joinpath(@__DIR__, filename), driver=:netcdf, overwrite=true, compress=1)
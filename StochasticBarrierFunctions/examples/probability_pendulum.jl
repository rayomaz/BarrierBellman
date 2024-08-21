# Start julia with multiple threads
# julia --threads 16

using Revise, BenchmarkTools
using StochasticBarrierFunctions
using YAXArrays, NetCDF, MAT

# System
system_flag = "pendulum"
number_hypercubes = 240
number_layers = 1
σ = [0.01, 0.01]

filename = "models/$system_flag/partition_data_$number_hypercubes.mat"
file = matopen(joinpath(@__DIR__, filename))

Xs = load_dynamics(file)
close(file)

system = AdditiveGaussianUncertainPWASystem(Xs, σ)
plot_posterior(system; figname_prefix="mat")

filename = "../data/nndm/$system_flag/$(number_layers)_layer/dynamics_$number_hypercubes.nc"
dataset = open_dataset(joinpath(@__DIR__, filename))

Xs = load_dynamics(dataset)

system = AdditiveGaussianUncertainPWASystem(Xs, σ)
plot_posterior(system; figname_prefix="bp")

# Extract probability data
@time probability_bounds = transition_probabilities(system)

# Save to a .nc file
filename = "models/pendulum/probability_data_$(number_hypercubes)_sigma_$σ.nc"
savedataset(probability_bounds; path=joinpath(@__DIR__, filename), driver=:netcdf, overwrite=true, compress=1)

""" Piecewise Barrier Functions: compute transition probabilities [Pendulum]

    © Rayan Mazouz

"""

# Start julia with multiple threads
# julia --threads 16

# Import packages
using Revise, BenchmarkTools
using StochasticBarrierFunctions
using YAXArrays, NetCDF, MAT

# System
system_flag = "cartpole"
number_hypercubes = 960
number_layers = 1
σ = [0.1, 0.1, 0.1, 0.1]

filename = "models/$system_flag/partition_data_$number_hypercubes.mat"
file = matopen(joinpath(@__DIR__, filename))

Xs = load_dynamics(file)
close(file)

system = AdditiveGaussianUncertainPWASystem(Xs, σ)

@time probability_bounds = transition_probabilities(system)

# Save to a .nc file
filename = "models/cartpole/probability_data_$(number_hypercubes)_sigma_$σ.nc"
savedataset(probability_bounds; path=joinpath(@__DIR__, filename), driver=:netcdf, overwrite=true, compress=1)

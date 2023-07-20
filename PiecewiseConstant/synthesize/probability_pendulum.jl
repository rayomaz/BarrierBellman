""" Piecewise Barrier Functions: compute transition probabilities [Pendulum]

    © Rayan Mazouz

"""

# Start julia with multiple threads
# julia --threads 16

# Import packages
using Revise, BenchmarkTools
using PiecewiseConstant
using YAXArrays, NetCDF

# System
system_flag = "pendulum"
number_hypercubes = 480
filename = "models/$system_flag/partition_data_$number_hypercubes.mat"
file = matopen(joinpath(@__DIR__, filename))

Xs = load_dynamics(file)
close(file)
σ = [0.01, 0.01]

system = AdditiveGaussianUncertainPWASystem(Xs, σ)

# Extract probability data
@time probability_bounds = transition_probabilities(system)

# Save to a .nc file
filename = "models/pendulum/probability_data_$(number_hypercubes)_sigma_$σ.nc"
savedataset(probability_bounds; path=joinpath(@__DIR__, filename), driver=:netcdf, overwrite=true)

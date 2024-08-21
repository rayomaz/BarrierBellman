using Revise, BenchmarkTools
using StochasticBarrierFunctions
using LazySets
using YAXArrays, NetCDF

# system
fertitily_rate = 0.4
survial_juvenile = 0.3
survial_adult = 0.8
A = [0.0 fertitily_rate; survial_juvenile survial_adult]
b = [0.0, 0.0]
σ = [0.01, 0.01]

system = AdditiveGaussianLinearSystem(A, b, σ)

# State partitions
num_regions_x = 10
x = range(-2.0, 2.0, length=num_regions_x + 1)
num_regions_y = 10
y = range(-2.0, 2.0, length=num_regions_y + 1)

state_partitions = [Hyperrectangle(low=[low_x, low_y], high=[high_x, high_y]) for (low_x, high_x) in zip(x[1:end-1], x[2:end]) for (low_y, high_y) in zip(y[1:end-1], y[2:end])]

# Extract probability data
@time probability_bounds = transition_probabilities(system, state_partitions)

# Save to a .nc file
filename = "models/linear_2d/probability_data_$(length(state_partitions))_sigma_$σ.nc"
savedataset(probability_bounds; path=joinpath(@__DIR__, filename), driver=:netcdf, overwrite=true)

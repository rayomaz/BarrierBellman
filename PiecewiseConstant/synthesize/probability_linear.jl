""" Piecewise Barrier Functions based on Bellman's Equation

    © Rayan Mazouz

"""

# Import packages
using Revise, BenchmarkTools
using PiecewiseConstant

using LazySets

using YAXArrays, NetCDF
using DelimitedFiles
using MAT

# System
f = 1.05
A = [f][:, :]
b = [0.0]
σ = [0.01]

system = AdditiveGaussianLinearSystem(A, b, σ)

# State partitions
num_regions = 5
a = range(-0.25, 0.25, length=num_regions + 1)
state_partitions = [Hyperrectangle(low=[low], high=[high]) for (low, high) in zip(a[1:end-1], a[2:end])]

# Extract probability data
@time probability_bounds = transition_probabilities(system, state_partitions)

# Save to a .nc file
filename = "models/linear/probability_data_$(length(state_partitions))_f_$(f)_sigma_$σ.nc"
savedataset(probability_bounds; path=joinpath(@__DIR__, filename), driver=:netcdf, overwrite=true)

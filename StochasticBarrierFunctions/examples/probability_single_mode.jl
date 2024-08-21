using Revise, BenchmarkTools
using StochasticBarrierFunctions, LazySets, LinearAlgebra
using YAXArrays, NetCDF

# System
F = 1.0
dim = 2

A = F*I(dim)
b = zeros(dim)
σ = 0.1*ones(dim)

system = AdditiveGaussianLinearSystem(A, b, σ)

# State partitions
Δ = 0.20
lower_bound = -1.0
upper_bound = 0.5
Q = Int(round((upper_bound - lower_bound) / Δ))
x = range(lower_bound, upper_bound, length=Q+1)

if dim == 1
    state_partitions = [Hyperrectangle(low=[low_x], high=[high_x]) 
                        for (low_x, high_x) in zip(x[1:end-1], x[2:end])]
elseif dim == 2
    state_partitions = [Hyperrectangle(low=[low_x, low_y], high=[high_x, high_y]) 
                        for (low_x, high_x) in zip(x[1:end-1], x[2:end])
                            for (low_y, high_y) in zip(x[1:end-1], x[2:end])]
end

# Extract probability data
@time probability_bounds = transition_probabilities(system, state_partitions)

# Save to a .nc file
filename = "models/single_mode/probability_data_$(length(state_partitions))_f_$(F)_sigma_$σ.nc"
savedataset(probability_bounds; path=joinpath(@__DIR__, filename), driver=:netcdf, overwrite=true)

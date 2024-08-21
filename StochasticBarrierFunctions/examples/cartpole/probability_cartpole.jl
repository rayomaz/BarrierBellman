using Revise, BenchmarkTools
using StochasticBarrierFunctions
using YAXArrays, NetCDF

# System
system_flag = "cartpole"
number_hypercubes = 40000
σ = [0.01, 0.01, 0.005, 0.005]

filename = "data/dynamics_$number_hypercubes.nc"
dataset = open_dataset(joinpath(@__DIR__, filename))

Xs = load_dynamics(dataset)
system = AdditiveGaussianUncertainPWASystem(Xs, σ)

# Extract probability data
@time probability_bounds = transition_probabilities(system; alg=TransitionProbabilityAlgorithm(upper_bound_method=FrankWolfeSolver()))

# Save to a .nc file
filename = "models/probability_data_$(number_hypercubes)_sigma_$σ.nc"
savedataset(probability_bounds; path=joinpath(@__DIR__, filename), driver=:netcdf, overwrite=true, compress=1)
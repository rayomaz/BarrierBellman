""" Piecewise Barrier Functions based on Bellman's Equation

    © Rayan Mazouz

"""

# Import packages
using Revise, BenchmarkTools
using PiecewiseConstant

using LazySets

using DelimitedFiles
using MAT

# System
A = [1.05][:, :]
b = [0.0]
σ = [0.01]

system = AdditiveGaussianLinearSystem(A, b, σ)

# State partitions
filename = "models/linear/state_partitions.txt"
state_partitions = readdlm(joinpath(@__DIR__, filename), ' ')
state_partitions = [Hyperrectangle(low=[low], high=[high]) for (low, high) in eachrow(state_partitions)]

# Extract probability data
@time probability_bounds = transition_probabilities(system, state_partitions)

(matrix_prob_lower, 
 matrix_prob_upper,
 matrix_prob_unsafe_lower,
 matrix_prob_unsafe_upper) = probability_bounds

# Save data
data = Dict("matrix_prob_lower" => matrix_prob_lower,
            "matrix_prob_upper" => matrix_prob_upper,
            "matrix_prob_unsafe_lower" => matrix_prob_unsafe_lower,
            "matrix_prob_unsafe_upper" => matrix_prob_unsafe_upper)

# Save the dictionary in a .mat file
filename = "models/linear/probability_data_$(length(state_partitions))_sigma_$σ.mat"
matwrite(joinpath(@__DIR__, filename), data)
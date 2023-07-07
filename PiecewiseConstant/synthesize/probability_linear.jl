""" Piecewise Barrier Functions based on Bellman's Equation

    © Rayan Mazouz

"""

# Import packages
using Revise, BenchmarkTools
using PiecewiseConstant

using LazySets
using MultivariatePolynomials, DynamicPolynomials

using DelimitedFiles
using MAT

# System
@polyvar x
fx = 0.95 * x
σ = 0.05

system = AdditiveGaussianPolynomialSystem{Float64, 1}(x, fx, σ)

# State partitions
state_partitions = readdlm("models/linear/state_partitions.txt", ' ')
state_partitions = [Hyperrectangle(low=[low], high=[high]) for (low, high) in eachrow(state_partitions)]

# Extract probability data
@time probability_bounds = linear_transition_probabilities(system, state_partitions)

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
save_dir = "models/linear/probability_data_" *string(length(state_partitions)) * "_sigma_" * string(σ) * ".mat"
matwrite(save_dir, data)
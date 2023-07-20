""" Piecewise Barrier Functions: compute transition probabilities [Pendulum]

    © Rayan Mazouz

"""

# Start julia with multiple threads
# julia --threads 16

# Import packages
using Revise, BenchmarkTools
using PiecewiseConstant
using MAT

# System
system_flag = "pendulum"
number_hypercubes = 480
filename = "models/$system_flag/partition_data_$number_hypercubes.mat"
file = matopen(joinpath(@__DIR__, filename))

Xs = load_dynamics(file)
close(file)
σ = [0.01, 0.01]

system = AdditiveGaussianUncertainPWASystem(Xs, σ)

# Optimize
@time probability_bounds = transition_probabilities(system)

# Extract probability data
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
filename = "models/pendulum/probability_data_$(number_hypercubes)_sigma_$σ.mat"
matwrite(joinpath(@__DIR__, filename), data)

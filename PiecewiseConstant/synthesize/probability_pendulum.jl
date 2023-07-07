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
number_hypercubes = 120
filename = "/models/" * system_flag * "/partition_data_"  * string(number_hypercubes) * ".mat"
file = matopen(pwd()*filename)

# Optimize
σ = 0.01
@time probability_bounds = neural_transition_probabilities(file, number_hypercubes, σ)

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
save_dir = "models/pendulum/probability_data_" *string(number_hypercubes) * "_sigma_" * string(σ) * ".mat"
matwrite(save_dir, data)
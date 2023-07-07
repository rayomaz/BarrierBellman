""" Piecewise Barrier Function: Neural Network Dynamic Model [Pendulum]

    © Rayan Mazouz

"""

# Import packages
using Revise, BenchmarkTools
using PiecewiseConstant
using MAT

# System
system_flag = "linear"
number_hypercubes = 5
σ = 0.01
probabilities = "models/" * system_flag * "/probability_data_"  * string(number_hypercubes) * "_sigma_" *string(σ) * ".mat"
probabilities = matopen(probabilities)

# Optimize: method 1 (revise beta values)
@time b, beta = constant_barrier(probabilities)
@time beta_updated = post_compute_beta(b, probabilities)

# Optimize: method 2 (dual approach)
@time b_dual, beta_dual = dual_constant_barrier(probabilities)

# Sanity checks
for jj = 1:5
    sum_probabilities(jj, probabilities)
    # sum_barrier_probabilities(jj, b, beta, probabilities)
end

println("\n Linear model verified.")
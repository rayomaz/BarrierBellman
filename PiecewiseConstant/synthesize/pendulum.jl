""" Piecewise Barrier Function: Neural Network Dynamic Model [Pendulum]

    © Rayan Mazouz

"""

# Import packages
using Revise, BenchmarkTools
using PiecewiseConstant
using MAT

# System
system_flag = "pendulum"
number_hypercubes = 120
probabilities = "/models/" * system_flag * "/probability_data_"  * string(number_hypercubes) * ".mat"
probabilities = matopen(pwd()*probabilities)

# Optimize: method 1 (revise beta values)
# @time b, beta = constant_barrier(probabilities)
# @time beta_updated = post_compute_beta(b, probabilities)

# Optimize: method 2 (dual approach)
@time b_dual, beta_dual = dual_constant_barrier(probabilities)
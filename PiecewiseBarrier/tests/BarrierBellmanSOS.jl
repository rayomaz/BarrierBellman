""" Piecewise Barrier Functions based on Bellman's Equation

    © Rayan Mazouz

"""

# Import packages
using Revise
using PiecewiseBarrier

# Optimize certificate
system_dimension = 2
partitions_eps = 0.5
state_space = ([-3, 3], [-3, 3])
system_flag = "population_growth"
neural_flag = false

# Optimize controller
@time X = barrier_bellman_sos(system_dimension,
                                 partitions_eps,
                                 state_space,
                                 system_flag,
                                 neural_flag)

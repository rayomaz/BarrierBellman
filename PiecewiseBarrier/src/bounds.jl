""" Construct bounds using Python

    https://github.com/Zinoex/abstract-barrier

    https://github.com/Zinoex/bound_propagation

    © 

"""

#! To do: generate bounds in Python, load in Julia for piecewise barrier construction

# Bounds
bounds_file = "partitions/test/linearsystem_5.mat"
bounds = matopen(bounds_file)


# const Pij = [read(bounds, "lower_probability_bounds_A")]

# # Pij = lower_prob_A = 
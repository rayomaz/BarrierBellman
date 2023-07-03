""" Piecewise Barrier Functions based on Bellman's Equation

    © Rayan Mazouz

"""

# Import packages
using Revise, BenchmarkTools

using PiecewiseBarrier, MosekTools
using MultivariatePolynomials, DynamicPolynomials
using LazySets

using HDF5

# System
@polyvar x
fx = 0.95 * x

# Rayan: noise not needed in piecewise, embedded in bounds on P(x) and E(x)
# Frederik: I would still keep it here, so that we can test different models.
#           That small amount of data requires literally only 8 bytes, 
#           but is such a pain if we need it at one point, and don't have it.
σ = 0.1           

system = AdditiveGaussianPolynomialSystem(x, fx, σ)

# Bounds

# bounds_file = "partitions/test/linearsystem_5.mat"
# bounds = matopen(bounds_file)
# lower_partitions = read(bounds, "lower_partition")
# upper_partitions = read(bounds, "upper_partition")
# state_partitions = hcat(lower_partitions, upper_partitions)
# state_partitions = [Hyperrectangle(low=[low], high=[high]) for (low, high) in eachrow(state_partitions)]

# lower_prob_A = read(bounds, "lower_probability_bounds_A")
# lower_prob_b = read(bounds, "lower_probability_bounds_b")

# upper_prob_A = read(bounds, "upper_probability_bounds_A")
# upper_prob_b = read(bounds, "upper_probability_bounds_b")
# prob_bounds = (lower_prob_A, lower_prob_b), (upper_prob_A, upper_prob_b)


tocolumnmajor(dset) = permutedims(dset, reverse(1:ndims(dset)))

bounds_file = "partitions/linearsystem_250.hdf5"
file = h5open(bounds_file, "r")
lower_partitions = tocolumnmajor(read(file["partitioning/lower"]))
upper_partitions = tocolumnmajor(read(file["partitioning/upper"]))
state_partitions = [Hyperrectangle(low=low, high=high) for (low, high) in zip(eachrow(lower_partitions), eachrow(upper_partitions))]

lower_prob_A = tocolumnmajor(read(file["prob_bounds/lower/A"]))
lower_prob_b = tocolumnmajor(read(file["prob_bounds/lower/b"]))

upper_prob_A = tocolumnmajor(read(file["prob_bounds/upper/A"]))
upper_prob_b = tocolumnmajor(read(file["prob_bounds/upper/b"]))

prob_bounds = (lower_prob_A, lower_prob_b), (upper_prob_A, upper_prob_b)

unsafe_lower_prob_A = tocolumnmajor(read(file["unsafe_prob_bounds/lower/A"]))
unsafe_lower_prob_b = tocolumnmajor(read(file["unsafe_prob_bounds/lower/b"]))

unsafe_upper_prob_A = tocolumnmajor(read(file["unsafe_prob_bounds/upper/A"]))
unsafe_upper_prob_b = tocolumnmajor(read(file["unsafe_prob_bounds/upper/b"]))

unsafe_prob_bounds = (lower_prob_A, lower_prob_b), (upper_prob_A, upper_prob_b)

close(file)

# Optimization flags
initial_set = Hyperrectangle(low=[-0.05], high=[0.05])

# Optimize using Mosek as the SDP solver
optimizer = optimizer_with_attributes(Mosek.Optimizer,
    "MSK_DPAR_INTPNT_TOL_STEP_SIZE" => 1e-6,
    "MSK_IPAR_OPTIMIZER" => 0,
    "MSK_IPAR_BI_CLEAN_OPTIMIZER" => 0,
    "MSK_IPAR_NUM_THREADS" => 16,
    "MSK_IPAR_PRESOLVE_USE" => 0)

@time piecewise_barrier(optimizer, system, state_partitions, prob_bounds, unsafe_prob_bounds, initial_set)
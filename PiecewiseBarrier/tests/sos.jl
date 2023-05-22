"""
    - Generation of Stochastic Barrier Functions

    © Rayan Mazouz

"""

# Stochastic Barrier Verification
using Revise, BenchmarkTools

using PiecewiseBarrier, MosekTools
using MultivariatePolynomials, DynamicPolynomials
using LazySets

using DelimitedFiles

# System
@polyvar x
fx = 0.95 * x
σ = 0.1

system = AdditiveGaussianPolynomialSystem(x, fx, σ)

# Optimization flags
initial_state_partition = 3

# State partitions
state_partitions = readdlm("partitions/test/state_partitions.txt", ' ')
state_partitions = [Hyperrectangle(low=[low], high=[high]) for (low, high) in eachrow(state_partitions)]
state_space = state_space_generation(state_partitions)

# Optimization using Mosek as the SDP solver
optimizer = optimizer_with_attributes(Mosek.Optimizer,
    "MSK_DPAR_INTPNT_TOL_STEP_SIZE" => 1e-6,
    "MSK_IPAR_OPTIMIZER" => 0,
    "MSK_IPAR_BI_CLEAN_OPTIMIZER" => 0,
    "MSK_IPAR_NUM_THREADS" => 16,
    "MSK_IPAR_PRESOLVE_USE" => 0)

eta, beta = @time sos_barrier(optimizer, system, state_space, state_partitions, initial_state_partition)
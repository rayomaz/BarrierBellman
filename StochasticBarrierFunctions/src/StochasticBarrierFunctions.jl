module StochasticBarrierFunctions

using LinearAlgebra, SparseArrays
using Combinatorics: doublefactorial

using SpecialFunctions: erf, logerf, logerfc
using IrrationalConstants: invsqrt2, sqrtπ
import LogExpFunctions
# TODO: Make Mosek and Ipopt optional through extensions
using JuMP, MosekTools, Mosek, Ipopt, HiGHS, SCS, FrankWolfe

default_lp_solver() = HiGHS.Optimizer
default_socp_solver() = SCS.Optimizer
default_sdp_solver() = Mosek.Optimizer
default_non_linear_solver() = Ipopt.Optimizer

using MultivariatePolynomials, DynamicPolynomials
using PolyJuMP, SumOfSquares

using LazySets, Polyhedra, CDDLib
using Optimisers, ParameterSchedulers
using ReachabilityBase.Commutative

const MP = MultivariatePolynomials
const APL{T} = MP.AbstractPolynomialLike{T}

# Basic system types
include("system.jl")
export AdditiveGaussianLinearSystem, AdditiveGaussianUncertainPWASystem, UncertainPWARegion
export dynamics, noise_distribution, dimensionality

include("region.jl")
export region, prob_lower, prob_upper, prob_unsafe_lower, prob_unsafe_upper

include("probabilities.jl")
export transition_probabilities, plot_posterior
export TransitionProbabilityAlgorithm, BoxApproximation, GlobalSolver, FrankWolfeSolver

include("barrier.jl")
export SumOfSquaresBarrier, PiecewiseConstantBarrier
export barrier, eta, beta, psafe

function synthesize_barrier end
export synthesize_barrier

include("barrier_algorithms/base.jl")
include("barrier_algorithms/beta.jl")

# Various barrier synthesis algorithms
include("barrier_algorithms/upper_bound_barrier.jl")
include("barrier_algorithms/iterative_upper_bound_barrier.jl")
include("barrier_algorithms/dual_barrier.jl")
include("barrier_algorithms/gradient_descent_barrier.jl")
include("barrier_algorithms/sum_of_squares_barrier.jl")


# Plotting
using Plots

include("plots.jl")
export plot_environment, plot_3d_barrier


# Helper functions for loading data
using YAXArrays, YAXArrayBase, DimensionalData
using MAT.MAT_v4, MAT.MAT_v5, MAT.MAT_HDF5, DelimitedFiles

const MatlabFile = Union{MAT_v4.Matlabv4File, MAT_v5.Matlabv5File, MAT_HDF5.MatlabHDF5File}

include("data.jl")
export load_regions, load_dynamics, load_probabilities

end # module StochasticBarrierFunctions

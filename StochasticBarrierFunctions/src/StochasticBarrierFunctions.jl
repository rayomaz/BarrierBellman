module StochasticBarrierFunctions

using LinearAlgebra, SparseArrays
using Combinatorics: doublefactorial

using SpecialFunctions: erf, logerf, logerfc
using IrrationalConstants: invsqrt2, sqrtπ
import LogExpFunctions
using JuMP, Ipopt, HiGHS, SCS, FrankWolfe

default_lp_solver() = HiGHS.Optimizer
default_socp_solver() = SCS.Optimizer
default_sdp_solver() = SCS.Optimizer
default_non_linear_solver() = Ipopt.Optimizer

using MultivariatePolynomials, DynamicPolynomials
using PolyJuMP, SumOfSquares

using Distributions: quantile, Normal
using ProgressMeter: Progress, next!
using DimensionalData: DimArray, Dim

using LazySets, Polyhedra, CDDLib
using Optimisers, ParameterSchedulers
using ReachabilityBase.Commutative

const MP = MultivariatePolynomials
const APL{T} = MP.AbstractPolynomialLike{T}

# Basic system types
include("system.jl")
export AdditiveGaussianLinearSystem, AdditiveGaussianPolySystem, AdditiveGaussianUncertainPWASystem, UncertainPWARegion
export dynamics, noise_distribution, dimensionality

include("region.jl")
export RegionWithProbabilities
export region, prob_lower, prob_upper, prob_unsafe_lower, prob_unsafe_upper

include("barrier.jl")
export SumOfSquaresBarrier, PiecewiseConstantBarrier
export barrier, eta, beta, psafe, total_time

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

# Submodules
include("Plots/Plots.jl")
include("Data/Data.jl")

using StochasticBarrierFunctions.Data

include("probabilities.jl")
export transition_probabilities, plot_posterior
export TransitionProbabilityAlgorithm, BoxApproximation, GlobalSolver, FrankWolfeSolver

end # module StochasticBarrierFunctions

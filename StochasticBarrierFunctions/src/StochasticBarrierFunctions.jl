module StochasticBarrierFunctions

using LinearAlgebra, SparseArrays, StaticArrays
using LoopVectorization, ProgressMeter
using Distributions, Combinatorics, StatsBase

using SpecialFunctions: erf, logerf, logerfc
using IrrationalConstants: invsqrt2, sqrtπ
import LogExpFunctions
# TODO: Make Mosek and Ipopt optional through extensions
using JuMP, MosekTools, Mosek, Ipopt, HiGHS

function default_lp_solver end
function default_sdp_solver end
function default_non_linear_solver end

default_lp_solver() = HiGHS.Optimizer
default_qp_solver() = HiGHS.Optimizer
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
export AbstractDiscreteTimeStochasticSystem, AbstractAdditiveGaussianSystem
export AdditiveGaussianLinearSystem, AdditiveGaussianUncertainPWASystem, UncertainPWARegion
export dynamics, noise_distribution, dimensionality

include("region.jl")
export region, prob_lower, prob_upper, prob_unsafe_lower, prob_unsafe_upper, update_regions

include("probabilities.jl")
export transition_probabilities, plot_posterior
export TransitionProbabilityAlgorithm, BoxApproximation, GlobalSolver, FrankWolfe

include("barrier.jl")
export StochasticBarrier, SOSBarrier, ConstantBarrier
export StochasticBarrierAlgorithm, DualAlgorithm, UpperBoundAlgorithm, IterativeUpperBoundAlgorithm
export GradientDescentAlgorithm, StochasticGradientDescentAlgorithm, SumOfSquaresAlgorithm

function synthesize_barrier end
export synthesize_barrier

include("beta.jl")

# Various barrier synthesis algorithms
include("constant_barrier.jl")
include("iterative_barrier.jl")
include("dual_barrier.jl")
include("gradient_descent_barrier.jl")
include("sum_of_squares_barrier.jl")


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

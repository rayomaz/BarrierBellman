module PiecewiseConstant

using LinearAlgebra, StatsBase, Combinatorics
using MultivariatePolynomials, DynamicPolynomials, MultivariateBases
using SpecialFunctions: erf
using JuMP, HiGHS, Optim, NLopt, Ipopt, MosekTools, Mosek
using LazySets, Polyhedra, CDDLib
using Optimisers, ParameterSchedulers
using ReachabilityBase.Commutative

const MP = MultivariatePolynomials
const MB = MultivariateBases

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
export TransitionProbabilityMethod, BoxApproximation, GradientDescent

include("barrier.jl")
export StochasticBarrier, SOSBarrier, ConstantBarrier
export StochasticBarrierAlgorithm, DualAlgorithm, UpperBoundAlgorithm, IterativeUpperBoundAlgorithm
export GradientDescentAlgorithm, SumOfSquaresAlgorithm

function synthesize_barrier end
export synthesize_barrier

include("beta.jl")

# Various barrier synthesis algorithms
include("constant_barrier.jl")
include("iterative_barrier.jl")
include("dual_barrier.jl")
include("gradient_descent_barrier.jl")


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

end # module PiecewiseConstant

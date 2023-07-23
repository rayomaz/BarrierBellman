module PiecewiseConstant

using LinearAlgebra, StatsBase, Combinatorics
using MultivariatePolynomials, DynamicPolynomials, MultivariateBases
using SpecialFunctions: erf
using JuMP, HiGHS, Optim, NLopt, Ipopt
using LazySets, Polyhedra, CDDLib
using ReachabilityBase.Commutative
using YAXArrays, YAXArrayBase, DimensionalData
using MAT.MAT_v4, MAT.MAT_v5, MAT.MAT_HDF5, DelimitedFiles

const MatlabFile = Union{MAT_v4.Matlabv4File, MAT_v5.Matlabv5File, MAT_HDF5.MatlabHDF5File}

const MP = MultivariatePolynomials

const MB = MultivariateBases

include("region.jl")
export region, prob_lower, prob_upper, prob_unsafe_lower, prob_unsafe_upper, update_regions

include("system.jl")
export AbstractDiscreteTimeStochasticSystem, AbstractAdditiveGaussianSystem
export AdditiveGaussianLinearSystem, AdditiveGaussianUncertainPWASystem, UncertainPWARegion
export dynamics, noise_distribution, dimensionality

include("data.jl")
export load_regions, load_dynamics, load_probabilities

include("probabilities.jl")
export transition_probabilities, neural_transition_probabilities

include("constant_barrier.jl")
export constant_barrier

include("post_compute.jl")
export post_compute_beta, accelerated_post_compute_beta

include("iterative_barrier.jl")
export iterative_barrier

include("dual_barrier.jl")
export dual_constant_barrier

include("tests.jl")
export sum_probabilities, sum_barrier_probabilities

end # module PiecewiseConstant

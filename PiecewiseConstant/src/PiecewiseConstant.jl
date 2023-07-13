module PiecewiseConstant

using LinearAlgebra, StatsBase, Combinatorics
using MultivariatePolynomials, DynamicPolynomials, MultivariateBases
using SpecialFunctions: erf
using JuMP, HiGHS, Optim, NLopt, Ipopt
using LazySets, Polyhedra, CDDLib
using Distributed
using MAT.MAT_v4, MAT.MAT_v5, MAT.MAT_HDF5, DelimitedFiles 

const MatlabFile = Union{MAT_v4.Matlabv4File, MAT_v5.Matlabv5File, MAT_HDF5.MatlabHDF5File}

const MP = MultivariatePolynomials

const MB = MultivariateBases

include("region.jl")
export region, prob_lower, prob_upper, prob_unsafe_lower, prob_unsafe_upper

include("utility.jl")
export vectorize, state_space_generation, read_regions

include("system.jl")
export AbstractDiscreteTimeStochasticSystem, AdditiveGaussianPolynomialSystem
export variables, dynamics, noise_distribution, dimensionality

include("probabilities.jl")
export linear_transition_probabilities, neural_transition_probabilities

include("constant_barrier.jl")
export constant_barrier

include("post_compute.jl")
export post_compute_beta, accelerated_post_compute_beta

include("dual_barrier.jl")
export dual_constant_barrier

include("tests.jl")
export sum_probabilities, sum_barrier_probabilities

end # module PiecewiseConstant

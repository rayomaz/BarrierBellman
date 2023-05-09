module PiecewiseBarrier

using SumOfSquares
using MultivariatePolynomials, DynamicPolynomials
const MP = MultivariatePolynomials

using MultivariateBases
const MB = MultivariateBases

using MosekTools
using LazySets
using StatsBase
using Combinatorics
using LinearAlgebra
using GLPK
using Optim
using JuMP
import JuMP.@variable
using LaTeXStrings
using SpecialFunctions: erf

include("constants.jl")

include("utility.jl")
export state_space_generation, vectorize

include("system.jl")
export AbstractDiscreteTimeStochasticSystem, AdditiveGaussianPolynomialSystem
export variables, dynamics, noise_distribution

include("expectation.jl")

include("exponential.jl")
export exponential_bounds

include("validate.jl")

include("sumofsquares.jl")
export sos_barrier

include("barrier.jl")

include("certificate.jl")
include("piecewise.jl")
export piecewise_barrier

include("constantbarrier.jl")
export constant_barrier


end # module PiecewiseBarrier

module PiecewiseBarrier

using SumOfSquares
using MultivariatePolynomials, DynamicPolynomials
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

include("constants.jl")

include("utility.jl")
export state_space_generation, maximum_beta_constraint

include("expectation.jl")

include("validate.jl")

include("sumofsquares.jl")
export sos_barrier

include("piecewise.jl")
export piecewise_barrier

end # module PiecewiseBarrier

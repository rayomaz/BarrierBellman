module PiecewiseBarrier

using SumOfSquares
using DynamicPolynomials
using MosekTools
using Polynomials
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
export state_space_generation

include("expectation.jl")

include("validate.jl")

include("sumofsquares.jl")
export sos_barrier

include("piecewise.jl")
export piecewise_barrier

end # module PiecewiseBarrier

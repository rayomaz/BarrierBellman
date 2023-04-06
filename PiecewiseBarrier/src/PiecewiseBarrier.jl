module PiecewiseBarrier

using SumOfSquares
using DynamicPolynomials
using MosekTools
using MAT
using Polynomials
using StatsBase
using Combinatorics
using LinearAlgebra
using GLPK
using Optim
using JuMP
import JuMP.@variable

include(joinpath(@__DIR__,"utility.jl"))

include(joinpath(@__DIR__,"expectation.jl"))

include(joinpath(@__DIR__,"sumofsquares.jl"))
export sos_barrier

include(joinpath(@__DIR__,"piecewise.jl"))
export piecewise_barrier

end # module PiecewiseBarrier

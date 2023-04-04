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

include(joinpath(@__DIR__,"functions.jl"))

include(joinpath(@__DIR__,"expectation.jl"))

include(joinpath(@__DIR__,"sos_optim.jl"))
export optimization

include(joinpath(@__DIR__,"optimizer.jl"))
export barrier_bellman_sos

end # module PiecewiseBarrier

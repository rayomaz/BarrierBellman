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

include(joinpath(@__DIR__,"optimizer.jl"))
export add_it_up #barrier_bellman_sos

end # module PiecewiseBarrier

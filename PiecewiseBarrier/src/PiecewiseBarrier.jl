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

include(joinpath(@__DIR__,"expectation.jl"))
export expectation_noise

include(joinpath(@__DIR__,"optimizer.jl"))
export barrier_bellman_sos

end # module PiecewiseBarrier

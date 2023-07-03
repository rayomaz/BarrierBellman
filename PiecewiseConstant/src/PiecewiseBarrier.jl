module PiecewiseBarrier

using LazySets
using StatsBase
using Combinatorics
using LinearAlgebra
using GLPK
using Optim
using JuMP
import JuMP.@variable
using SpecialFunctions: erf
using NLopt
using Ipopt
using Distributed

include("probabilities.jl")
export transition_probabilities

include("constant_barrier.jl")
export constant_barrier

include("post_compute.jl")
export post_compute_beta

include("dual_barrier.jl")
export dual_constant_barrier

end # module PiecewiseBarrier

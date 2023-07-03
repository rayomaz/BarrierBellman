module PiecewiseConstant

using LinearAlgebra, StatsBase, Combinatorics, SpecialFunctions: erf
using JuMP, HiGHS, Optim, NLopt, Ipopt
using Distributed

const MatlabFile = Union{MatlabHDF5File, Matlabv4File, Matlabv5File}

# include("probabilities.jl")
# export transition_probabilities

include("constant_barrier.jl")
export constant_barrier

include("post_compute.jl")
export post_compute_beta

include("dual_barrier.jl")
export dual_constant_barrier

end # module PiecewiseConstant

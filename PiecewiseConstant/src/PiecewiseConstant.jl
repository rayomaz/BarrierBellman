module PiecewiseConstant

using LinearAlgebra, StatsBase, Combinatorics
using SpecialFunctions: erf
using JuMP, HiGHS, Optim, NLopt, Ipopt
using LazySets, Polyhedra, CDDLib
using Distributed
using MAT.MAT_v4, MAT.MAT_v5, MAT.MAT_HDF5

const MatlabFile = Union{MAT_v4.Matlabv4File, MAT_v5.Matlabv5File, MAT_HDF5.MatlabHDF5File}

include("constant_barrier.jl")
export constant_barrier

include("post_compute.jl")
export post_compute_beta

include("dual_barrier.jl")
export dual_constant_barrier

include("utils.jl")
export sum_probabilities, sum_barrier_probabilities

end # module PiecewiseConstant

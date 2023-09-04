""" Piecewise Barrier Function: Neural Network Dynamic Model [Pendulum]

    © Rayan Mazouz

"""

# Import packages
using Revise, BenchmarkTools
using StochasticBarrierFunctions, LazySets, LinearAlgebra
using YAXArrays, NetCDF

# System
F = 1.0
dim = 1

A = F*I(dim)
b = zeros(dim)
σ = 0.1*ones(dim)

system = AdditiveGaussianLinearSystem(A, b, σ)

# Create partitions
Δ = 0.04
lower_bound = -1.0
upper_bound = 0.5
Q = Int(round((upper_bound - lower_bound) / Δ))
x = range(lower_bound, upper_bound, length=Q+1)

if dim == 1
    state_partitions = [Hyperrectangle(low=[low_x], high=[high_x]) 
                        for (low_x, high_x) in zip(x[1:end-1], x[2:end])]
elseif dim == 2
    state_partitions = [Hyperrectangle(low=[low_x, low_y], high=[high_x, high_y]) 
                        for (low_x, high_x) in zip(x[1:end-1], x[2:end])
                            for (low_y, high_y) in zip(x[1:end-1], x[2:end])]
end
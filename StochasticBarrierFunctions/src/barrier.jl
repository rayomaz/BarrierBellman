### Barrier types
abstract type StochasticBarrier end

struct SOSBarrier{T} <: StochasticBarrier
    p::APL{T}
end
(B::SOSBarrier)(x) = B.p(x)

struct ConstantBarrier{T, VT<:AbstractVector{T}, ST<:AbstractVector{<:LazySet{T}}} <: StochasticBarrier
    regions::ST
    b::VT

    function ConstantBarrier(regions::ST, b::VT) where {T, VT<:AbstractVector{T}, ST<:AbstractVector{<:LazySet{T}}}
        if length(regions) != length(b)
            throw(DimensionMismatch("The number of regions and the number of barriers must be equal"))
        end

        new{T, VT, ST}(regions, b)
    end
end

function (B::ConstantBarrier)(x)
    for (i, P) in enumerate(regions)
        if x in P
            return b[i]
        end
    end
    
    throw(DomainError(x, "The point x resides outside the defined domain of the barrier"))
end
Base.length(B::ConstantBarrier) = length(B.b)
Base.getindex(B::ConstantBarrier, i) = B.b[i]
Base.iterate(B::ConstantBarrier) = iterate(B.b)
Base.iterate(B::ConstantBarrier, state) = iterate(B.b, state)


### Algorithm types
abstract type StochasticBarrierAlgorithm end
abstract type ConstantBarrierAlgorithm <: StochasticBarrierAlgorithm end
abstract type SOSBarrierAlgorithm <: StochasticBarrierAlgorithm end

Base.@kwdef struct DualAlgorithm <: ConstantBarrierAlgorithm
    ϵ = 1e-6
    linear_solver = default_lp_solver()
end

Base.@kwdef struct UpperBoundAlgorithm <: ConstantBarrierAlgorithm
    ϵ = 1e-6
    linear_solver = default_lp_solver()
end

Base.@kwdef struct IterativeUpperBoundAlgorithm <: ConstantBarrierAlgorithm
    ϵ = 1e-6
    num_iterations = 10
    guided = true
    distributed = false
end

Base.@kwdef struct GradientDescentAlgorithm <: ConstantBarrierAlgorithm
    num_iterations = 1000
    initial_lr = 1e-2
    decay = 0.995
    momentum = 0.99
end

Base.@kwdef struct SumOfSquaresAlgorithm <: SOSBarrierAlgorithm
    ϵ = 1e-6
    barrier_degree = 4
    lagrange_degree = 2
    sdp_solver = default_sdp_solver()
end
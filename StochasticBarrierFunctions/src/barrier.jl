### Barrier types
abstract type StochasticBarrier end

struct SumOfSquaresBarrier{T} <: StochasticBarrier
    p::APL{T}
end
(B::SumOfSquaresBarrier)(x) = B.p(x)

struct PiecewiseConstantBarrier{T, VT<:AbstractVector{T}, ST<:AbstractVector{<:LazySet{T}}} <: StochasticBarrier
    regions::ST
    b::VT

    function PiecewiseConstantBarrier(regions::ST, b::VT) where {T, VT<:AbstractVector{T}, ST<:AbstractVector{<:LazySet{T}}}
        if length(regions) != length(b)
            throw(DimensionMismatch("The number of regions and the number of local barriers must be equal"))
        end

        new{T, VT, ST}(regions, b)
    end
end

function (B::PiecewiseConstantBarrier)(x)
    for (i, P) in enumerate(regions)
        if x in P
            return B[i]
        end
    end
    
    throw(DomainError(x, "The point x resides outside the defined domain of the barrier"))
end

Base.length(B::PiecewiseConstantBarrier) = length(B.b)
Base.getindex(B::PiecewiseConstantBarrier, i) = B.b[i]
Base.iterate(B::PiecewiseConstantBarrier) = iterate(B.b)
Base.iterate(B::PiecewiseConstantBarrier, state) = iterate(B.b, state)



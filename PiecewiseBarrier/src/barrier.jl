
struct LinearBarrier{T, M<:AbstractMonomialLike}
    x::AbstractVector{M}
    A::AbstractVector{T}
    b::T
end

MP.polynomial(barrier::LinearBarrier) = dot(barrier.A, barrier.x) + barrier.b

function Base.show(io::IO, barrier::LinearBarrier)
    print(io, "LinearBarrier[")
    print(io, polynomial(barrier))
    print(io, "]")
end

# Create Linear Barrier Function
function barrier_construct(system, A, b) 
    x = variables(system)
    return LinearBarrier(x, A, b)
end

# Compute the final barrier certificate
piecewise_barrier_certificate(barrier::LinearBarrier) = LinearBarrier(barrier.x, value.(barrier.A), value.(barrier.b))
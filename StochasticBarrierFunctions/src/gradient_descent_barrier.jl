""" Piecewise barrier function construction

    © Rayan Mazouz

"""

# Optimization function
function synthesize_barrier(alg::GradientDescentAlgorithm, regions::Vector{<:RegionWithProbabilities}, initial_region::LazySet, obstacle_region::LazySet; time_horizon=1)
    n = length(regions)

    initial_indices = findall(X -> !isdisjoint(initial_region, region(X)), regions)
    unsafe_indices = findall(X -> !isdisjoint(obstacle_region, region(X)), regions)

    ws = GradientDescentWorkspace(n, initial_indices, unsafe_indices)
    project!(ws)

    decay = Exp(λ = alg.initial_lr, γ = alg.decay)
    optim = Optimisers.Nesterov(alg.initial_lr, alg.momentum)

    state = Optimisers.setup(optim, ws.B)

    p = [zero(region.gap) for region in regions]
    q = collect(UnitRange{Int64}(1, n + 1))

    prev_q = copy(q)

    for k in 0:alg.num_iterations
        state = gradient_descent_barrier_iteration!(ws, state, regions, prev_q, q, p, decay(k))
    end

    η = maximum(ws.B_init)

    sortperm!(q, ws.B, rev=true)
    ivi_prob!.(p, regions, tuple(q))

    βⱼ = beta(ws, p)

    @info "Solution Gradient Descent" η β=maximum(βⱼ) Pₛ=1 - (η + maximum(βⱼ) * time_horizon) iterations=alg.num_iterations

    Xs = map(region, regions)
    return ConstantBarrier(Xs, ws.B_regions), βⱼ
end

mutable struct GradientDescentWorkspace{T, BT<:AbstractVector{T}, VT<:AbstractVector{T}, RT<:AbstractVector{T}}
    B::BT
    dB::BT
    β::BT  # This only functions as a cache. It should never be accessed directly. 

    B_init::VT
    B_unsafe::VT
    B_regions::RT
end

function GradientDescentWorkspace(n::Integer, initial_indices::AbstractVector, unsafe_indices::AbstractVector)
    B = fill(0.5, n + 1)
    dB = similar(B)
    β = zeros(n)

    # The sinking state is considered unsafe
    push!(unsafe_indices, n + 1)

    B_init = @view(B[initial_indices])
    B_unsafe = @view(B[unsafe_indices])
    B_regions = @view(B[1:end - 1])

    return GradientDescentWorkspace(B, dB, β, B_init, B_unsafe, B_regions)
end

function project!(ws::GradientDescentWorkspace)
    # Projection onto [0, 1]^n x {1}
    clamp!(ws.B, 0, 1)
    ws.B_init .= 0
    ws.B_unsafe .= 1
end

function beta(ws::GradientDescentWorkspace{T}, p) where {T}
    ws.β .= dot.(tuple(ws.B), p)
    ws.β .-= ws.B_regions
    clamp!(ws.β, T(0), T(Inf))

    return ws.β
end

function gradient!(ws::GradientDescentWorkspace, p)
    βⱼ = beta(ws, p)
    j = argmax(βⱼ)

    ws.dB .= p[j]
    ws.dB[j] -= 1

    return ws.dB
end

function gradient_descent_barrier_iteration!(ws, state, regions, prev_q, q, p, lr)
    sortperm!(q, ws.B, rev=true)

    if q != prev_q
        copyto!(prev_q, q)
        ivi_prob!.(p, regions, tuple(q))
    end

    gradient!(ws, p)

    Optimisers.adjust!(state, lr)
    state, ws.B = Optimisers.update!(state, ws.B, ws.dB)

    project!(ws)

    return state
end
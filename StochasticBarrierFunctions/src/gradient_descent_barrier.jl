""" Piecewise barrier function construction

    © Rayan Mazouz

"""

# Optimization function
function synthesize_barrier(alg::GradientDescentAlgorithm, regions::Vector{<:RegionWithProbabilities}, initial_region::LazySet, obstacle_region::LazySet; time_horizon=1)
    ws, p, q = setup_gd(regions, initial_region, obstacle_region)

    decay = Exp(λ = alg.initial_lr, γ = alg.decay)
    optim = Optimisers.Nesterov(alg.initial_lr, alg.momentum)

    state = Optimisers.setup(optim, ws.B)

        state = gradient_descent_barrier_iteration!(ws, state, regions, p, q, decay(k))
    end

    η = maximum(ws.B_init)

    ivi_value_assignment!(ws, regions, p, q)
    βⱼ = beta!(ws, p)

    @info "Solution Gradient Descent" η β=maximum(βⱼ) Pₛ=1 - (η + maximum(βⱼ) * time_horizon) iterations=alg.num_iterations

    Xs = map(region, regions)
    return ConstantBarrier(Xs, ws.B_regions), βⱼ
end

function setup_gd(regions::Vector{<:RegionWithProbabilities}, initial_region::LazySet, obstacle_region::LazySet)
    n = length(regions)

    initial_indices = findall(X -> !isdisjoint(initial_region, region(X)), regions)
    unsafe_indices = findall(X -> !isdisjoint(obstacle_region, region(X)), regions)

    P̅ᵤ = map(X -> prob_unsafe_upper(X), regions)
    ws = GradientDescentWorkspace(P̅ᵤ, initial_indices, unsafe_indices)
    project!(ws)

    p = [zero(region.gap) for region in regions]
    q = collect(UnitRange{Int64}(1, n + 1))

    return ws, p, q
end

mutable struct GradientDescentWorkspace{T, BT<:AbstractVector{T}, VT<:AbstractVector{T}, RT<:AbstractVector{T}}
    B::BT
    dB::BT
    β::BT  # This only functions as a cache. It should never be accessed directly. 

    B_init::VT
    B_unsafe::VT
    B_regions::RT
    dB_regions::RT
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
    dB_regions = @view(dB[1:end - 1])

    return GradientDescentWorkspace(B, dB, β, B_init, B_unsafe, B_regions, dB_regions)
end

function GradientDescentWorkspace(P̅ᵤ::AbstractVector, initial_indices::AbstractVector, unsafe_indices::AbstractVector)
    n = length(P̅ᵤ)

    ws = GradientDescentWorkspace(n, initial_indices, unsafe_indices)
    ws.B_regions .= P̅ᵤ

    return ws
end

num_regions(ws::GradientDescentWorkspace) = length(ws.B_regions)

function project!(ws::GradientDescentWorkspace)
    # Projection onto [0, 1]^n x {1}
    clamp!(ws.B, 0, 1)
    ws.B_init .= 0
    ws.B_unsafe .= 1
end

function beta!(ws::GradientDescentWorkspace, p)
    ws.β .= dot.(tuple(ws.B), p)
    ws.β .-= ws.B_regions
    clamp!(ws.β, 0, Inf)

    return ws.β
end

function gradient!(ws::GradientDescentWorkspace, p; t=5000.0)
    # Gradient for the following loss: ||βⱼ||ₜ
    # This is an Lp-norm, which approaches a suprenum norm as t -> Inf

    # It turns out it is equivalent to a tempered LogSumExp loss, 1/t * log(sum(exp.(t .* x)))
    # where we assume xⱼ = ln(βⱼ)

    βⱼ = beta!(ws, p)

    z = norm(βⱼ, t)
    βⱼ ./= z
    βⱼ .^= t - 1
    
    βⱼ .*= -1

    ws.dB[end] = 0
    ws.dB_regions .= βⱼ
    for j in eachindex(βⱼ)
        ws.dB .-= βⱼ[j] .* p[j]
    end

    return ws.dB
end

function gradient_descent_barrier_iteration!(ws, state, regions, p, q, lr)
    ivi_value_assignment!(ws, regions, p, q)
    gradient!(ws, p)

    Optimisers.adjust!(state, lr)
    state, ws.B = Optimisers.update!(state, ws.B, ws.dB)

    project!(ws)

    return state
end

function ivi_value_assignment!(ws, regions, p, q)
    sortperm!(q, ws.B, rev=true)
        
    Threads.@threads for i in eachindex(p)
        @inbounds ivi_prob!(p[i], regions[i], q)
    end
end
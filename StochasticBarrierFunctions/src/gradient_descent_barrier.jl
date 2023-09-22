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
    initial_indices = findall(X -> !isdisjoint(initial_region, region(X)), regions)
    unsafe_indices = findall(X -> !isdisjoint(obstacle_region, region(X)), regions)

    P̅ᵤ = map(X -> prob_unsafe_upper(X), regions)
    ws = GradientDescentWorkspace(P̅ᵤ, initial_indices, unsafe_indices)
    project!(ws)

    p = [copy(region.gap) for region in regions]
    q = prepare_q(p)

    return ws, p, q
end

function prepare_q(p::VVT) where {VVT<:AbstractVector{<:AbstractVector}}
    n = length(p)
    return collect(UnitRange{Int64}(1, n + 1))
end

mutable struct PermutationSubset{T<:Integer, VT<:AbstractVector{T}}
    ptr::T
    items::VT
end

struct ReversiblePermutationItem{T<:Integer, VT<:AbstractVector{T}}
    value::T
    index::VT
end

function reset_subsets!(q_subsets)
    for j in eachindex(q_subsets)
        q_subsets[j].ptr = 1
    end
end

function populate_subsets!(q, q_order, q_subsets)
    reset_subsets!(q_subsets)

    for i in q
        qo = q_order[i]
        @assert qo.value == i

        for j in qo.index
            q_subsets[j].items[q_subsets[j].ptr] = qo.value
            q_subsets[j].ptr += 1
        end
    end
end

function prepare_q(p::VVT) where {VVT<:AbstractVector{<:AbstractSparseVector}}
    n = length(p)
    q = collect(UnitRange{Int64}(1, n + 1))

    q_order = Vector{ReversiblePermutationItem{Int64, Vector{Int64}}}(undef, n + 1)
    for i in 1:n + 1
        q_order[i] = ReversiblePermutationItem(i, Int64[])
    end

    q_subsets = Vector{PermutationSubset{Int64, Vector{Int64}}}(undef, n)
    for j in 1:n
        q_subsets[j] = PermutationSubset(1, Vector{Int64}(undef, nnz(p[j])))

        ids = SparseArrays.nonzeroinds(p[j])
        for i in ids
            push!(q_order[i].index, j)
        end
    end

    return q, q_order, q_subsets
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

function gradient!(ws::GradientDescentWorkspace, p::VVT; t=5000.0) where {VVT<:AbstractVector{<:AbstractVector}}
    # Gradient for the following loss: ||βⱼ||ₜ
    # This is an Lp-norm, which approaches a suprenum norm as t -> Inf

    # It turns out it is equivalent to a tempered LogSumExp loss, 1/t * log(sum(exp.(t .* x)))
    # where we assume xⱼ = ln(βⱼ)

    βⱼ = beta!(ws, p)

    logz = log(norm(βⱼ, t))
    βⱼ .= log.(βⱼ)
    βⱼ .-= logz
    βⱼ .*= t - 1

    ws.dB[end] = 0
    ws.dB_regions .= (-).(exp.(βⱼ))
    for j in eachindex(βⱼ)
        logspace_add_prod!(ws.dB, βⱼ[j], p[j])
    end

    return ws.dB
end

function logspace_add_prod!(dB, β, p::VT) where {VT<:AbstractVector}
    dB .+= exp.(β .+ log.(p))
end

function logspace_add_prod!(dB, β, p::VT) where {VT<:AbstractSparseVector}
    ids = SparseArrays.nonzeroinds(p)
    values = nonzeros(p)

    for (i, v) in zip(ids, values)
        dB[i] += exp(β + log(v))
    end
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

function ivi_value_assignment!(ws, regions, p::VVT, q) where {VVT<:AbstractVector{<:AbstractVector}}
    sortperm!(q, ws.B, rev=true)
        
    Threads.@threads for j in eachindex(p)
        @inbounds ivi_prob!(p[j], regions[j], q)
    end
end

function ivi_value_assignment!(ws, regions, p::VVT, q) where {VVT<:AbstractVector{<:AbstractSparseVector}}
    q, q_order, q_subsets = q
    sortperm!(q, ws.B, rev=true)
    populate_subsets!(q, q_order, q_subsets)
        
    Threads.@threads for j in eachindex(p)
        @inbounds ivi_prob!(p[j], regions[j], q_subsets[j].items)
    end
end
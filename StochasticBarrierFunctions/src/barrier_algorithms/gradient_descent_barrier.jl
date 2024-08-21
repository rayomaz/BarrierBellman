export GradientDescentAlgorithm, StochasticGradientDescentAlgorithm, GradientDescentAlgResult

abstract type ConstantGDBarrierAlgorithm <: ConstantBarrierAlgorithm end

Base.@kwdef struct GradientDescentAlgorithm <: ConstantGDBarrierAlgorithm
    num_iterations = 10000
    initial_lr = 1e-2
    decay = 0.9999
    momentum = 0.9
end

Base.@kwdef struct StochasticGradientDescentAlgorithm <: ConstantGDBarrierAlgorithm
    num_iterations = 30000
    subsampling_fraction = 0.1
    initial_lr = 1e-2
    decay = 0.999875
    momentum = 0.9
end

struct GradientDescentAlgResult <: BarrierResult
    B::PiecewiseConstantBarrier
    η::Float64
    βs::Vector{Float64}
    synthesis_time::Float64  # Total time to solve the optimization problem in seconds
end

barrier(res::GradientDescentAlgResult) = res.B
eta(res::GradientDescentAlgResult) = res.η
beta(res::GradientDescentAlgResult) = maximum(res.βs)
betas(res::GradientDescentAlgResult) = res.βs
total_time(res::GradientDescentAlgResult) = res.synthesis_time

# Optimization function
function synthesize_barrier(alg::ConstantGDBarrierAlgorithm, regions::Vector{<:RegionWithProbabilities}, initial_region::LazySet, obstacle_region::LazySet; time_horizon=1)
    synthesis_time = @elapsed begin
        ws, p, q = setup_gd(alg, regions, initial_region, obstacle_region)

        decay = Exp(λ = alg.initial_lr, γ = alg.decay)
        optim = Optimisers.Nesterov(alg.initial_lr, alg.momentum)

        state = Optimisers.setup(optim, ws.B)

        for k in 0:alg.num_iterations
            @debug "Iteration $k/$(alg.num_iterations)"
            state = gradient_descent_barrier_iteration!(ws, state, regions, p, q, decay(k); time_horizon=time_horizon)
        end

        η = maximum(ws.B_init)

        ivi_value_assignment!(ws, regions, p, q)
        βⱼ = beta!(ws, p)

        Xs = map(region, regions)
    end

    res = GradientDescentAlgResult(PiecewiseConstantBarrier(Xs, ws.B_regions), η, βⱼ, synthesis_time)

    @info "Solution Gradient Descent" η=eta(res) β=beta(res) Pₛ=psafe(res, time_horizon) time=total_time(res) iterations=alg.num_iterations

    return res
end

function setup_gd(alg, regions::Vector{<:RegionWithProbabilities}, initial_region::LazySet, obstacle_region::LazySet)
    initial_indices = findall(X -> !isdisjoint(initial_region, region(X)), regions)
    unsafe_indices = findall(X -> !isdisjoint(obstacle_region, region(X)), regions)

    P̅ᵤ = map(X -> prob_unsafe_upper(X), regions)
    ws = setup_workspace(alg, P̅ᵤ, initial_indices, unsafe_indices)
    project!(ws)

    p = [copy(region.gap) for region in regions]
    q = prepare_q(p)

    return ws, p, q
end

function setup_workspace(::GradientDescentAlgorithm, P̅ᵤ, initial_indices, unsafe_indices)
    return GradientDescentWorkspace(P̅ᵤ, initial_indices, unsafe_indices)
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

function Base.empty!(subset::PermutationSubset)
    subset.ptr = 1
end

function Base.push!(subset::PermutationSubset, item)
    subset.items[subset.ptr] = item
    subset.ptr += 1
end

function reset_subsets!(q_subsets)
    for subset in q_subsets
        empty!(subset)
    end
end

function populate_subsets!(q, q_order, q_subsets)
    reset_subsets!(q_subsets)

    for i in q
        qo = q_order[i]
        @assert qo.value == i

        for j in qo.index
            push!(q_subsets[j], qo.value)
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

function project_gradient!(ws::GradientDescentWorkspace, lr)
    # Projection so that B(k + 1) ∈ [0, 1]^n
    rmul!(ws.dB, -lr)
    ws.dB .+= ws.B
    clamp!(ws.dB, 0, 1)
    ws.dB .-= ws.B
    rmul!(ws.dB, -1/lr)
end

function beta!(ws, p)
    Threads.@threads for j in eachindex(ws.β)
        ws.β[j] = dot(ws.B, p[j])
    end

    # ws.β .= dot.(tuple(ws.B), p)
    ws.β .-= ws.B_regions
    clamp!(ws.β, 0, Inf)

    return ws.β
end

function gradient!(ws::GradientDescentWorkspace, p::VVT; time_horizon, t=20.0) where {VVT<:AbstractVector{<:AbstractVector}}
    # Gradient for the following loss: ||βⱼ||ₜ
    # This is an Lp-norm, which approaches a suprenum norm as t -> Inf

    # It turns out it is equivalent to a tempered LogSumExp loss, 1/t * log(sum(exp.(t .* x)))
    # where we assume xⱼ = ln(βⱼ)

    # Because we do exponentiation with large values, we use logspace arithmetic
    # Also, don't look - it's ugly

    βⱼ = beta!(ws, p)
    βⱼ .*= time_horizon

    logz = log(norm(βⱼ, t))
    βⱼ .= (log.(βⱼ) .- logz) .* (t - 1)

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

    for k in eachindex(ids)
        dB[ids[k]] += exp(β + log(max(values[k], 1e-16)))
    end
end

function gradient_descent_barrier_iteration!(ws::GradientDescentWorkspace, state, regions, p, q, lr; time_horizon)
    ivi_value_assignment!(ws, regions, p, q)
    gradient!(ws, p; time_horizon=time_horizon)

    project_gradient!(ws, lr)

    # Grad norm clipping
    # This allows us to take bigger step sizes without worrying about overstepping
    # norm_grad = norm(ws.dB)
    # if norm_grad > 0.05
    #     rmul!(ws.dB, 0.05 / norm_grad)
    # end

    Optimisers.adjust!(state, lr)
    state, ws.B = Optimisers.update!(state, ws.B, ws.dB)

    project!(ws)

    return state
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

function setup_workspace(alg::StochasticGradientDescentAlgorithm, P̅ᵤ, initial_indices, unsafe_indices)
    return StochasticGradientDescentWorkspace(P̅ᵤ, alg.subsampling_fraction, initial_indices, unsafe_indices)
end

mutable struct StochasticGradientDescentWorkspace{T, BT<:AbstractVector{T}, VT<:AbstractVector{T}, RT<:AbstractVector{T}}
    B::BT
    dB::BT
    β::BT  # This only functions as a cache. It should never be accessed directly. 
    grad_β::BT
    index_β::Vector{Int64}

    B_init::VT
    B_unsafe::VT
    B_regions::RT
    dB_regions::RT
end

function StochasticGradientDescentWorkspace(n::Integer, subsampling_fraction, initial_indices::AbstractVector, unsafe_indices::AbstractVector)
    B = fill(0.5, n + 1)
    dB = similar(B)
    β = zeros(n)
    grad_β = zeros(ceil(Int64, n * subsampling_fraction))
    index_β = Vector{Int64}(undef, length(grad_β))

    # The sinking state is considered unsafe
    push!(unsafe_indices, n + 1)

    B_init = @view(B[initial_indices])
    B_unsafe = @view(B[unsafe_indices])
    B_regions = @view(B[1:end - 1])
    dB_regions = @view(dB[1:end - 1])

    return StochasticGradientDescentWorkspace(B, dB, β, grad_β, index_β, B_init, B_unsafe, B_regions, dB_regions)
end

function StochasticGradientDescentWorkspace(P̅ᵤ::AbstractVector, subsampling_fraction, initial_indices::AbstractVector, unsafe_indices::AbstractVector)
    n = length(P̅ᵤ)

    ws = StochasticGradientDescentWorkspace(n, subsampling_fraction, initial_indices, unsafe_indices)
    ws.B_regions .= P̅ᵤ

    return ws
end

num_regions(ws::StochasticGradientDescentWorkspace) = length(ws.B_regions)

function project!(ws::StochasticGradientDescentWorkspace)
    # Projection onto [0, 1]^n x {1}
    clamp!(ws.B, 0, 1)
    ws.B_init .= 0
    ws.B_unsafe .= 1
end

function project_gradient!(ws::StochasticGradientDescentWorkspace, lr)
    # Projection so that B(k + 1) ∈ [0, 1]^n
    rmul!(ws.dB, -lr)
    ws.dB .+= ws.B
    clamp!(ws.dB, 0, 1)
    ws.dB .-= ws.B
    rmul!(ws.dB, -1/lr)
end

function sample_regions!(ws::StochasticGradientDescentWorkspace)
    StatsBase.seqsample_a!(1:num_regions(ws), ws.index_β)
end

function gradient_beta!(ws::StochasticGradientDescentWorkspace, p)
    Threads.@threads for i in eachindex(ws.index_β)
        j = ws.index_β[i]
        ws.grad_β[i] = dot(ws.B, p[j])
    end

    # ws.β .= dot.(tuple(ws.B), p)
    ws.grad_β .-= @view(ws.B_regions[ws.index_β])
    clamp!(ws.grad_β, 0, Inf)

    return ws.grad_β
end

function gradient!(ws::StochasticGradientDescentWorkspace, p::VVT; time_horizon, t=20.0) where {VVT<:AbstractVector{<:AbstractVector}}
    # Gradient for the following loss: ||βⱼ||ₜ
    # This is an Lp-norm, which approaches a suprenum norm as t -> Inf

    # It turns out it is equivalent to a tempered LogSumExp loss, 1/t * log(sum(exp.(t .* x)))
    # where we assume xⱼ = ln(βⱼ)

    # Because we do exponentiation with large values, we use logspace arithmetic
    # Also, don't look - it's ugly

    βⱼ = gradient_beta!(ws, p)
    βⱼ .*= time_horizon

    logz = log(norm(βⱼ, t))
    βⱼ .= (log.(βⱼ) .- logz) .* (t - 1)

    ws.dB .= 0  # We need to do this explicitly because of view in the next line
    @view(ws.dB_regions[ws.index_β]) .= (-).(exp.(βⱼ))
    for i in eachindex(ws.index_β)
        j = ws.index_β[i]

        logspace_add_prod!(ws.dB, βⱼ[i], p[j])
    end

    return ws.dB
end

function gradient_descent_barrier_iteration!(ws::StochasticGradientDescentWorkspace, state, regions, p, q, lr; time_horizon)
    sample_regions!(ws)
    subsample_ivi_value_assignment!(ws, regions, p, q)
    gradient!(ws, p; time_horizon=time_horizon)

    # Grad norm clipping
    # This allows us to take bigger step sizes without worrying about overstepping
    # norm_grad = norm(ws.dB)
    # if norm_grad > 0.05
    #     rmul!(ws.dB, 0.05 / norm_grad)
    # end

    project_gradient!(ws, lr)

    Optimisers.adjust!(state, lr)
    state, ws.B = Optimisers.update!(state, ws.B, ws.dB)

    project!(ws)

    return state
end

function subsample_ivi_value_assignment!(ws, regions, p::VVT, q) where {VVT<:AbstractVector{<:AbstractVector}}
    sortperm!(q, ws.B, rev=true)
        
    Threads.@threads for j in ws.index_β
        @inbounds ivi_prob!(p[j], regions[j], q)
    end
end

function subsample_ivi_value_assignment!(ws, regions, p::VVT, q) where {VVT<:AbstractVector{<:AbstractSparseVector}}
    q, q_order, q_subsets = q
    sortperm!(q, ws.B, rev=true)
    populate_subsets!(q, q_order, q_subsets)
        
    Threads.@threads for j in ws.index_β
        @inbounds ivi_prob!(p[j], regions[j], q_subsets[j].items)
    end
end

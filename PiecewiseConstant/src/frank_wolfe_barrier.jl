""" Piecewise barrier function construction

    © Rayan Mazouz

"""

# Optimization function
function frank_wolfe_barrier(regions::Vector{<:RegionWithProbabilities}, initial_region::LazySet, obstacle_region::LazySet; time_horizon=1, ϵ=1e-6)
    n = length(regions)

    # Construct linear minization oracles
    barrier_oracle = BarrierOracle{Float64}()
    transition_prob_oracle = FrankWolfe.ProbabilitySimplexOracle{Float64}()
    # TODO: Add transition bound oracle

    direction_indices = [[1:n]; [(n + (i - 1) * (n + 1) + 1):(n + i * (n + 1)) for i in 1:n]]
    lmo = ProductOracle([[barrier_oracle]; [transition_prob_oracle for _ in 1:n]...], direction_indices)

    # Set-up loss function and gradient
    η_indices = findall(X -> !isdisjoint(initial_region, region(X)), regions)

    function loss(x)
        b = x[1:n]
        p = reshape(x[n + 1:end], n + 1, n)
        p, pᵤ = p[1:end - 1, :], p[end, :]

        exp = vec(reshape(b, 1, :) * p)

        βⱼ = @. max(exp + pᵤ - b, 0)

        β = maximum(βⱼ)
        η = maximum(b[η_indices])

        return η + β * time_horizon
    end

    function grad!(storage, x)
        println(x)
        g = gradient(loss, x)
        println(g)

        storage .= g[1]

        return storage
    end

    # Initial point
    d₀ = collect(ones(n + (n + 1) * n))
    x₀ = FrankWolfe.compute_extreme_point(lmo, d₀)

    # Gradient (for structure)
    ∇x₀ = collect(ones(n + (n + 1) * n))
    grad!(∇x₀, x₀)

    x_lmo, v, primal, dual_gap, trajectory_lmo = FrankWolfe.frank_wolfe(
        loss, grad!, lmo, x₀; 
        max_iteration=100, 
        line_search=FrankWolfe.Shortstep(2.0),
        print_iter=100 / 10,
        memory_mode=FrankWolfe.OutplaceEmphasis(),
        verbose=true
    )

    B = x_lmo[direction_indices[1]]
    η = maximum(B[η_indices])

    βⱼ = map(enumerate(direction_indices[2:end])) do (j, idx)
        pⱼ = x_lmo[idx]
        p = @view pⱼ[1:end - 1]
        pᵤ = pⱼ[end]

        return max(dot(B, p) + pᵤ - B[j], 0)
    end

    @info "Solution Frank-Wolfe" η β=maximum(βⱼ)

    return B, βⱼ
end

struct BarrierOracle{T} <: FrankWolfe.LinearMinimizationOracle end

function FrankWolfe.compute_extreme_point(lmo::BarrierOracle, direction;
    kwargs...)

    v = zero(direction)
    for i in eachindex(direction)
        if direction[i] < 0
            v[i] = 1
        end
    end
    return v
end

struct ProductOracle <: FrankWolfe.LinearMinimizationOracle
    lmos::Vector{FrankWolfe.LinearMinimizationOracle}
    indices::Vector{AbstractArray}
end

function FrankWolfe.compute_extreme_point(
    lmo::ProductOracle,
    direction::AbstractArray;
    storage=similar(direction),
    kwargs...
)
    for (l, indices) in zip(lmo.lmos, lmo.indices)
        storage[indices] .= compute_extreme_point(l, direction[indices]; kwargs...)
    end
    return storage
end

# struct ScaledBoundLInfNormBall{T,N,VT1<:AbstractArray{T,N},VT2<:AbstractArray{T,N}} <:
#     LinearMinimizationOracle
#  lower_bounds::VT1
#  upper_bounds::VT2
# end

# function compute_extreme_point(
#  lmo::ScaledBoundLInfNormBall,
#  direction;
#  v=similar(lmo.lower_bounds),
#  kwargs...,
# )
#  copyto!(v, lmo.lower_bounds)
#  for i in eachindex(direction)
#      if direction[i] * lmo.upper_bounds[i] < direction[i] * lmo.lower_bounds[i]
#          v[i] = lmo.upper_bounds[i]
#      end
#  end
#  return v
# end
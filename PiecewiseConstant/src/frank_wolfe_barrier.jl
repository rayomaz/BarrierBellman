""" Piecewise barrier function construction

    © Rayan Mazouz

"""

# Optimization function
function frank_wolfe_barrier(regions::Vector{<:RegionWithProbabilities}, initial_region::LazySet, obstacle_region::LazySet; time_horizon=1, ϵ=1e-6)
    n = length(regions)

    # Construct linear minization oracles
    barrier_oracle = BarrierOracle{Float64}()

    # Set-up loss function and gradient
    η_indices = findall(X -> !isdisjoint(initial_region, region(X)), regions)
    cache = zeros(n + 1)

    function loss(b)
        b_ext = [b; [1]]
        q = sortperm(b_ext, rev=true)

        β = maximum(1:n) do j
            p = ivi_prob!(cache, regions[j], q)
            exp = dot(p, b_ext)
    
            βⱼ = max(exp - b[j], 0)
            return βⱼ
        end
        η = maximum(b[η_indices])

        return η + β * time_horizon
    end

    function grad!(storage, b)
        b_ext = [b; [1]]
        q = sortperm(b_ext, rev=true)

        βⱼ, j = findmax(1:n) do j
            p = ivi_prob!(cache, regions[j], q)
            exp = dot(p, b_ext)
    
            return exp - b[j]
        end

        if βⱼ > 0
            pⱼ = ivi_prob(regions[j], q)
            storage .= pⱼ[1:end - 1]
            storage[j] += -1
        else
            fill!(storage, 0)
        end

        j = η_indices[argmax(b[η_indices])]
        storage[j] += 1

        return storage
    end

    # Initial point
    d₀ = collect(-ones(n))
    x₀ = FrankWolfe.compute_extreme_point(barrier_oracle, d₀)

    x_lmo, v, primal, dual_gap, trajectory_lmo = FrankWolfe.frank_wolfe(
        loss, grad!, barrier_oracle, x₀;
        max_iteration=10000,
        momentum=0.8,
        line_search=FrankWolfe.Agnostic(),
        print_iter=100,
        memory_mode=FrankWolfe.InplaceEmphasis(),
        verbose=true
    )

    B = x_lmo
    η = maximum(B[η_indices])

    
    b_ext = [B; [1]]
    q = sortperm(b_ext, rev=true)

    p = ivi_prob.(regions, tuple(q))
    exp = dot.(p, tuple(b_ext))

    βⱼ = @. max(exp - B, 0)

    @info "Solution Frank-Wolfe" η β = maximum(βⱼ)

    return B, βⱼ
end

struct BarrierOracle{T} <: FrankWolfe.LinearMinimizationOracle end

function FrankWolfe.compute_extreme_point(lmo::BarrierOracle, direction;
    v=similar(direction),
    kwargs...)

    fill!(v, 0)
    @inbounds for (i, d) in enumerate(direction)
        if d < 0
            v[i] = 1
        end
    end
    return v
end

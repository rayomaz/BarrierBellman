""" Piecewise barrier function construction

    © Rayan Mazouz

"""

# Optimization function
function gradient_descent_barrier(regions::Vector{<:RegionWithProbabilities}, initial_region::LazySet, obstacle_region::LazySet; time_horizon=1, ϵ=1e-6)
    n = length(regions)

    initial_indices = findall(X -> !isdisjoint(initial_region, region(X)), regions)
    unsafe_indices = findall(X -> !isdisjoint(obstacle_region, region(X)), regions)
    push!(unsafe_indices, n + 1)

    B = fill(0.5, n + 1)
    B_init = @view(B[initial_indices])
    B_unsafe = @view(B[unsafe_indices])
    B_regions = @view(B[1:end - 1])
    dB = similar(B)

    B[initial_indices] .= 0
    B[unsafe_indices] .= 1
    
    # decay = Step(λ = 1e-1, γ = 0.995, step_sizes = 10)
    # optim = Optimisers.Adam(1e-2)

    decay = Exp(λ = 1e-1, γ = 0.9995)
    optim = Optimisers.Adam(1e-1, (8e-1, 9.99e-1))

    state = Optimisers.setup(optim, B)

    p = [zeros(n + 1) for j in 1:n]
    q = collect(1:n + 1)
    βⱼ = zeros(n)

    for t in 1:10000
        fill!(dB, 0)
        sortperm!(q, B, rev=true)

        for j in 1:n
            ivi_prob!(p[j], regions[j], q)
        end
        βⱼ .= dot.(tuple(B), p)
        βⱼ .-= B_regions
        j = argmax(βⱼ)

        # if t % 100 == 99
        #     println(βⱼ[j])
        # end

        copyto!(dB, B)
        dB .-= decay(t) .* p[j] #ivi_prob!(p[j], regions[j], q)
        dB[j] += decay(t) * 1
        clamp!(dB, 0, 1)
        dB .*= -1
        dB .+= B
        dB ./= decay(t)

        Optimisers.adjust!(state, decay(t))
        state, B = Optimisers.update!(state, B, dB)

        # Projection onto [0, 1]^n x {1}
        clamp!(B, 0, 1)
        B_init .= 0
        B_unsafe .= 1
    end

    η = maximum(B_init)

    sortperm!(q, B, rev=true)
    for j in 1:n
        ivi_prob!(p[j], regions[j], q)
    end
    βⱼ .= dot.(tuple(B), p)
    βⱼ .-= B_regions
    clamp!(βⱼ, 0, Inf)

    B = B[1:end - 1]

    @info "Solution Gradient Descent" η β = maximum(βⱼ)

    return B_regions, βⱼ
end

# This function bould be useful as a template for parallelizing onto the gpu.
function β_direct(B, X::RegionWithProbabilities, p::AbstractVector{<:Int})    
    remaining = 1 - X.sum_lower
    s = dot(B, X.lower)
    
    for i in p
        @inbounds s += min(remaining, X.gap[i]) * B[i] 
        @inbounds remaining -= X.gap[i]

        if remaining < 0
            break
        end
    end

    return s
end
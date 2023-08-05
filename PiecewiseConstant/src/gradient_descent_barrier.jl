""" Piecewise barrier function construction

    © Rayan Mazouz

"""

# Optimization function
function gradient_descent_barrier(regions::Vector{<:RegionWithProbabilities}, initial_region::LazySet, obstacle_region::LazySet; time_horizon=1, ϵ=1e-6)
    n = length(regions)

    B = ones(n + 1)
    dB = similar(B)
    decay = Exp(λ = 1e-1, γ = 0.999)
    optim = Optimisers.Adam(1e-1, (8e-1, 9.9e-1))
    state = Optimisers.setup(optim, B)

    initial_indices = findall(X -> !isdisjoint(initial_region, region(X)), regions)
    B[initial_indices] .= 0

    unsafe_indices = findall(X -> !isdisjoint(obstacle_region, region(X)), regions)
    push!(unsafe_indices, n + 1)

    p = zeros(n + 1, n)
    q = collect(1:n + 1)
    βⱼ = zeros(n)

    for t in 1:10000
        fill!(dB, 0)
        sortperm!(q, B, rev=true)

        ivi_prob!.(eachcol(p), regions, tuple(q))
        βⱼ .= (B' * p)' .- B[1:end - 1] 
        j = argmax(βⱼ)

        dB .= p[:, j]
        dB[j] += -1

        Optimisers.adjust!(state, decay(t))
        state, B = Optimisers.update!(state, B, dB)

        # Projection onto [0, 1]^n x {1}
        clamp!(B, 0, 1)
        B[initial_indices] .= 0
        B[unsafe_indices] .= 1
    end

    η = maximum(B[initial_indices])

    sortperm!(q, B, rev=true)
    ivi_prob!.(eachcol(p), regions, tuple(q))
    βⱼ .= dot.(eachcol(p), tuple(B)) .- B[1:end - 1]
    clamp!(βⱼ, 0, Inf)

    B = B[1:end - 1]

    @info "Solution Gradient Descent" η β = maximum(βⱼ)

    return B, βⱼ
end

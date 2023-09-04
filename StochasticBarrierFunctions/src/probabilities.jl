""" Functions to compute :

    Transition probability bounds P̲ᵢⱼ ≤ P(f(x) ∈ qᵢ | x ∈ qⱼ) ≤ P̅ᵢⱼ for Linear Systems and Neural Network Dynamic Models

    © Rayan Mazouz, Frederik Baymler Mathiesen

"""
abstract type AbstractLowerBoundAlgorithm end
struct VertexEnumeration <: AbstractLowerBoundAlgorithm end

abstract type AbstractUpperBoundAlgorithm end
Base.@kwdef struct GradientDescent <: AbstractUpperBoundAlgorithm
    non_linear_solver = default_non_linear_solver()
end
struct BoxApproximation <: AbstractUpperBoundAlgorithm end

Base.@kwdef struct TransitionProbabilityAlgorithm
    lower_bound_method::AbstractLowerBoundAlgorithm = VertexEnumeration()
    upper_bound_method::AbstractUpperBoundAlgorithm = GradientDescent()
    sparisty_ϵ = 1e-12
    log_ϵ = 1e-16
end

transition_probabilities(system::AdditiveGaussianUncertainPWASystem; kwargs...) = transition_probabilities(system, regions(system); kwargs...)
function transition_probabilities(system, Xs; alg=TransitionProbabilityAlgorithm())
    # Construct barriers
    @info "Computing transition probabilities"

    safe_set = Hyperrectangle(low=minimum(low.(Xs)), high=maximum(high.(Xs)))
    # TODO: Check if ⋃Xs = state_space

    # Anything beyond `μ ± σ * nσ_search` will always have a probability < sparisty_ϵ
    nσ_search = -quantile(Normal(), alg.sparisty_ϵ)

    # Size definition
    number_hypercubes = length(Xs)

    # Pre-allocate probability matrices
    P̲ = Vector{SparseVector{Float64, Int32}}(undef, number_hypercubes)
    P̅ = Vector{SparseVector{Float64, Int32}}(undef, number_hypercubes)

    # Generate
    Threads.@threads for jj in eachindex(Xs)
        P̲ⱼ, P̅ⱼ = transition_prob_from_region(system, (jj, Xs[jj]), Xs, safe_set, alg; nσ_search=nσ_search)

        P̲[jj] = P̲ⱼ
        P̅[jj] = P̅ⱼ
    end

    # Combine into a single matrix
    P̲ = reduce(sparse_hcat, P̲)
    P̅ = reduce(sparse_hcat, P̅)

    density = (nnz(P̅) + nnz(P̲)) / (length(P̅) + length(P̲))
    @info "Density of the probability matrix" density sparsity=1-density

    axlist = (Dim{:to}(1:number_hypercubes + 1), Dim{:from}(1:number_hypercubes))
    P̲, P̅ = DimArray(P̲, axlist), DimArray(P̅, axlist)

    # Return as a YAXArrays dataset
    return create_sparse_probability_dataset(Xs, P̲, P̅)
end

function post(system::AdditiveGaussianLinearSystem, Xind)
    (jj, X) = Xind

    X = convert(VPolytope, X)

    # Compute post(qᵢ, f(x)) for all qⱼ ∈ Q
    A, b = dynamics(system)

    VY = affine_map(A, X, b)
    HY = convert(HPolytope, VY)
    box_Y = box_approximation(VY)

    return VY, HY, box_Y
end

function post(system::AdditiveGaussianUncertainPWASystem, Xind)
    (jj, X) = Xind

    # Compute post(qᵢ, f(x)) for all qⱼ ∈ Q    
    (Xprime, dyn) = dynamics(system)[jj]
    # This is a necessary but not suffiecient condition.
    # The complete condition is that the intersection of the two regions has a non-zero measure.
    @assert !isdisjoint(X, Xprime)

    X = convert(VPolytope, X)
    VY = VPolytope(mapreduce(vcat, dyn) do (A, b)
        vertices_list(affine_map(A, X, b))
    end)

    HY = convert(HPolytope, VY)
    box_Y = box_approximation(VY)

    return VY, HY, box_Y
end

# Transition probability P̲ᵢⱼ ≤ P(f(x) ∈ qᵢ | x ∈ qⱼ) ≤ P̅ᵢⱼ based on proposition 1, http://dx.doi.org/10.1145/3302504.3311805
function transition_prob_from_region(system, Xⱼ, Xs, safe_set, alg; nσ_search)
    VY, HY, box_Y = post(system, Xⱼ)

    # Fetch noise
    n = length(Xs)
    σ = noise_distribution(system)

    # Search for overlap with box(f(Xⱼ)) + σ * nσ_search as 
    # any region beyond that will always have a probability < sparisty_ϵ
    query_set = minkowski_sum(box_Y, Hyperrectangle(zero(σ), σ * nσ_search)) 

    indices = findall(X -> !isdisjoint(X, query_set), Xs)

    P̲ⱼ = zeros(Float64, length(indices))
    P̅ⱼ = zeros(Float64, length(indices))

    for (i, Xᵢ) in enumerate(@view(Xs[indices]))    
        # Obtain min and max of T(qᵢ | x) over Y
        P̲ᵢⱼ, P̅ᵢⱼ = transition_prob_to_region(system, VY, HY, box_Y, Xᵢ, alg)
        
        P̲ⱼ[i] = P̲ᵢⱼ
        P̅ⱼ[i] = P̅ᵢⱼ
    end

    # Prune regions with P(f(x) ∈ qᵢ | x ∈ qⱼ) < sparisty_ϵ
    keep_indices = findall(p -> p >= alg.sparisty_ϵ, P̅ⱼ)
    P̲ⱼ = P̲ⱼ[keep_indices]
    P̅ⱼ = P̅ⱼ[keep_indices]
    indices = indices[keep_indices]

    # Compute P(f(x) ∈ qᵤ | x ∈ qⱼ) including sparsity pruning
    P̲ₛⱼ, P̅ₛⱼ = transition_prob_to_region(system, VY, HY, box_Y, safe_set, alg)
    Psparse = (n - length(indices)) * alg.sparisty_ϵ
    P̲ᵤⱼ, P̅ᵤⱼ = (1 - P̅ₛⱼ), (1 - P̲ₛⱼ) + Psparse
    
    # If you ever hit this case, then you are in trouble. Either sparisty_ϵ
    # is too high for the amount of regions, or the system is inherently unsafe.
    @assert P̅ᵤⱼ <= 1.0

    P̲ⱼ = SparseVector(n + 1, [indices; [n + 1]], [P̲ⱼ; [P̲ᵤⱼ]])
    P̅ⱼ = SparseVector(n + 1, [indices; [n + 1]], [P̅ⱼ; [P̅ᵤⱼ]])

    return P̲ⱼ, P̅ⱼ
end

# Transition probability P̲ᵢⱼ ≤ P(f(x) ∈ qᵢ | x ∈ qⱼ) ≤ P̅ᵢⱼ based on proposition 1, http://dx.doi.org/10.1145/3302504.3311805
function transition_prob_to_region(system, VY, HY, box_Y, Xᵢ, alg)
    vₗ = low(Xᵢ)
    vₕ = high(Xᵢ)
    v = LazySets.center(Xᵢ)

    # Fetch noise
    m = dimensionality(system)
    σ = noise_distribution(system)
    
    # Transition kernel T(qᵢ | x)
    erf_lower(y, i) = erf((y[i] - vₗ[i]) / (σ[i] * sqrt(2)))
    erf_upper(y, i) = erf((y[i] - vₕ[i]) / (σ[i] * sqrt(2)))

    T(y) = (1 / 2^m) * prod(i -> erf_lower(y, i) - erf_upper(y, i), 1:m)
    # clamp to ϵ is added to avoid log(0).
    logT(y) = log(1) - m * log(2) + sum(i -> log(max(erf_lower(y, i) - erf_upper(y, i), alg.log_ϵ)), 1:m)

    # Obtain min of T(qᵢ | x) over Y
    prob_transition_lower = min_log_concave_over_polytope(alg.lower_bound_method, T, VY)

    # Obtain max of T(qᵢ | x) over Y
    prob_transition_upper = exp(max_concave_over_polytope(alg.upper_bound_method, logT, v, HY, box_Y))

    return prob_transition_lower, prob_transition_upper
end

function min_log_concave_over_polytope(::VertexEnumeration, f, X)
    vertices = vertices_list(X)

    return minimum(f, vertices)
end

function max_log_concave_over_polytope(::BoxApproximation, f, global_max, X, box_X)
    if global_max in X
        return f(global_max)
    end

    l, h = low(box_X), high(box_X)
    x_max = @. min(h, max(global_max, l))

    return f(x_max)
end

function max_log_concave_over_polytope(alg::GradientDescent, f, global_max, X, box_X)
    if global_max in X
        return f(global_max)
    end

    m = LazySets.dim(X)
    fsplat(y...) = f(y)

    model = Model(alg.non_linear_solver)
    set_silent(model)
    register(model, :fsplat, m, fsplat; autodiff = true)

    @variable(model, x[1:m])

    H, h = tosimplehrep(X)
    @constraint(model, H * x <= h)

    @NLobjective(model, Max, fsplat(x...))

    # Optimize for maximum
    JuMP.optimize!(model)

    return JuMP.objective_value(model)
end

function max_concave_over_polytope(::BoxApproximation, f, global_max, X, box_X)
    if global_max in X
        return f(global_max)
    end

    l, h = low(box_X), high(box_X)
    x_max = @. min(h, max(global_max, l))

    return f(x_max)
end

function max_concave_over_polytope(alg::GradientDescent, f, global_max, X, box_X)
    if global_max in X
        return f(global_max)
    end

    m = LazySets.dim(X)
    fsplat(y...) = f(y)

    model = Model(alg.non_linear_solver)
    set_silent(model)
    register(model, :fsplat, m, fsplat; autodiff = true)

    @variable(model, x[1:m])

    H, h = tosimplehrep(X)
    @constraint(model, H * x <= h)

    @NLobjective(model, Max, fsplat(x...))

    # Optimize for maximum
    JuMP.optimize!(model)

    return JuMP.objective_value(model)
end

plot_posterior(system::AdditiveGaussianUncertainPWASystem; kwargs...) = plot_posterior(system, regions(system); kwargs...)
function plot_posterior(system, Xs; figname_prefix="")
    pwa_dynamics = dynamics(system)

    VYs = map(pwa_dynamics) do (X, dyn)
        X = convert(VPolytope, X)

        Y = map(dyn) do (A, b)
            affine_map(A, X, b)
        end
        return Y
    end

    postXs = post.(tuple(system), zip(eachindex(Xs), Xs))

    box_Ys = map(postXs) do (VY, HY, box_Y)
        box_Y
    end

    for (i, (X, Y, box_Y)) in enumerate(zip(Xs[1:10], VYs, box_Ys))
        p = plot(X, color=:blue, alpha=0.2, xlim=(-deg2rad(15), deg2rad(15)), ylim=(-1.2, 1.2), size=(1200, 800))
        plot!(p, Y[1], color=:red, alpha=0.2)
        plot!(p, Y[2], color=:red, alpha=0.2)
        plot!(p, box_Y, color=:green, alpha=0.2)

        savefig(p, figname_prefix * "posterior_$i.png")
    end
end

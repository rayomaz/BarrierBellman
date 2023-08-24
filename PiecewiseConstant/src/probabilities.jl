""" Functions to compute :

    Transition probability bounds P̲ᵢⱼ ≤ P(f(x) ∈ qᵢ | x ∈ qⱼ) ≤ P̅ᵢⱼ for Linear Systems and Neural Network Dynamic Models

    © Rayan Mazouz, Frederik Baymler Mathiesen

"""

transition_probabilities(system::AdditiveGaussianUncertainPWASystem) = transition_probabilities(system, regions(system))
function transition_probabilities(system, Xs)

    # Construct barriers
    @info "Computing transition probabilities"

    # Size definition
    number_hypercubes = length(Xs)

    # Compute post(qⱼ, f(x)) for all qⱼ ∈ Q
    VYs, HYs, box_Ys = post(system, Xs)

    # Pre-allocate probability matrices
    P̲ = zeros(number_hypercubes, number_hypercubes)
    P̅ = zeros(number_hypercubes, number_hypercubes)

    # Generate
    Threads.@threads for ii in eachindex(Xs)
        P̲ᵢ, P̅ᵢ = transition_prob_to_region(system, VYs, HYs, box_Ys, Xs[ii])

        P̲[ii, :] = P̲ᵢ
        P̅[ii, :] = P̅ᵢ
    end

    Xₛ = Hyperrectangle(low=minimum(low.(Xs)), high=maximum(high.(Xs)))

    P̲ₛ, P̅ₛ  = transition_prob_to_region(system, VYs, HYs, box_Ys, Xₛ)
    P̲ᵤ, P̅ᵤ = (1 .- P̅ₛ), (1 .- P̲ₛ)

    axlist = (Dim{:to}(1:number_hypercubes), Dim{:from}(1:number_hypercubes))
    P̲, P̅ = DimArray(P̲, axlist), DimArray(P̅, axlist)
    
    axlist = (Dim{:from}(1:number_hypercubes),)
    P̲ᵤ, P̅ᵤ = DimArray(P̲ᵤ, axlist), DimArray(P̅ᵤ, axlist)

    # Return as a YAXArrays dataset
    return create_probability_dataset(Xs, P̲, P̅, P̲ᵤ, P̅ᵤ)
end

function post(system::AdditiveGaussianLinearSystem, Xs)
    # Compute post(qᵢ, f(x)) for all qⱼ ∈ Q
    A, b = dynamics(system)
    f(x) = A * x + b

    Xs = convert.(VPolytope, Xs)
    VYs = f.(Xs)
    HYs = convert.(HPolytope, VYs)
    box_Ys = box_approximation.(VYs)

    return VYs, HYs, box_Ys
end

function post(system::AdditiveGaussianUncertainPWASystem, Xs)
    # Input Xs is also contained in dynamics(system) since _piece-wise_ affine.

    # Compute post(qᵢ, f(x)) for all qⱼ ∈ Q    
    pwa_dynamics = dynamics(system)

    VYs = map(pwa_dynamics) do (X, dyn)
        X = convert(VPolytope, X)

        vertices = mapreduce(vcat, dyn) do (A, b)
            vertices_list(A * X + b)
        end
        return VPolytope(vertices)
    end
    HYs = convert.(HPolytope, VYs)
    box_Ys = box_approximation.(VYs)

    return VYs, HYs, box_Ys
end

# Transition probability P̲ᵢⱼ ≤ P(f(x) ∈ qᵢ | x ∈ qⱼ) ≤ P̅ᵢⱼ based on proposition 1, http://dx.doi.org/10.1145/3302504.3311805
function transition_prob_to_region(system, VYs, HYs, box_Ys, Xᵢ; gradient_descent = false)
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
    Tsplat(y...) = T(y)
    logT(y...) = log(1) - m * log(2) + sum(i -> log(erf_lower(y, i) - erf_upper(y, i)), 1:m)

    # Obtain min of T(qᵢ | x) over Ys
    prob_transition_lower = map(VYs) do Y
        vertices = vertices_list(Y)

        P_min = minimum(T, vertices)
        return P_min
    end

    # Obtain max of T(qᵢ | x) over Ys
    prob_transition_upper = map(zip(HYs, box_Ys)) do (Y, box_Y)
        if v in Y
            return T(v)
        end

        if gradient_descent == true

            model = Model(Ipopt.Optimizer)
            set_silent(model)
            register(model, :logT, m, logT; autodiff = true)
            register(model, :Tsplat, m, Tsplat; autodiff = true)

            @variable(model, y[1:m])

            H, h = tosimplehrep(Y)
            @constraint(model, H * y <= h)

            @NLobjective(model, Max, Tsplat(y...))

            # Optimize for maximum
            JuMP.optimize!(model)
            P_max = JuMP.objective_value(model)

        elseif gradient_descent == false

           # Steven's box_approximation method
            l, h = low(box_Y), high(box_Y)
            y_max = @. min(h, max(v, l))
            P_max = T(y_max)
        end

        return P_max
    end

    return prob_transition_lower, prob_transition_upper
end

plot_posterior(system::AdditiveGaussianUncertainPWASystem; kwargs...) = plot_posterior(system, regions(system); kwargs...)
function plot_posterior(system, Xs; figname_prefix="")
    pwa_dynamics = dynamics(system)

    VYs = map(pwa_dynamics) do (X, dyn)
        X = convert(VPolytope, X)

        Y = map(dyn) do (A, b)
            A * X + b
        end
        return Y
    end

    _, HYs, box_Ys = post(system, Xs)

    for (i, (X, Y, box_Y)) in enumerate(zip(Xs[1:10], VYs, box_Ys))
        p = plot(X, color=:blue, alpha=0.2, xlim=(-deg2rad(15), deg2rad(15)), ylim=(-1.2, 1.2), size=(1200, 800))
        plot!(p, Y[1], color=:red, alpha=0.2)
        plot!(p, Y[2], color=:red, alpha=0.2)
        plot!(p, box_Y, color=:green, alpha=0.2)

        savefig(p, figname_prefix * "posterior_$i.png")
    end
end
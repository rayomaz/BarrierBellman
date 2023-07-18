""" Functions to compute :

    Transition probability bounds P̲ᵢⱼ ≤ P(f(x) ∈ qᵢ | x ∈ qⱼ) ≤ P̅ᵢⱼ for Linear Systems and Neural Network Dynamic Models

    © Rayan Mazouz, Frederik Baymler Mathiesen

"""

function linear_transition_probabilities(system, Xs)

    # Construct barriers
    # @info "Computing transition probabilities"

    # Size definition
    number_hypercubes = length(Xs)

    # Compute post(qⱼ, f(x)) for all qⱼ ∈ Q
    f = MP.coefficients.(dynamics(system))
    Ys = f .* Xs
    box_Ys = box_approximation.(Ys)

    # Pre-allocate probability matrices
    P̲ = zeros(number_hypercubes, number_hypercubes)
    P̅ = zeros(number_hypercubes, number_hypercubes)

    # Generate
    Threads.@threads for ii in eachindex(Xs)
        P̲ᵢ, P̅ᵢ = linear_transition_prob_to_region(system, Ys, box_Ys, Xs[ii])

        P̲[ii, :] = P̲ᵢ
        P̅[ii, :] = P̅ᵢ
    end

    Xₛ = Hyperrectangle(low=minimum(low.(Xs)), high=maximum(high.(Xs)))

    P̲ₛ, P̅ₛ  = linear_transition_prob_to_region(system, Ys, box_Ys, Xₛ)
    P̲ᵤ, P̅ᵤ = (1 .- P̅ₛ), (1 .- P̲ₛ)

    # Return as a tuple
    return P̲, P̅, P̲ᵤ, P̅ᵤ
end

# Transition probability P̲ᵢⱼ ≤ P(f(x) ∈ qᵢ | x ∈ qⱼ) ≤ P̅ᵢⱼ based on proposition 1, http://dx.doi.org/10.1145/3302504.3311805
function linear_transition_prob_to_region(system, Ys, box_Ys, Xᵢ)
    vₗ = low(Xᵢ)
    vₕ = high(Xᵢ)
    v = center(Xᵢ)

    # Fetch noise
    m = dimensionality(system)
    σ = noise_distribution(system)
    
    # Transition kernel T(qᵢ | x)
    erf_lower(y, i) = erf((y[i] - vₗ[i]) / (σ[i] * sqrt(2)))
    erf_upper(y, i) = erf((y[i] - vₕ[i]) / (σ[1] * sqrt(2)))
    T(y) = (1 / 2^m) * prod(i -> erf_lower(y, i) - erf_upper(y, i), 1:m)

    # Obtain min of T(qᵢ | x) over Ys
    prob_transition_lower = map(Ys) do Y
        vertices = vertices_list(Y)

        P_min = minimum(T, vertices)
        return P_min
    end

    # Obtain max of T(qᵢ | x) over Ys
    prob_transition_upper = map(box_Ys) do Y
        if v in Y
            return T(v)
        end

        l, h = low(Y), high(Y)

        y_max = @. min(h, max(v, l))

        P_max = T(y_max)
        return P_max
    end

    return prob_transition_lower, prob_transition_upper
end

function neural_transition_probabilities(file, number_hypercubes, σ)

    # Extract hypercube data (avoid using float64 for precision issues)
    state_partitions = read(file, "partitions")

    # Extract Neural Network Bounds [CROWN]
    M_upper = read(file, "M_h")
    M_lower = read(file, "M_l")
    b_upper = read(file, "B_h")
    b_lower = read(file, "B_l")

    # Construct barriers
    println("Computing transition probabilities ... ")

    # Pre-generate probability matrices (parallel computation)
    matrix_prob_lower = zeros(number_hypercubes, number_hypercubes)
    matrix_prob_upper = zeros(number_hypercubes, number_hypercubes)
    matrix_prob_unsafe_lower = zeros(1, number_hypercubes)
    matrix_prob_unsafe_upper = zeros(1, number_hypercubes)

    for jj = 1:number_hypercubes

        """ Probability bounds
            - P(j → i)
            - P(j → Xᵤ)
        """
        neural_bounds = (transpose(M_upper[jj, :, :]), 
                         transpose(M_lower[jj, :, :]),
                         b_upper[jj,:], 
                         b_lower[jj,:])

        prob_lower, prob_upper = neural_probability_distribution(neural_bounds::Tuple{LinearAlgebra.Transpose{Float64, Matrix{Float64}}, LinearAlgebra.Transpose{Float64, Matrix{Float64}}, Vector{Float64}, Vector{Float64}},  
                                                                    σ::Float64, state_partitions:: Array{Float64, 3}, jj::Int64, "transition_j_to_i"::String)
        prob_unsafe_lower, prob_unsafe_upper = neural_probability_distribution(neural_bounds::Tuple{LinearAlgebra.Transpose{Float64, Matrix{Float64}}, LinearAlgebra.Transpose{Float64, Matrix{Float64}}, Vector{Float64}, Vector{Float64}},  
                                                                    σ::Float64, state_partitions:: Array{Float64, 3}, jj::Int64, "transition_unsafe"::String)

        # Build matrices in parallel format
        matrix_prob_lower[jj, :] = prob_lower
        matrix_prob_upper[jj, :] = prob_upper
        matrix_prob_unsafe_lower[jj] = prob_unsafe_lower
        matrix_prob_unsafe_upper[jj] = prob_unsafe_upper

    end

    # Save probability values in tuple
    prob_bounds = (matrix_prob_lower, 
                   matrix_prob_upper,
                   matrix_prob_unsafe_lower,
                   matrix_prob_unsafe_upper)

    return prob_bounds
end

# Transition probability, P(qᵢ | x ∈ qⱼ), based on proposition 1, http://dx.doi.org/10.1145/3302504.3311805
function neural_probability_distribution(neural_bounds::Tuple{LinearAlgebra.Transpose{Float64, Matrix{Float64}}, LinearAlgebra.Transpose{Float64, Matrix{Float64}}, Vector{Float64}, Vector{Float64}},
                                  σ::Float64, state_partitions::Array{Float64, 3}, jj::Int64, type::String)

    # Hypercube bounds
    hypercubeⱼ = state_partitions[jj, :, :]
    x_lower = hypercubeⱼ[1:2:end]
    x_upper = hypercubeⱼ[2:2:end]

    if type == "transition_j_to_i"

        number_hypercubes = Int(size(state_partitions)[1])
        prob_transition_lower = zeros(1, number_hypercubes)
        prob_transition_upper = zeros(1, number_hypercubes)

        for ii = 1:number_hypercubes

            # Hypercube bounds
            hypercubeᵢ = state_partitions[ii, :, :]
            v_l = hypercubeᵢ[1:2:end]
            v_u = hypercubeᵢ[2:2:end]

            # Obtain bounds on P
            P_min, P_max = neural_optimize_prod_of_erf(neural_bounds::Tuple{LinearAlgebra.Transpose{Float64, Matrix{Float64}}, LinearAlgebra.Transpose{Float64, Matrix{Float64}}, Vector{Float64}, Vector{Float64}},
                                                        σ::Float64, v_l::Vector{Float64}, v_u::Vector{Float64}, 
                                                        x_lower::Vector{Float64}, x_upper::Vector{Float64}, type::String)
            prob_transition_lower[ii] = P_min
            prob_transition_upper[ii] = P_max

        end

    return prob_transition_lower, prob_transition_upper

    elseif type == "transition_unsafe"

        # State space bounds
        hypercube₁ = state_partitions[1, :, :]
        hypercubeₑ = state_partitions[end, :, :]

        v_l = hypercube₁[1:2:end]
        v_u = hypercubeₑ[2:2:end]

        P_min, P_max = neural_optimize_prod_of_erf(neural_bounds::Tuple{LinearAlgebra.Transpose{Float64, Matrix{Float64}}, LinearAlgebra.Transpose{Float64, Matrix{Float64}}, Vector{Float64}, Vector{Float64}},
                                                    σ::Float64, v_l::Vector{Float64}, v_u::Vector{Float64}, 
                                                    x_lower::Vector{Float64}, x_upper::Vector{Float64}, type::String)

        return P_min, P_max

    end

end

function neural_optimize_prod_of_erf(neural_bounds::Tuple{LinearAlgebra.Transpose{Float64, Matrix{Float64}}, LinearAlgebra.Transpose{Float64, Matrix{Float64}}, Vector{Float64}, Vector{Float64}},
                                     σ::Float64, v_l::Vector{Float64}, v_u::Vector{Float64}, 
                                     x_lower::Vector{Float64}, x_upper::Vector{Float64}, type::String)

    # Bounds on Neural Network Dynamic Model (tuple)
    (M_upper, M_lower, b_upper, b_lower) = neural_bounds

    # Fetch state space matrix size
    m = length(b_upper)

    # Gradient descent on log-concave function:
    model_gradient = Model(Ipopt.Optimizer)
    set_silent(model_gradient)
    @variable(model_gradient, x[1:m])

    # Add constraints to original hypercube dimensions
    @constraint(model_gradient, x .>= x_lower)
    @constraint(model_gradient, x .<= x_upper)

    # Loop for f(y, q), Proposition 3, http://dx.doi.org/10.1145/3302504.3311805
    Kσ =  σ * sqrt(2)
    vector_erf_vars = []
    P_min_vector = []
    P_max_vector = []

    for kk = 1:2

        if kk == 1
            M_matrix = M_lower
            b_vector = b_lower
        elseif kk == 2
            M_matrix = M_upper
            b_vector = b_upper
        end

        # Generate Δ erf
        for jj = 1:m

            # Transform system dynamics
            system_dynamics = dot(M_matrix[jj, :], x[1:m]) + b_vector[jj]

            # Function per dimension
            func_lower = (1 / Kσ) * (system_dynamics - v_l[jj])
            func_upper = (1 / Kσ) * (system_dynamics - v_u[jj])

            vector_erf_vars = push!(vector_erf_vars, [func_lower, func_upper])

        end

        P_min, P_max = gradient_descent(model_gradient, m, vector_erf_vars, type)

        push!(P_min_vector, P_min)
        push!(P_max_vector, P_max)

    end

    return minimum(P_min_vector), maximum(P_max_vector)

end

function gradient_descent(model_gradient, m, vector_erf_vars, type)

    # Objective function to be minimized
    if type == "transition_j_to_i"
        @NLobjective(model_gradient, Min, (1/(2^m))*prod(erf(vector_erf_vars[kk][1]) - erf(vector_erf_vars[kk][2]) for kk in 1:m))
    elseif type == "transition_unsafe"
        @NLobjective(model_gradient, Min, 1 - (1/(2^m))*prod(erf(vector_erf_vars[kk][1]) - erf(vector_erf_vars[kk][2]) for kk in 1:m))
    end

    # Optimize for minimum
    JuMP.optimize!(model_gradient)
    P_min = JuMP.objective_value(model_gradient)

    # Objective function to be maximized
    if type == "transition_j_to_i"
        @NLobjective(model_gradient, Max, (1/(2^m))*prod(erf(vector_erf_vars[kk][1]) - erf(vector_erf_vars[kk][2]) for kk in 1:m))
    elseif type == "transition_unsafe"
        @NLobjective(model_gradient, Max, 1 - (1/(2^m))*prod(erf(vector_erf_vars[kk][1]) - erf(vector_erf_vars[kk][2]) for kk in 1:m))
    end

    # Optimize for maximum
    JuMP.optimize!(model_gradient)
    P_max = JuMP.objective_value(model_gradient)

    return P_min, P_max

end
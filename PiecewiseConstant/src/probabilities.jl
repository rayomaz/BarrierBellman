""" Functions to compute :

    Transition probabilities P(qᵢ | x ∈ qⱼ) for Linear Systems and Neural Network Models

    © Rayan Mazouz

"""

function linear_transition_probabilities(system, state_partitions)

    # Construct barriers
    println("Computing transition probabilities ... ")

    # Size definition
    number_hypercubes = length(state_partitions)

    # Pre-generate probability matrices (parallel computation)
    matrix_prob_lower = zeros(number_hypercubes, number_hypercubes)
    matrix_prob_upper = zeros(number_hypercubes, number_hypercubes)
    matrix_prob_unsafe_lower = zeros(number_hypercubes)
    matrix_prob_unsafe_upper = zeros(number_hypercubes)

    for jj = eachindex(state_partitions)

        """ Probability bounds
            - P(j → i)
            - P(j → Xᵤ)
        """

        prob_lower, prob_upper = linear_probability_distribution(system, state_partitions, jj, "transition_j_to_i")
        prob_unsafe_lower, prob_unsafe_upper = linear_probability_distribution(system, state_partitions, jj, "transition_unsafe")

        # Build matrices in parallel format
        matrix_prob_lower[:, jj] = prob_lower
        matrix_prob_upper[:, jj] = prob_upper
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
function linear_probability_distribution(system, state_partitions, jj, type)

    # Identify current hypercube
    Xⱼ = state_partitions[jj]
    x_lower = low(Xⱼ)
    x_upper = high(Xⱼ)
    x_initial = center(Xⱼ)

    if type == "transition_j_to_i"

        hyper = length(state_partitions)
        prob_transition_lower = zeros(hyper)
        prob_transition_upper = zeros(hyper)

        for ii in eachindex(state_partitions)

            # hynercube bounds
            Xᵢ = state_partitions[ii]
            v_l = low(Xᵢ)
            v_u = high(Xᵢ)

            P_min, P_max = linear_optimize_prod_of_erf(system, v_l, v_u, x_lower, x_upper, x_initial)

            prob_transition_lower[ii] = P_min
            prob_transition_upper[ii] = P_max

        end

        return prob_transition_lower, prob_transition_upper

    elseif type == "transition_unsafe"
        v_l = minimum(low.(state_partitions))
        v_u = maximum(high.(state_partitions))

        P_min, P_max = linear_optimize_prod_of_erf(system, v_l, v_u, x_lower, x_upper, x_initial)

        # Convert to transition unsafe set
        return (1 - P_max), (1 - P_min)
    end
end

function linear_optimize_prod_of_erf(system, v_l, v_u, x_lower, x_upper, x_initial)

    # Fetch dynamics and noise
    f = dynamics(system)
    σ = noise_distribution(system)

    # Loop for f(y, q), Proposition 3, http://dx.doi.org/10.1145/3302504.3311805
    m = dimensionality(system)

    # Gradient descent on log-concave function: 
    inner_optimizer = GradientDescent()

    # print("Note: needs to be automated! Double check dynamics here")
    erf_lower(y, i) = erf((y[i] - v_l[i]) / (σ[i] * sqrt(2)))
    erf_upper(y, i) = erf((y[i] - v_u[i]) / (σ[i] * sqrt(2)))

    g(x) = let y = map(fᵢ -> fᵢ(x), f)
        (1 / 2^m) * prod(i -> erf_lower(y, i) - erf_upper(y, i), 1:m)
    end
    h(x) = -g(x)

    # Obtain min-max on P
    y_lower = map(fᵢ -> fᵢ(x_lower), f)
    y_upper = map(fᵢ -> fᵢ(x_upper), f)
    erf_low = map(i -> erf_lower(y_lower, i) - erf_upper(y_lower, i), 1:m)
    erf_high = map(i -> erf_lower(y_upper, i) - erf_upper(y_upper, i), 1:m)
    erf_min = min.(erf_low, erf_high)

    P_min = (1 / 2^m) * prod(erf_min)

    # Gradient descent to find max
    results_max = Optim.optimize(h, x_lower, x_upper, x_initial, Fminbox(inner_optimizer))
    P_max = -results_max.minimum

    return P_min, P_max
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
""" Piecewise barrier function construction

    © Rayan Mazouz

"""

# Optimization function
function constant_barrier(system, state_partitions, initial_state_partition)

    # Using GLPK as the LP solver
    optimizer = optimizer_with_attributes(GLPK.Optimizer)
    model = Model(optimizer)

    # Hyperspace
    number_state_hypercubes = length(state_partitions)

    # Create optimization variables
    @variable(model, b[1:number_state_hypercubes])    
    @constraint(model, b .>= ϵ)
    @constraint(model, b .<= 1 - ϵ)

    # Create probability decision variables β
    @variable(model, ϵ <= β_parts_var[1:number_state_hypercubes] <= 1 - ϵ)
    @variable(model, β)
    @constraint(model, β_parts_var .<= β)

    # Construct barriers
    for (jj, ~) in enumerate(state_partitions)

        """ Probability bounds
            - P(j → i)
            - P(j → Xᵤ)
        """
        prob_lower, prob_upper = probability_distribution(system, state_partitions, jj, "transition_j_to_i")
        prob_unsafe_lower, prob_unsafe_upper = probability_distribution(system, state_partitions, jj, "transition_unsafe")
        probability_bounds = [prob_upper, prob_unsafe_upper]

        # Martingale condition
        expectation_constraint_centralized!(model, b, jj, probability_bounds, β_parts_var[jj])

    end

    # Define optimization objective
    time_horizon = 1
    η = b[initial_state_partition]
    @objective(model, Min, η + β * time_horizon)

    println("Objective made")

    # Optimize model
    optimize!(model)

    # Barrier certificate
    for jj in 1:number_state_hypercubes
        certificate = value.(b[jj])
        println(certificate)
    end

    # Print optimal values
    β_values = value.(β_parts_var)
    max_β = maximum(β_values)
    η = value.(b[initial_state_partition])
    println("Solution: [η = $(value(η)), β = $max_β]")

    println("")
    println(solution_summary(model))

    return value.(b), β_values

end

function expectation_constraint_centralized!(model, b, jj, probability_bounds, βⱼ) 

    """ Barrier martingale condition
    * ∑B[f(x)]*p(x) + Pᵤ <= B(x) + β: expanded in summations
    """

    # Construct piecewise martingale constraint
    martingale = 0

    (prob_upper, prob_unsafe) = probability_bounds

    # Barrier jth partition
    Bⱼ = b[jj]

    # Bounds on Eij
    for ii in eachindex(b)

        # Martingale
        martingale -= b[ii] * prob_upper[ii]

    end

    # Transition to unsafe set
    martingale -= prob_unsafe

    # Constraint martingale
    martingale_condition_multivariate = martingale + Bⱼ + βⱼ
    @constraint(model, martingale_condition_multivariate >= 0)

end


# Transition probability, P(qᵢ | x ∈ qⱼ), based on proposition 1, http://dx.doi.org/10.1145/3302504.3311805
function probability_distribution(system, state_partitions, jj, type)

    # Identify current hypercube
    hypercubeⱼ = state_partitions[jj]
    x_lower = low(hypercubeⱼ)
    x_upper = high(hypercubeⱼ)
    x_initial = 1/2 * (x_lower + x_upper)

    if type == "transition_j_to_i"

        hyper = length(state_partitions)
        prob_transition_lower = zeros(1, hyper);
        prob_transition_upper = zeros(1, hyper);

        for ii = 1:hyper

            # Hypercube bounds
            hypercubeᵢ = state_partitions[ii]
            v_l = low(hypercubeᵢ)
            v_u = high(hypercubeᵢ)

            P_min, P_max = optimize_prod_of_erf(system, v_l, v_u, x_lower, x_upper, x_initial)

            prob_transition_lower[ii] = P_min
            prob_transition_upper[ii] = P_max

        end

        return prob_transition_lower, prob_transition_upper

    elseif type == "transition_unsafe"

        v_l = low(state_partitions[1])
        v_u = high(state_partitions[end])

        P_min, P_max = optimize_prod_of_erf(system, v_l, v_u, x_lower, x_upper, x_initial)

        # Convert to transition unsafe set
        return (1 - P_max), (1 - P_min)

    end

end

function optimize_prod_of_erf(system, v_l, v_u, x_lower, x_upper, x_initial)

    # Fetch state space, noise and dynamics
    σ = noise_distribution(system)
    fx = dynamics(system)

    # Loop for f(y, q), Proposition 3, http://dx.doi.org/10.1145/3302504.3311805
    m = length(fx)
    if m > 1
        error("Specify dynamics ... ")
    end

    # Gradient descent on log-concave function: 
    inner_optimizer = GradientDescent()

    erf_low(x) = (0.95*x[1] - v_l[m]) / (σ[1] * sqrt(2))
    erf_up(x)  = (0.95*x[1] - v_u[m]) / (σ[1] * sqrt(2))

    f(x) = 1/(2^m)*(erf(erf_low(x)) - erf(erf_up(x)))
    g(x) = -f(x)

    # Obtain min-max on P
    results_min = Optim.optimize(f, x_lower, x_upper, x_initial, Fminbox(inner_optimizer))
    P_min = results_min.minimum

    results_max = Optim.optimize(g, x_lower, x_upper, x_initial, Fminbox(inner_optimizer))
    P_max = -results_max.minimum

    return P_min, P_max

end
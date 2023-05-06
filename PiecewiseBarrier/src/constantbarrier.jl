""" Piecewise barrier function construction

    © Rayan Mazouz

"""

# Optimization function
function constant_barrier(bounds, initial_state_partition)
    
    # Using GLPK as the LP solver
    optimizer = optimizer_with_attributes(GLPK.Optimizer)
    model = Model(optimizer)

    # Hyperspace
    lower_partitions = read(bounds, "lower_partition")
    upper_partitions = read(bounds, "upper_partition")
    state_partitions = hcat(lower_partitions, upper_partitions)
    state_partitions = [Hyperrectangle(low=[low], high=[high]) for (low, high) in eachrow(state_partitions)]
    number_state_hypercubes = length(state_partitions)

    # Create optimization variables
    @variable(model, b[1:number_state_hypercubes])    
    @constraint(model, b .>= ϵ)
    @constraint(model, b .<= 1 - ϵ)

    # Create probability decision variables β
    @variable(model, ϵ <= β_parts_var[1:number_state_hypercubes] <= 1 - ϵ)
    @variable(model, β)
    @constraint(model, β_parts_var .<= β)

    """ Probability bounds
    """
    prob_upper = read(bounds, "prob_transition_upper")
    prob_unsafe = read(bounds, "prob_unsafe_upper")

    # Construct barriers
    for (jj, ~) in enumerate(state_partitions)

        probability_bounds = [prob_upper[jj, :], 
                              prob_unsafe[jj]]

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
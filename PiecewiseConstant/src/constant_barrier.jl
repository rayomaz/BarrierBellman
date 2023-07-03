""" Piecewise barrier function construction

    © Rayan Mazouz

"""

function constant_barrier(probabilities::MatlabFile, obstacle)
    # Load probability matrices
    prob_upper = read(probabilities, "matrix_prob_upper")
    prob_unsafe_upper = read(probabilities, "matrix_prob_unsafe_upper")

    return constant_barrier(prob_upper, prob_unsafe_upper, obstacle)
end

# Optimization function
function constant_barrier(prob_upper, prob_unsafe_upper, obstacle; ϵ=1e-6)
    
    # Number of hypercubes
    number_hypercubes = length(prob_unsafe_upper)

    # Number of hypercubes
    initial_state_partition = Int(round(number_hypercubes/2))

    # Using HiGHS as the LP solver
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # Create optimization variables
    @variable(model, b[1:number_hypercubes] >= ϵ)    

    # Obstacle barrier
    # @constraint(model, b[obstacle] == 1)

    # Create probability decision variables β
    @variable(model, ϵ <= β_parts_var[1:number_hypercubes] <= 1 - ϵ)
    @variable(model, β)
    @constraint(model, β_parts_var .<= β)

    # Construct barriers
    @inbounds for jj in eachindex(b)
        probability_bounds = [prob_upper[jj, :], prob_unsafe_upper[jj]]
        expectation_constraint!(model, b, jj, probability_bounds, β_parts_var[jj])
    end

    # println("Synthesizing barries ... ")

    # Define optimization objective
    time_horizon = 1
    η = b[initial_state_partition]
    @objective(model, Min, η + β * time_horizon)

    # println("Objective made ... ")

    # Optimize model
    JuMP.optimize!(model)

    # Barrier certificate
    b = value.(b)
    # for Bⱼ in b
    #     println(Bⱼ)
    # end

    # Print optimal values
    β_values = value.(β_parts_var)
    max_β = maximum(β_values)
    η = value.(b[initial_state_partition])
    println("Solution: [η = $(value(η)), β = $max_β]")

    # Print model summary and number of constraints
    # println("")
    # println(" Number of constraints ", sum(num_constraints(model, F, S) for (F, S) in list_of_constraint_types(model)))
    # println("")

    return b, β_values

end

function expectation_constraint!(model, b, jj, probability_bounds, βⱼ) 

    """ Barrier martingale condition
    * ∑B[f(x)]*p(x) + Pᵤ <= B(x) + β: expanded in summations
    """

    (prob_upper, prob_unsafe) = probability_bounds

    # Construct piecewise martingale constraint
    # Transition to unsafe set
    exp = AffExpr(0)

    # Barrier jth partition
    Bⱼ = b[jj]

    # Bounds on Eᵢⱼ
    @inbounds for (Bᵢ, P̅ᵢ) in zip(b, prob_upper)
        # Martingale
        add_to_expression!(exp, P̅ᵢ, Bᵢ)
    end

    # Constraint martingale
    @constraint(model, exp + prob_unsafe <= Bⱼ + βⱼ)
end


""" Piecewise barrier function construction

    © Rayan Mazouz

"""

function post_compute_beta(b, probabilities::MatlabFile)
    # Bounds
    prob_lower = read(probabilities, "matrix_prob_lower")
    prob_upper = read(probabilities, "matrix_prob_upper")
    prob_unsafe_lower = read(probabilities, "matrix_prob_unsafe_lower")
    prob_unsafe_upper = read(probabilities, "matrix_prob_unsafe_upper")

    return post_compute_beta(b, prob_lower, prob_upper, prob_unsafe_lower, prob_unsafe_upper)
end

function post_compute_beta(b, prob_lower, prob_upper, prob_unsafe_lower, prob_unsafe_upper; ϵ=1e-6)
    # Using HiGHS as the LP solver
    optimizer = optimizer_with_attributes(HiGHS.Optimizer)
    model = Model(optimizer)

    # Create optimization variables
    number_hypercubes = length(b)
    @variable(model, p[1:number_hypercubes, 1:number_hypercubes]) 
    @variable(model, Pᵤ[1:number_hypercubes])    

    # Create probability decision variables β
    @variable(model, ϵ <= β_parts_var[1:number_hypercubes] <= 1 - ϵ)
    @variable(model, β)
    @constraint(model, β_parts_var .<= β)
    
    for jj in eachindex(b)

        # Establish accuracy
        val_low, val_up = accuracy_threshold(prob_unsafe_lower[jj], prob_unsafe_upper[jj])

        # Constraint Pᵤ
        @constraint(model, val_low <= Pᵤ[jj] <= val_up)

        # Constraint ∑i=1 →k pᵢ + Pᵤ == 1
        @constraint(model, sum(p[jj, :]) + Pᵤ[jj] == 1)

        # Setup martingale (-∑i=1 →k bᵢ⋅pᵢ - Pᵤ + bⱼ + βⱼ ≥ 0)
        martingale = @expression(model, b[jj] + β_parts_var[jj] - Pᵤ[jj])

        for ii in eachindex(b)
            # Establish accuracy
            val_low, val_up = accuracy_threshold(prob_lower[jj, ii], prob_upper[jj, ii])

            # Constraint Pⱼ → Pᵢ (Plower ≤ Pᵢ ≤ Pupper)
            @constraint(model, val_low <= p[jj, ii] <= val_up)
                
            add_to_expression!(martingale, -p[jj, ii], b[ii])
        end

        @constraint(model, martingale == 0)
    end

    # Define optimization objective
    @objective(model, Max, β)

    println("Objective made ... ")

    # Optimize model
    JuMP.optimize!(model)

    # Print optimal values
    β_values = value.(β_parts_var)
    β_values = abs.(β_values)
    max_β = maximum(β_values)
    println("Solution: [β = $max_β]")

    # Print model summary and number of constraints
    # println("")
    # println(solution_summary(model))
    # println("")
    # println(" Number of constraints ", sum(num_constraints(model, F, S) for (F, S) in list_of_constraint_types(model)))
    # println("")

    return β_values

end

function accuracy_threshold(val_low, val_up)

    if val_up < val_low
        val_up = val_low
    end

    return val_low, val_up
end
""" Piecewise barrier function construction

    © Rayan Mazouz

"""

function post_compute_beta(b, probabilities)

    # Using HiGHS as the LP solver
    optimizer = optimizer_with_attributes(HiGHS.Optimizer)
    model = Model(optimizer)

    # Create optimization variables
    number_hypercubes = length(b)
    @variable(model, p[1:number_hypercubes, 1:number_hypercubes]) 
    @variable(model, Pᵤ[1:number_hypercubes])    

    # Create probability decision variables β
    ϵ = 1e-6
    @variable(model, ϵ <= β_parts_var[1:number_hypercubes] <= 1 - ϵ)
    @variable(model, β)
    @constraint(model, β_parts_var .<= β)
    
    # Bounds
    matrix_prob_lower = read(probabilities, "matrix_prob_lower")
    matrix_prob_upper = read(probabilities, "matrix_prob_upper")
    matrix_prob_unsafe_lower = read(probabilities, "matrix_prob_unsafe_lower")
    matrix_prob_unsafe_upper = read(probabilities, "matrix_prob_unsafe_upper")
    
    for jj in eachindex(b)

        # Establish accuracy
        val_low, val_up = accuracy_threshold(matrix_prob_unsafe_lower[jj], matrix_prob_unsafe_upper[jj])

        # Constraint Pᵤ
        @constraint(model, val_low <= Pᵤ[jj] <= val_up)
        add_constraint_to_model!(model, Pᵤ[jj], val_low, val_up)

        # Constraint ∑i=1 →k pᵢ + Pᵤ == 1
        @constraint(model, sum(p[jj, :]) + Pᵤ[jj] == 1)

        # Setup martingale
        martingale = 0 

        for ii in eachindex(b)

                # Establish accuracy
                val_low, val_up = accuracy_threshold(matrix_prob_lower[jj, ii], matrix_prob_upper[jj, ii])

                # Constraint Pⱼ → Pᵢ (Plower ≤ Pᵢ ≤ Pupper)
                add_constraint_to_model!(model, p[jj, ii], val_low, val_up)
                 
                # Martingale (-∑i=1 →k bᵢ⋅pᵢ - Pᵤ + bⱼ + βⱼ ≥ 0)
                martingale -= b[ii] * p[jj, ii]
                martingale -= Pᵤ[jj]
                martingale += b[jj] + β_parts_var[jj]

                @constraint(model, martingale == 0)

        end

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
    println("")
    println(" Number of constraints ", sum(num_constraints(model, F, S) for (F, S) in list_of_constraint_types(model)))
    println("")

    return β_values

end

function add_constraint_to_model!(model, var, val_low, val_up)

    @constraint(model, val_low <= var <= val_up)

end

function accuracy_threshold(val_low, val_up)

    if val_up < val_low
        val_up = val_low
    end

    return val_low, val_up
end
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
    number_hypercubes = length(b)

    β_parts = Vector{Float64}(undef, number_hypercubes)
    p_distribution = Matrix{Float64}(undef, number_hypercubes, number_hypercubes + 1)
    
    for jj in eachindex(b)
        # Using HiGHS as the LP solver
        model = Model(HiGHS.Optimizer)
        set_silent(model)

        # Create optimization variables
        number_hypercubes = length(b)
        @variable(model, p[1:number_hypercubes], lower_bound=0, upper_bound=1) 
        @variable(model, Pᵤ, lower_bound=0, upper_bound=1)    

        # Create probability decision variables β
        @variable(model, β)

        # Establish accuracy
        val_low, val_up = accuracy_threshold(prob_unsafe_lower[jj], prob_unsafe_upper[jj])

        # Constraint Pᵤ
        @constraint(model, val_low <= Pᵤ <= val_up)

        # Constraint ∑i=1 →k pᵢ + Pᵤ == 1
        @constraint(model, sum(p) + Pᵤ == 1)

        # Setup expectation (-∑i=1 →k bᵢ⋅pᵢ - Pᵤ + bⱼ + βⱼ ≥ 0)
        exp = AffExpr(0)

        @inbounds for ii in eachindex(b)
            # Establish accuracy
            val_low, val_up = accuracy_threshold(prob_lower[jj, ii], prob_upper[jj, ii])

            # Constraint Pⱼ → Pᵢ (Plower ≤ Pᵢ ≤ Pupper)
            @constraint(model, val_low <= p[ii] <= val_up)
                
            add_to_expression!(exp, b[ii], p[ii])
        end

        @constraint(model, exp + Pᵤ == b[jj] + β)

        # Define optimization objective
        @objective(model, Max, β)
    
        # Optimize model
        JuMP.optimize!(model)
    
        # Print optimal values
        @inbounds β_parts[jj] = max(value(β), 0)
        # @inbounds p_values[jj, :] = p_val

        p_values = value.(p)
        push!(p_values, value(Pᵤ))

        p_distribution[jj, :] = p_values

    end

    max_β = maximum(β_parts)
   
    println("Solution updated beta: [β = $max_β]")

    # Print beta values to txt file
    # if isfile("probabilities/beta_updated.txt") == true
    #     rm("probabilities/beta_updated.txt")
    # end

    # open("probabilities/beta_updated.txt", "a") do io
    #     println(io, β_parts)
    # end

    return β_parts, p_distribution
end

function post_compute_beta_centralized(b, probabilities::MatlabFile)
    # Bounds
    prob_lower = read(probabilities, "matrix_prob_lower")
    prob_upper = read(probabilities, "matrix_prob_upper")
    prob_unsafe_lower = read(probabilities, "matrix_prob_unsafe_lower")
    prob_unsafe_upper = read(probabilities, "matrix_prob_unsafe_upper")

    return post_compute_beta(b, prob_lower, prob_upper, prob_unsafe_lower, prob_unsafe_upper)
end

function post_compute_beta_centralized(b, prob_lower, prob_upper, prob_unsafe_lower, prob_unsafe_upper; ϵ=1e-6)
    number_hypercubes = length(b)

    # Using HiGHS as the LP solver
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    
    # Create optimization variables
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

        # Setup expectation (-∑i=1 →k bᵢ⋅pᵢ - Pᵤ + bⱼ + βⱼ ≥ 0)
        exp = AffExpr(0)

        @inbounds for ii in eachindex(b)
            # Establish accuracy
            val_low, val_up = accuracy_threshold(prob_lower[jj, ii], prob_upper[jj, ii])

            # Constraint Pⱼ → Pᵢ (Plower ≤ Pᵢ ≤ Pupper)
            @constraint(model, val_low <= p[jj, ii] <= val_up)
                
            add_to_expression!(exp, b[ii], p[jj, ii])
        end

        @constraint(model, exp + Pᵤ == b[jj] + β)

    end

    # Define optimization objective
    @objective(model, Max, β)

    # Optimize model
    JuMP.optimize!(model)

    # Print optimal values
    β_values = value.(β_parts_var)
    β_values = abs.(β_values)
    max_β = maximum(β_values)

    p_distribution = value.(p)
    push!(p_distribution, value(Pᵤ))

    println("Solution updated beta: [β = $max_β]")

    # Print beta values to txt file
    # if isfile("probabilities/beta_updated.txt") == true
    #     rm("probabilities/beta_updated.txt")
    # end

    # open("probabilities/beta_updated.txt", "a") do io
    #     println(io, β_parts)
    # end

    return β_parts, p_distribution
end

function accuracy_threshold(val_low, val_up)

    if val_up < val_low
        val_up = val_low
    end

    return val_low, val_up
end
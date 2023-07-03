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

    Threads.@threads for jj in eachindex(b)
        # Using HiGHS as the LP solver
        model = Model(HiGHS.Optimizer)
        set_silent(model)

        # Create optimization variables
        number_hypercubes = length(b)
        @variable(model, p[1:number_hypercubes]) 
        @variable(model, Pᵤ)    

        # Create probability decision variables β
        @variable(model, β)

        # Establish accuracy
        val_low, val_up = accuracy_threshold(prob_unsafe_lower[jj], prob_unsafe_upper[jj])

        # Constraint Pᵤ
        @constraint(model, val_low <= Pᵤ <= val_up)

        # Constraint ∑i=1 →k pᵢ + Pᵤ == 1
        @constraint(model, sum(p) + Pᵤ == 1)

        # Setup martingale (-∑i=1 →k bᵢ⋅pᵢ - Pᵤ + bⱼ + βⱼ ≥ 0)
        martingale = @expression(model, b[jj] + β - Pᵤ)

        @inbounds for ii in eachindex(b)
            # Establish accuracy
            val_low, val_up = accuracy_threshold(prob_lower[jj, ii], prob_upper[jj, ii])

            # Constraint Pⱼ → Pᵢ (Plower ≤ Pᵢ ≤ Pupper)
            @constraint(model, val_low <= p[ii] <= val_up)
                
            add_to_expression!(martingale, -b[ii], p[ii])
        end

        @constraint(model, martingale == 0)

        # Define optimization objective
        @objective(model, Min, β)
    
        # Optimize model
        JuMP.optimize!(model)
    
        # Print optimal values
        @inbounds β_parts[jj] = value(β)
    end

    max_β = maximum(β_parts)
    # println("Solution: [β = $max_β]")

    return β_parts

end

function accuracy_threshold(val_low, val_up)

    if val_up < val_low
        val_up = val_low
    end

    return val_low, val_up
end
""" Piecewise barrier function construction

    © Rayan Mazouz

"""

function post_compute_beta(B, regions::Vector{<:RegionWithProbabilities}; ϵ=1e-6)
    β_parts = Vector{Float64}(undef, length(B))
    p_distribution = Matrix{Float64}(undef, length(B), length(B) + 1)

    Threads.@threads for jj in eachindex(regions)
        Xⱼ, Bⱼ = regions[jj], B[jj]

        # Using HiGHS as the LP solver
        model = Model(HiGHS.Optimizer)
        set_silent(model)

        # Create optimization variables
        @variable(model, P[eachindex(regions)], lower_bound=0, upper_bound=1) 
        @variable(model, Pᵤ, lower_bound=0, upper_bound=1)    

        # Create probability decision variables β
        @variable(model, β)

        # Establish accuracy
        P̲ᵤ, P̅ᵤ = prob_unsafe_lower(Xⱼ), prob_unsafe_upper(Xⱼ)
        val_low, val_up = accuracy_threshold(P̲ᵤ, P̅ᵤ)

        # Constraint Pᵤ
        @constraint(model, val_low <= Pᵤ <= val_up)

        # Constraint ∑i=1 →k pᵢ + Pᵤ == 1
        @constraint(model, sum(P) + Pᵤ == 1)

        # Setup expectation (-∑i=1 →k bᵢ⋅pᵢ - Pᵤ + bⱼ + βⱼ ≥ 0)
        exp = AffExpr(0)

        P̲, P̅ = prob_lower(Xⱼ), prob_upper(Xⱼ)
        @inbounds for (Bᵢ, P̲ᵢ, P̅ᵢ, Pᵢ) in zip(B, P̲, P̅, P)
            # Establish accuracy
            val_low, val_up = accuracy_threshold(P̲ᵢ, P̅ᵢ)

            # Constraint Pⱼ → Pᵢ (Plower ≤ Pᵢ ≤ Pupper)
            @constraint(model, val_low <= Pᵢ <= val_up)
                
            add_to_expression!(exp, Bᵢ, Pᵢ)
        end

        @constraint(model, exp + Pᵤ == Bⱼ + β)

        # Define optimization objective
        @objective(model, Max, β)
    
        # Optimize model
        JuMP.optimize!(model)
    
        # Print optimal values
        @inbounds β_parts[jj] = max(value(β), 0)
        # @inbounds p_values[jj, :] = p_val

        p_values = [value.(P); [value(Pᵤ)]]
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

function accuracy_threshold(val_low, val_up)

    if val_up < val_low
        val_up = val_low
    end

    return val_low, val_up
end
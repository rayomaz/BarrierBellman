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
        # Since we are bounding the variables by constants, we can do so on creation
        P̲, P̅ = prob_lower(Xⱼ), prob_upper(Xⱼ)
        val_low, val_up = min.(P̲, P̅), max.(P̲, P̅)
        @variable(model, val_low[ii] <= P[ii=eachindex(regions)] <= val_up[ii]) 

        P̲ᵤ, P̅ᵤ = prob_unsafe_lower(Xⱼ), prob_unsafe_upper(Xⱼ)
        val_low, val_up = min(P̲ᵤ, P̅ᵤ), max(P̲ᵤ, P̅ᵤ)
        @variable(model, Pᵤ, lower_bound=val_low, upper_bound=val_up)    

        # Create probability decision variables β
        @variable(model, β)

        # Constraint ∑i=1 →k pᵢ + Pᵤ == 1
        @constraint(model, sum(P) + Pᵤ == 1)

        # Setup expectation (∑i=1→k Bᵢ⋅Pᵢ + Pᵤ ≤ Bⱼ + βⱼ)
        @constraint(model, dot(B, P) + Pᵤ == Bⱼ + β)

        # Define optimization objective
        @objective(model, Max, β)
    
        # Optimize model
        JuMP.optimize!(model)
    
        # Print optimal values
        @inbounds β_parts[jj] = max(value(β), ϵ)

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

function accelerated_post_compute_beta(B, regions::Vector{<:RegionWithProbabilities})
    # Don't ask. It's not pretty... But it's fast!

    β_parts = Vector{Float64}(undef, length(B))

    Threads.@threads for jj in eachindex(regions)
        Xⱼ, Bⱼ = regions[jj], B[jj]

        model = get!(task_local_storage(), "post_compute_model") do
            # Using HiGHS as the LP solver
            model = Model(HiGHS.Optimizer)
            set_silent(model)

            # Create optimization variables
            @variable(model, P[ii=eachindex(regions)], lower_bound=0, upper_bound=1) 
            @variable(model, Pᵤ, lower_bound=0, upper_bound=1)

            # Constraint ∑i=1 →k pᵢ + Pᵤ == 1
            @constraint(model, sum(P) + Pᵤ == 1)

            # Define optimization objective
            @objective(model, Max, dot(B, P) + Pᵤ)

            return model
        end

        P̲, P̅ = prob_lower(Xⱼ), prob_upper(Xⱼ)
        val_low, val_up = min.(P̲, P̅), max.(P̲, P̅)

        P = model[:P]
        set_lower_bound.(P, val_low)
        set_upper_bound.(P, val_up)

        P̲ᵤ, P̅ᵤ = prob_unsafe_lower(Xⱼ), prob_unsafe_upper(Xⱼ)
        val_low, val_up = min(P̲ᵤ, P̅ᵤ), max(P̲ᵤ, P̅ᵤ)

        Pᵤ = model[:Pᵤ]
        set_lower_bound(Pᵤ, val_low)
        set_upper_bound(Pᵤ, val_up)

        # Optimize model
        JuMP.optimize!(model)
    
        # Print optimal values
        @inbounds β_parts[jj] = max(objective_value(model) - Bⱼ, 0)
    end

    max_β = maximum(β_parts)
   
    println("Solution updated beta: [β = $max_β]")

    return β_parts
end

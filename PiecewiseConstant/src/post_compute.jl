""" Piecewise barrier function construction

    © Rayan Mazouz

"""

function post_compute_beta(B, regions::Vector{<:RegionWithProbabilities})
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
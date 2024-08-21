function compute_beta(linear_solver, B, regions::Vector{<:RegionWithProbabilities{T}}) where {T}
    # Don't ask. It's not pretty... But it's fast!

    β_parts = Vector{T}(undef, length(B))
    p_distribution = [similar(region.gap) for region in regions]

    Threads.@threads for jj in eachindex(regions)
        Xⱼ, Bⱼ = regions[jj], B[jj]

        # Using Mosek as the LP solver
        model = Model(linear_solver)
        set_silent(model)

        # Create optimization variables
        @variable(model, P[ii=eachindex(regions)], lower_bound=0, upper_bound=1) 
        @variable(model, Pᵤ, lower_bound=0, upper_bound=1)

        # Constraint ∑i=1 →k pᵢ + Pᵤ == 1
        @constraint(model, sum(P) + Pᵤ == 1)

        # Define optimization objective
        @objective(model, Max, dot(B, P) + Pᵤ)

        P̲, P̅ = prob_lower(Xⱼ), prob_upper(Xⱼ)
        val_low, val_up = min.(P̲, P̅), max.(P̲, P̅)

        P = model[:P]
        for ii in eachindex(P)
            @inbounds set_lower_bound(P[ii], val_low[ii])
            @inbounds set_upper_bound(P[ii], val_up[ii])
        end

        P̲ᵤ, P̅ᵤ = prob_unsafe_lower(Xⱼ), prob_unsafe_upper(Xⱼ)
        val_low, val_up = min(P̲ᵤ, P̅ᵤ), max(P̲ᵤ, P̅ᵤ)

        Pᵤ = model[:Pᵤ]
        set_lower_bound(Pᵤ, val_low)
        set_upper_bound(Pᵤ, val_up)

        # Optimize model
        JuMP.optimize!(model)
    
        # Print optimal values
        @inbounds β_parts[jj] = max(objective_value(model) - Bⱼ, 0)
        
        p_values = [value.(P); [value(Pᵤ)]]
        @inbounds copyto!(p_distribution[jj], p_values)
    end
   
    @debug "Solution updated beta" β = maximum(β_parts)

    return β_parts, p_distribution
end

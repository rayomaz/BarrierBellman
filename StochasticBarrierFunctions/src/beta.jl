""" Piecewise barrier function construction

    © Rayan Mazouz

"""

function verify_beta(B, regions::Vector{<:RegionWithProbabilities{T}}) where {T}
    # Don't ask. It's not pretty... But it's fast!

    β_parts = Vector{T}(undef, length(B))
    p_distribution = [zero(region.gap) for region in regions]

    Threads.@threads for jj in eachindex(regions)
        Xⱼ, Bⱼ = regions[jj], B[jj]

        model = get!(task_local_storage(), "verify_beta_model") do
            # Using Mosek as the LP solver
            model = Model(Mosek.Optimizer)
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

function OMaximization(B, indBSorted, indBSorted_perm, regions::Vector{<:RegionWithProbabilities{T}}) where {T}

    # O Maximization approach for quick sorting
    # Based on Section V.A of https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7029024

    β_parts = Vector{T}(undef, length(B))
    p_distribution = [zero(region.gap) for region in regions]

    Threads.@threads for jj in eachindex(regions)
        Xⱼ, Bⱼ = regions[jj], B[jj]

        P̲, P̅ = prob_lower(Xⱼ), prob_upper(Xⱼ)

        P̲ᵤ, P̅ᵤ = prob_unsafe_lower(Xⱼ), prob_unsafe_upper(Xⱼ)

        # Append P̲ and P̲ᵤ
        P̲ = [P̲; P̲ᵤ]

        # Append P̅ and P̅ᵤ
        P̅ = [P̅; P̅ᵤ]

        # O-maximization
        used = sum(P̲)
        remain = 1 - used

        # Initialize p_maximization
        p_maximization = zeros(1,length(indBSorted ))

        for kk in eachindex(indBSorted)

            actual_index = indBSorted_perm[kk]

            if P̅[actual_index] <= (remain + P̲[actual_index])
                p_maximization[actual_index] = P̅[actual_index]
            else
                p_maximization[actual_index] = P̲[actual_index] + remain
            end

            # Update remain
            remain = maximum([0, remain - (P̅[actual_index] - P̲[actual_index]) ])

        end

        @inbounds β_parts[jj] = max(dot(p_maximization, [B[1:length(B)]; 1.0]) - Bⱼ, 0)
        @inbounds copyto!(p_distribution[jj], p_maximization)

    end

    return β_parts, p_distribution
    
end
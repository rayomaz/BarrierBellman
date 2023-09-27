function synthesize_barrier(alg::IterativeUpperBoundAlgorithm, regions::Vector{<:RegionWithProbabilities}, initial_region::LazySet, obstacle_region::LazySet; time_horizon=1)
    iteration_prob = regions

    B, η, β = upper_bound_barrier(iteration_prob, initial_region, obstacle_region; ϵ=alg.ϵ)
    β_updated, p_distribution = verify_beta(B, regions)

    @debug "Iterations $i"

    P_distribution = []

    for i in 1:(alg.num_iterations - 1)

        iteration_prob = update_regions(iteration_prob, p_distribution)

        if alg.guided
            B, η, β = upper_bound_barrier(iteration_prob, initial_region, obstacle_region; guided=true, Bₚ = B, δ = 0.025, ϵ=alg.ϵ)    

        elseif alg.distributed 
            # Keep of track of distributions
            # push!(P_distribution, p_distribution)
            P_distribution = p_distribution

            B, η, β = upper_bound_barrier(iteration_prob, initial_region, obstacle_region; distributed=true, probability_distribution = P_distribution, ϵ=alg.ϵ) 
        else
            B, η, β = upper_bound_barrier(iteration_prob, initial_region, obstacle_region; ϵ=alg.ϵ)    
           
        end

        if alg.O_maximization
            indBSorted  = sort([B[1:length(B)]; 1.0], rev=true)
            indBSorted_perm = sortperm([B[1:length(B)]; 1.0], rev=true)
            β_updated, p_distribution = OMaximization(B, indBSorted, indBSorted_perm, regions)
            
        else     
            β_updated, p_distribution = verify_beta(B, regions)
        end

    end

    @info "CEGS Solution" η β=maximum(β_updated) Pₛ=1 - (η + maximum(β_updated) * time_horizon) iterations=alg.num_iterations

    return B, β_updated
end
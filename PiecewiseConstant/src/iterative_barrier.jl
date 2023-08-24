function iterative_barrier(regions, initial_region, obstacle_region; guided = true, distributed = false, max_iterations=10)
    iteration_prob = regions

    B, beta = constant_barrier(iteration_prob, initial_region, obstacle_region)
    beta_updated, p_distribution = accelerated_post_compute_beta(B, regions)

    @debug "Iterations $i"

    P_distribution = []

    for i in 1:(max_iterations - 1)

        iteration_prob = update_regions(iteration_prob, p_distribution)

        if guided
            B, beta = constant_barrier(iteration_prob, initial_region, obstacle_region, guided=true, Bₚ = B, δ = 0.025)    

        elseif distributed 
            B, beta = constant_barrier(iteration_prob, initial_region, obstacle_region, distributed=true, probability_distribution = P_distribution) 
            
            # Keep of track of distributions
            push!(P_distribution, p_distribution)

        else
            B, beta = constant_barrier(iteration_prob, initial_region, obstacle_region)    
           
        end

        beta_updated, p_distribution = accelerated_post_compute_beta(B, regions)

    end

    return B, beta_updated
end
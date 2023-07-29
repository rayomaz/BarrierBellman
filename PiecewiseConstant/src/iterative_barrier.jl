function iterative_barrier(regions, initial_region, obstacle_region; guided = true, δ = 0.05, max_iterations=40)
    iteration_prob = regions
    
    B, beta = constant_barrier(iteration_prob, initial_region, obstacle_region)
    beta_updated, p_distribution = accelerated_post_compute_beta(B, regions)
    
    for i in 1:(max_iterations - 1)
    
        if guided == false
            iteration_prob = update_regions(iteration_prob, p_distribution)
            B, beta = constant_barrier(iteration_prob, initial_region, obstacle_region)
            beta_updated, p_distribution = accelerated_post_compute_beta(B, regions)
        elseif guided == true
            # println(i, ", ", δ)
            iteration_prob = update_regions(iteration_prob, p_distribution)
            B, beta = guided_constant_barrier(iteration_prob, initial_region, obstacle_region, B, δ)
            beta_updated, p_distribution = accelerated_post_compute_beta(B, regions)
        end
    end

    return B, beta_updated
end
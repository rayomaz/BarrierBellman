function iterative_barrier(regions, initial_region, obstacle_region; max_iterations=10)
    iteration_prob = regions
    
    B, beta = constant_barrier(iteration_prob, initial_region, obstacle_region)
    beta_updated, p_distribution = accelerated_post_compute_beta(B, regions)
    
    for i in 1:(max_iterations - 1)
        iteration_prob = update_regions(iteration_prob, p_distribution)
        B, beta = constant_barrier(iteration_prob, initial_region, obstacle_region)
        beta_updated, p_distribution = accelerated_post_compute_beta(B, regions)
    end

    return B, beta_updated
end
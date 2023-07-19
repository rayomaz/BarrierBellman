function iterative_barrier(regions, initial_region, obstacle_region; max_iterations=10)
    iteration_prob = regions
    for i in 1:max_iterations
        @time B, beta = constant_barrier(iteration_prob, initial_region, obstacle_region)
        @time beta_updated, p_distribution = post_compute_beta(B, regions)
        iteration_prob = update_regions(iteration_prob, p_distribution)
    end

    return B, beta_updated
end
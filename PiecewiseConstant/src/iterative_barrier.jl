function iterative_barrier(regions, initial_region, obstacle_region; guided = true, distribute = true, δ = 0.025, max_iterations=10)
    iteration_prob = regions

    B, beta = constant_barrier(iteration_prob, initial_region, obstacle_region)
    beta_updated, p_distribution = accelerated_post_compute_beta(B, regions)

    println("Iterations: ", max_iterations)

    P_distribution = []

    for i in 1:(max_iterations - 1)

        iteration_prob = update_regions(iteration_prob, p_distribution)

        # Keep of track of distributions
        push!(P_distribution, p_distribution)

        if guided == false
            B, beta = constant_barrier(iteration_prob, initial_region, obstacle_region)    

        elseif guided == true
            if distribute == false
                B, beta = guided_constant_barrier(iteration_prob, initial_region, obstacle_region, B, δ)

            elseif distribute == true
                B, beta = distribution_constant_barrier(iteration_prob, initial_region, obstacle_region, P_distribution)
            end
        end

        beta_updated, p_distribution = accelerated_post_compute_beta(B, regions)

    end

    return B, beta_updated
end
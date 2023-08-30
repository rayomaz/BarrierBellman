function synthesize_barrier(alg::IterativeUpperBoundAlgorithm, regions::Vector{<:RegionWithProbabilities}, initial_region::LazySet, obstacle_region::LazySet; time_horizon=1)
    iteration_prob = regions

    B, beta = constant_barrier(iteration_prob, initial_region, obstacle_region; ϵ=alg.ϵ)
    beta_updated, p_distribution = verify_beta(B, regions)

    @debug "Iterations $i"

    P_distribution = []

    for i in 1:(max_iterations - 1)

        iteration_prob = update_regions(iteration_prob, p_distribution)

        if guided
            B, beta, η = constant_barrier(iteration_prob, initial_region, obstacle_region, guided=true, Bₚ = B, δ = 0.025; ϵ=alg.ϵ)    

        elseif distributed 
            B, beta, η = constant_barrier(iteration_prob, initial_region, obstacle_region, distributed=true, probability_distribution = P_distribution; ϵ=alg.ϵ) 
            
            # Keep of track of distributions
            push!(P_distribution, p_distribution)

        else
            B, beta, η = constant_barrier(iteration_prob, initial_region, obstacle_region)    
           
        end

        beta_updated, p_distribution = verify_beta(B, regions)

    end

    β = maximum(beta_updated)
    # @info "CEGS terminated in $(value(i)) iterations" η β=$β_values Pₛ=$(1 - η - max_β * time_horizon)

    Xs = map(region, regions)
    return ConstantBarrier(Xs, B), beta_updated
end
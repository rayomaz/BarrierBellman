export IterativeUpperBoundAlgorithm, IterativeUpperBoundAlgResult

Base.@kwdef struct IterativeUpperBoundAlgorithm <: ConstantBarrierAlgorithm
    linear_solver = default_lp_solver()
    δ = 0.025
    num_iterations = 10
    barrier_guided = false
    distribution_guided = true
end

struct IterativeUpperBoundAlgResult <: BarrierResult
    B::PiecewiseConstantBarrier
    η::Float64
    βs::Vector{Float64}
    synthesis_time::Float64  # Total time to solve the optimization problem in seconds
end

barrier(res::IterativeUpperBoundAlgResult) = res.B
eta(res::IterativeUpperBoundAlgResult) = res.η
beta(res::IterativeUpperBoundAlgResult) = maximum(res.βs)
betas(res::IterativeUpperBoundAlgResult) = res.βs
total_time(res::IterativeUpperBoundAlgResult) = res.synthesis_time

function synthesize_barrier(alg::IterativeUpperBoundAlgorithm, regions::Vector{<:RegionWithProbabilities}, initial_region::LazySet, obstacle_region::LazySet; time_horizon=1)
    synthesis_time = @elapsed begin 
        iteration_prob = regions

        B, η, β = upper_bound_barrier(alg, iteration_prob, initial_region, obstacle_region; time_horizon=time_horizon)
        β_updated, p_distribution = compute_beta(alg.linear_solver, B, regions)

        P_distributions = [p_distribution]

        for i in 1:(alg.num_iterations - 1)
            @debug "Iteration $i/$(alg.num_iterations)"

            # Note that update_regions is guarantee to return a new vector of regions
            # This is why this won't override the probabilities in `regions`.
            iteration_prob = update_regions(iteration_prob, p_distribution)

            if alg.barrier_guided
                B, η, β = upper_bound_barrier(alg, iteration_prob, initial_region, obstacle_region; time_horizon=time_horizon, Bprev=B, δ=alg.δ)
            elseif alg.distribution_guided 
                B, η, β = upper_bound_barrier(alg, iteration_prob, initial_region, obstacle_region; time_horizon=time_horizon, distributions=P_distributions) 
            else
                B, η, β = upper_bound_barrier(alg, iteration_prob, initial_region, obstacle_region; time_horizon=time_horizon)
            end

            β_updated, p_distribution = compute_beta(alg.linear_solver, B, regions)
            push!(P_distributions, p_distribution)
        end
    end

    res = IterativeUpperBoundAlgResult(B, η, β_updated, synthesis_time)

    @info "CEGS Solution" η=eta(res) β=beta(res) Pₛ=psafe(res, time_horizon) time=total_time(res) iterations=alg.num_iterations

    return res
end
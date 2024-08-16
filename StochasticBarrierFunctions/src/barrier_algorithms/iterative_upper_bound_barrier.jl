export IterativeUpperBoundAlgorithm, IterativeUpperBoundAlgResult

Base.@kwdef struct IterativeUpperBoundAlgorithm <: ConstantBarrierAlgorithm
    linear_solver = default_lp_solver()
    δ = 0.025
    num_iterations = 10
    barrier_guided = true
    distribution_guided = false
end

struct IterativeUpperBoundAlgResult <: BarrierResult
    B::PiecewiseConstantBarrier
    η::Float64
    βs::Vector{Float64}
end
barrier(res::IterativeUpperBoundAlgResult) = res.B
eta(res::IterativeUpperBoundAlgResult) = res.η
beta(res::IterativeUpperBoundAlgResult) = maximum(res.βs)
betas(res::IterativeUpperBoundAlgResult) = res.βs

function synthesize_barrier(alg::IterativeUpperBoundAlgorithm, regions::Vector{<:RegionWithProbabilities}, initial_region::LazySet, obstacle_region::LazySet; time_horizon=1)
    iteration_prob = regions

    B, η, β = upper_bound_barrier(alg, iteration_prob, initial_region, obstacle_region; time_horizon=time_horizon)
    β_updated, p_distribution = compute_beta(B, regions)

    P_distributions = []

    for i in 1:(alg.num_iterations - 1)
        @debug "Iteration $i/$(alg.num_iterations)"

        iteration_prob = update_regions(iteration_prob, p_distribution)

        if alg.barrier_guided
            B, η, β = upper_bound_barrier(alg, iteration_prob, initial_region, obstacle_region; time_horizon=time_horizon, Bprev=B, δ=alg.δ)
        elseif alg.distribution_guided 
            B, η, β = upper_bound_barrier(alg, iteration_prob, initial_region, obstacle_region; time_horizon=time_horizon, distributions=P_distributions) 
        else
            B, η, β = upper_bound_barrier(alg, iteration_prob, initial_region, obstacle_region; time_horizon=time_horizon)
        end

        β_updated, p_distribution = compute_beta(B, regions)
        push!(P_distributions, p_distribution)
    end

    res = IterativeUpperBoundAlgResult(B, η, β_updated)

    @info "CEGS Solution" η=eta(res) β=beta(res) Pₛ=psafe(res, time_horizon) iterations=alg.num_iterations

    return res
end
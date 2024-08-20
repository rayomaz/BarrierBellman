export UpperBoundAlgorithm, UpperBoundAlgResult

Base.@kwdef struct UpperBoundAlgorithm <: ConstantBarrierAlgorithm
    linear_solver = default_lp_solver()
end

struct UpperBoundAlgResult <: BarrierResult
    B::PiecewiseConstantBarrier
    η::Float64
    βs::Vector{Float64}
    synthesis_time::Float64  # Total time to solve the optimization problem in seconds
end

barrier(res::UpperBoundAlgResult) = res.B
eta(res::UpperBoundAlgResult) = res.η
beta(res::UpperBoundAlgResult) = maximum(res.βs)
betas(res::UpperBoundAlgResult) = res.βs
total_time(res::UpperBoundAlgResult) = res.synthesis_time

function synthesize_barrier(alg::UpperBoundAlgorithm, regions::Vector{<:RegionWithProbabilities}, initial_region::LazySet, obstacle_region::LazySet; time_horizon=1)
    synthesis_time = @elapsed begin 
        B, η, _ = upper_bound_barrier(alg, regions, initial_region, obstacle_region; time_horizon=time_horizon)
        β_updated, _ = compute_beta(alg.linear_solver, B, regions)
    end

    res = UpperBoundAlgResult(B, η, β_updated, synthesis_time)

    @info "Upper Bound Barrier Solution" η=eta(res) β=beta(res) Pₛ=psafe(res, time_horizon) time=total_time(res)

    return res
end

# Optimization function
function upper_bound_barrier(alg, regions::Vector{<:RegionWithProbabilities}, initial_region::LazySet, obstacle_region::LazySet; 
    Bprev=nothing, δ=0.0, distributions=[], time_horizon=1)

    # Using an LP solver
    model = Model(alg.linear_solver)
    set_silent(model)

    # Create optimization variables
    if !isnothing(Bprev)
        @variable(model, max(Bprev[j] - δ, 0) <= B[j=eachindex(regions)] <= min(Bprev[j] + δ, 1.0))
    else
        @variable(model, B[eachindex(regions)], lower_bound=0.0, upper_bound=1.0)
    end
    
    # Create probability decision variables η and β
    @variable(model, η, lower_bound=0.0)
    @variable(model, β, lower_bound=0.0)
    @variable(model, β_parts[eachindex(regions)], lower_bound=0.0)
    @constraint(model, β_parts .<= β)

    # Construct barriers
    @inbounds for (Xⱼ, Bⱼ, βⱼ) in zip(regions, B, β_parts)
        # Initial set
        if !isdisjoint(initial_region, region(Xⱼ))
            @constraint(model, Bⱼ .≤ η)
        end

        # Obstacle
        if !isdisjoint(obstacle_region, region(Xⱼ))
            @constraint(model, Bⱼ == 1)
        end

        expectation_constraint!(model, B, Xⱼ, Bⱼ, βⱼ)
    end

    # Previous distribution bounds
    for dist in distributions
        for (index, prob) in enumerate(eachcol(dist))
            P  = prob[1:end-1]
            P̅ᵤ = prob[end]
            Bⱼ = B[index]
            βⱼ = β_parts[index]

            @constraint(model, dot(B, P) + P̅ᵤ <= Bⱼ + βⱼ)
        end    
    end

    # Define optimization objective
    @objective(model, Min, η + β * time_horizon)

    # Optimize model
    JuMP.optimize!(model)

    # Barrier certificate
    B = value.(B)

    # Print optimal values
    β_values = value.(β_parts)
    η = value(η)

    Xs = map(region, regions)
    return PiecewiseConstantBarrier(Xs, B), η, β_values
end

function expectation_constraint!(model, B, Xⱼ, Bⱼ, βⱼ)

    """ Barrier martingale condition
    * ∑B[f(x)]*p(x) + Pᵤ <= B(x) + β: expanded in summations
    """

    P̅, P̅ᵤ = prob_upper(Xⱼ), prob_unsafe_upper(Xⱼ)

    # Constraint martingale
    @constraint(model, dot(B, P̅) + P̅ᵤ <= Bⱼ + βⱼ)
end
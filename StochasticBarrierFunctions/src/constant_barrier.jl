""" Piecewise barrier function construction

    © Rayan Mazouz

"""

function synthesize_barrier(alg::UpperBoundAlgorithm, regions::Vector{<:RegionWithProbabilities}, initial_region::LazySet, obstacle_region::LazySet; time_horizon=1)
    B, η, _ = upper_bound_barrier(regions, initial_region, obstacle_region; alg.ϵ)
    β_updated, _ = verify_beta(B, regions)

    @info "Upper Bound Barrier Solution" η β=maximum(β_updated) Pₛ=1 - (η + maximum(β_updated) * time_horizon)

    return B, β_updated
end

# Optimization function
function upper_bound_barrier(regions::Vector{<:RegionWithProbabilities}, initial_region::LazySet, obstacle_region::LazySet; 
    guided = false, distributed = false, Bₚ = [], δ = [], probability_distribution = [], time_horizon=1, ϵ=1e-6)

    # Using Mosek as the LP solver
    model = Model(Mosek.Optimizer)
    set_silent(model)

    # Create optimization variables
    if guided
        @variable(model, max(Bₚ[j] - δ, ϵ) <= B[j=eachindex(regions)] <= min(Bₚ[j] + δ, 1))
    else
        @variable(model, B[eachindex(regions)], lower_bound=ϵ, upper_bound=1)
    end

    # Create probability decision variables η and β
    @variable(model, η, lower_bound=ϵ)
    @variable(model, β_parts[eachindex(regions)], lower_bound=ϵ)
    @variable(model, β)
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
    if distributed
        for dist in probability_distribution

            for (index, prob) in enumerate(eachcol(dist))
                P  = prob[1:end-1]
                P̅ᵤ = prob[end]
                Bⱼ = B[index]
                βⱼ = β_parts[index]

                @constraint(model, dot(B, P) + P̅ᵤ <= Bⱼ + βⱼ)
            end    
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
    return ConstantBarrier(Xs, B), η, β_values
end

function expectation_constraint!(model, B, Xⱼ, Bⱼ, βⱼ)

    """ Barrier martingale condition
    * ∑B[f(x)]*p(x) + Pᵤ <= B(x) + β: expanded in summations
    """

    P̅, P̅ᵤ = prob_upper(Xⱼ), prob_unsafe_upper(Xⱼ)

    # Constraint martingale
    @constraint(model, dot(B, P̅) + P̅ᵤ <= Bⱼ + βⱼ)
end
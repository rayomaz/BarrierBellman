export DualAlgorithm, DualAlgResult

Base.@kwdef struct DualAlgorithm <: ConstantBarrierAlgorithm
    linear_solver = default_lp_solver()
end

struct DualAlgResult <: BarrierResult
    B::PiecewiseConstantBarrier
    η::Float64
    βs::Vector{Float64}
    synthesis_time::Float64  # Total time to solve the optimization problem in seconds
end

barrier(res::DualAlgResult) = res.B
eta(res::DualAlgResult) = res.η
beta(res::DualAlgResult) = maximum(res.βs)
betas(res::DualAlgResult) = res.βs
total_time(res::DualAlgResult) = res.synthesis_time

# Optimization function
function synthesize_barrier(alg::DualAlgorithm, regions::Vector{<:RegionWithProbabilities}, initial_region::LazySet, obstacle_region::LazySet; time_horizon=1)
    synthesis_time = @elapsed begin 
        model = Model(alg.linear_solver)
        set_silent(model)

        # Create optimization variables
        @variable(model, B[eachindex(regions)], lower_bound=0.0, upper_bound=1.0)   

        # Create probability decision variables η and β
        @variable(model, η, lower_bound=0.0)
        @variable(model, β_parts[eachindex(regions)], lower_bound=0.0)
        @variable(model, β)
        @constraint(model, β_parts .<= β)

        # Construct barriers
        for (Xⱼ, Bⱼ, βⱼ) in zip(regions, B, β_parts)
            # Initial set
            if !isdisjoint(initial_region, region(Xⱼ))
                @constraint(model, Bⱼ .≤ η)
            end

            # Obstacle
            if !isdisjoint(obstacle_region, region(Xⱼ))
                @constraint(model, Bⱼ == 1)
            end

            dual_expectation_constraint!(model, B, Xⱼ, Bⱼ, βⱼ)
        end

        # Define optimization objective
        @objective(model, Min, η + β * time_horizon)

        # Optimize model
        JuMP.optimize!(model)

        # Barrier certificate
        B = value.(B)
        β_values = value.(β_parts)
        η = value(η)

        Xs = map(region, regions)
    end

    res = DualAlgResult(PiecewiseConstantBarrier(Xs, B), η, β_values, synthesis_time)

    @info "Dual Solution" η=eta(res) β=beta(res) Pₛ=psafe(res, time_horizon) time=total_time(res)

    return res
end

function dual_expectation_constraint!(model, B, Xⱼ, Bⱼ, βⱼ) 

    """ Barrier martingale condition
    * ∑B[f(x)]*p(x) + Pᵤ <= B(x) + β: expanded in summations
    """


    P̅, P̅ᵤ = prob_upper(Xⱼ), prob_unsafe_upper(Xⱼ)
    P̲, P̲ᵤ = prob_lower(Xⱼ), prob_unsafe_lower(Xⱼ)

    # Add RHS dual constraint
    rhs = @expression(model, Bⱼ + βⱼ)

    # Construct identity matrix     H → dim([#num hypercubes + 1]) to account for Pᵤ
    H = [-I;
        I;
        -ones(1, length(B) + 1);
        ones(1, length(B) + 1)]

    # Setup c vector: [b; 1]
    c = [B; 1]

    h = [-P̲; -P̲ᵤ; P̅; P̅ᵤ; [-1]; [1]]

    # Define assynmetric constraint [Dual approach]
    asymmetric_dual_constraint!(model, c, rhs, H, h)

end

function asymmetric_dual_constraint!(model, c, rhs, H, h)
    # Constraints of the form              [max_x  cᵀx, s.t. Hx ≤ h] ≤ rhs
    # reformulated to the asymmetric dual  [min_{y ≥ 0} yᵀh, s.t. Hᵀx = c] ≤ rhs
    # and lifted to the outer minimization.
    # s.t. y ≥ 0
    #      yᵀh ≤ rhs
    #      Hᵀx = c

    m = size(H, 1)
    y = @variable(model, [1:m], lower_bound = 0)

    H, h = tosimplehrep(normalize(HPolyhedron(H, h)))

    @constraint(model, dot(y, h) ≤ rhs)
    @constraint(model, H' * y == c)
end



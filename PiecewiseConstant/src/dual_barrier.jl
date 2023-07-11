""" Piecewise barrier function construction

    © Rayan Mazouz

"""

# Optimization function
function dual_constant_barrier(regions::Vector{<:RegionWithProbabilities}, initial_region::LazySet, obstacle_region::LazySet; time_horizon=1, ϵ=1e-6)
    # Using HiGHS as the LP solver
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # Create optimization variables
    @variable(model, B[eachindex(regions)], lower_bound=ϵ, upper_bound=1)   

    # Create probability decision variables η and β
    @variable(model, η, lower_bound = ϵ)
    @variable(model, β_parts[eachindex(regions)], lower_bound=ϵ)
    @variable(model, β)
    @constraint(model, β_parts .<= β)

    # Construct barriers
    @inbounds for (Xⱼ, Bⱼ, βⱼ) in zip(regions, B, β_parts)
        # Initial set
        if !isempty(region(Xⱼ) ∩ initial_region)
            @constraint(model, Bⱼ .≤ η)
        end

        # Obstacle
        if !isempty(region(Xⱼ) ∩ obstacle_region)
            @constraint(model, Bⱼ == 1)
        end

        dual_expectation_constraint!(model, B, Xⱼ, Bⱼ, βⱼ)
    end

    # println("Synthesizing barries ... ")

    # Define optimization objective
    @objective(model, Min, η + β * time_horizon)

    # println("Objective made ... ")

    # Optimize model
    JuMP.optimize!(model)

    # Barrier certificate
    B = value.(B)

    # Print optimal values
    β_values = value.(β_parts)
    max_β = maximum(β_values)
    η = value(η)
    println("Solution dual approach: [η = $(value(η)), β = $max_β]")

    # Print model summary and number of constraints
    # println("")
    # println(solution_summary(model))
    # println("")
    # println(" Number of constraints ", sum(num_constraints(model, F, S) for (F, S) in list_of_constraint_types(model)))

    # # Print beta values to txt file
    # if isfile("probabilities/beta_dual.txt") == true
    #     rm("probabilities/beta_dual.txt")
    # end

    # open("probabilities/beta_dual.txt", "a") do io
    #     println(io, β_values)
    # end

    # if isfile("probabilities/barrier_dual.txt") == true
    #     rm("probabilities/barrier_dual.txt")
    # end

    # open("probabilities/barrier_dual.txt", "a") do io
    #     println(io, b)
    # end


    return B, β_values

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

    @constraint(model, y ⋅ h ≤ rhs)
    @constraint(model, H' * y .== c)
end



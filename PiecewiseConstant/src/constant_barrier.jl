""" Piecewise barrier function construction

    © Rayan Mazouz

"""

# Optimization function
function constant_barrier(regions::Vector{<:RegionWithProbabilities}, initial_region::LazySet, obstacle_region::LazySet; time_horizon=1, ϵ=1e-6)
    # Using HiGHS as the LP solver
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # Create optimization variables
    @variable(model, B[eachindex(regions)], lower_bound=ϵ, upper_bound=1)   

    # Create probability decision variables η and β
    @variable(model, η, lower_bound=ϵ)
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

        expectation_constraint!(model, B, Xⱼ, Bⱼ, βⱼ)
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
    println("Solution upper bound approach: [η = $(value(η)), β = $max_β]")

    # Print model summary and number of constraints
    # println("")
    # println(" Number of constraints ", sum(num_constraints(model, F, S) for (F, S) in list_of_constraint_types(model)))
    # println("")

    # # Print beta values to txt file
    # if isfile("probabilities/beta.txt") == true
    #     rm("probabilities/beta.txt")
    # end

    # open("probabilities/beta.txt", "a") do io
    #     println(io, β_values)
    # end

    # if isfile("probabilities/barrier.txt") == true
    #     rm("probabilities/barrier.txt")
    # end

    # open("probabilities/barrier.txt", "a") do io
    #     println(io, b)
    # end

    return B, β_values

end

function expectation_constraint!(model, B, Xⱼ, Bⱼ, βⱼ) 

    """ Barrier martingale condition
    * ∑B[f(x)]*p(x) + Pᵤ <= B(x) + β: expanded in summations
    """

    P̅, P̅ᵤ = prob_upper(Xⱼ), prob_unsafe_upper(Xⱼ)

    # Construct piecewise martingale constraint
    # Transition to unsafe set
    exp = AffExpr(0)

    # Bounds on Eᵢⱼ
    @inbounds for (Bᵢ, P̅ᵢ) in zip(B, P̅)
        # Martingale
        add_to_expression!(exp, P̅ᵢ, Bᵢ)
    end

    # Constraint martingale
    @constraint(model, exp + P̅ᵤ  <= Bⱼ + βⱼ)
end


""" Piecewise barrier function construction

    © Rayan Mazouz

"""

# Optimization function
function constant_barrier(regions::Vector{<:RegionWithProbabilities}, initial_region::LazySet, obstacle_region::LazySet; time_horizon=1, ϵ=1e-6)
    # Using Mosek as the LP solver
    model = Model(Mosek.Optimizer)
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
        if !isdisjoint(initial_region, region(Xⱼ))
            @constraint(model, Bⱼ .≤ η)
        end

        # Obstacle
        if !isdisjoint(obstacle_region, region(Xⱼ))
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
    #     println(io, B)
    # end

    return B, β_values

end

function guided_constant_barrier(regions::Vector{<:RegionWithProbabilities}, initial_region::LazySet, obstacle_region::LazySet, Bᵢ, δ; time_horizon=1, ϵ=1e-6)
    # Using Mosek as the LP solver
    model = Model(Mosek.Optimizer)
    set_silent(model)

    # Create optimization variables
    @variable(model, B[eachindex(regions)], lower_bound=ϵ, upper_bound=1)

    @constraint(model, Bᵢ .- δ .<= B .<= Bᵢ .+ δ)

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

    return B, β_values

end


function expectation_constraint!(model, B, Xⱼ, Bⱼ, βⱼ)

    """ Barrier martingale condition
    * ∑B[f(x)]*p(x) + Pᵤ <= B(x) + β: expanded in summations
    """

    P̅, P̅ᵤ = prob_upper(Xⱼ), prob_unsafe_upper(Xⱼ)

    P̲, P̲ᵤ = prob_lower(Xⱼ), prob_unsafe_lower(Xⱼ)

    # Trim over-conservative max probabilities in q' space
    sum_lower = 1 - sum(P̲) + P̲ᵤ
    P̅ = min.(P̅, sum_lower .- P̲)

    # Constraint martingale
    @constraint(model, dot(B, P̅) + P̅ᵤ <= Bⱼ + βⱼ)
end


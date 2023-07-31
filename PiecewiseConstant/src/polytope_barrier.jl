""" Piecewise barrier function construction

    © Rayan Mazouz

"""

# Optimization function
function polytope_constant_barrier(regions::Vector{<:RegionWithProbabilities}, initial_region::LazySet, obstacle_region::LazySet; time_horizon=1, ϵ=1e-6)
    # Using Mosek as the LP solver
    model = Model(Mosek.Optimizer)
    set_silent(model)

    # Create optimization variables
    @variable(model, B[eachindex(regions)], lower_bound=ϵ, upper_bound=1)   

    # Create probability decision variables η and β
    @variable(model, η, lower_bound = ϵ)
    @variable(model, β_parts[eachindex(regions)], lower_bound=ϵ)
    @variable(model, β)
    @constraint(model, β_parts .<= β)

    @variable(model, P[eachindex(regions), eachindex(regions)])
    @variable(model, Pᵤ[eachindex(regions)])

    # Construct barriers
    @inbounds for (jj, (Xⱼ, Bⱼ, βⱼ)) in enumerate(zip(regions, B, β_parts))

        # Initial set
        if !isdisjoint(initial_region, region(Xⱼ))
            @constraint(model, Bⱼ .≤ η)
        end

        # Obstacle
        if !isdisjoint(obstacle_region, region(Xⱼ))
            @constraint(model, Bⱼ == 1)
        end

        # Construct martingale
        polytope_expectation_constraint!(model, B, Xⱼ, Bⱼ, βⱼ, P[:, jj],  Pᵤ[jj])
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
    println("Solution polytope approach: [η = $(value(η)), β = $max_β]")

    # # Print beta values to txt file
    # if isfile("probabilities/beta_polytope.txt") == true
    #     rm("probabilities/beta_polytope.txt")
    # end

    # open("probabilities/beta_polytope.txt", "a") do io
    #     println(io, β_values)
    # end

    # if isfile("probabilities/barrier_polytope.txt") == true
    #     rm("probabilities/barrier_polytope.txt")
    # end

    # open("probabilities/barrier_polytope.txt", "a") do io
    #     println(io, B)
    # end


    return B, β_values
end


function polytope_expectation_constraint!(model, B, Xⱼ, Bⱼ, βⱼ, P, Pᵤ) 

    """ Barrier martingale condition
    * ∑B[f(x)]*p(x) + Pᵤ <= B(x) + β: expanded in summations
    """


    P̅, P̅ᵤ = prob_upper(Xⱼ), prob_unsafe_upper(Xⱼ)
    P̲, P̲ᵤ = prob_lower(Xⱼ), prob_unsafe_lower(Xⱼ)

    # Construct polytope
    H = [-I;
        I;
        -ones(1, length(B) + 1);
        ones(1, length(B) + 1)]

    h = [-P̲; -P̲ᵤ; P̅; P̅ᵤ; [-1]; [1]]

    polytope = HPolytope(H, h)

    vertices = LazySets.vertices_list(polytope)
    
    ## Add constraint here:

    # Comment: finding the vertices is too slow. This approach does not seem very promosing. 

end



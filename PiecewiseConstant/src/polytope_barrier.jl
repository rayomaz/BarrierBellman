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
    for jj in eachindex(regions)
        Xⱼ, Bⱼ, βⱼ = regions[jj], B[jj], β_parts[jj]

        # Initial set
        if !isdisjoint(initial_region, region(Xⱼ))
            @constraint(model, Bⱼ .≤ η)
        end

        # Obstacle
        if !isdisjoint(obstacle_region, region(Xⱼ))
            @constraint(model, Bⱼ == 1)
        end

        # Construct martingale
        polytope_expectation_constraint!(model, B, Xⱼ, Bⱼ, βⱼ, P[jj, :],  Pᵤ[jj])
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
    

        #     # Create optimization variables
    #     # Since we are bounding the variables by constants, we can do so on creation
    #     P̲, P̅ = prob_lower(Xⱼ), prob_upper(Xⱼ)
    #     val_low, val_up = min.(P̲, P̅), max.(P̲, P̅)
    #     @variable(model, val_low[ii] <= P[ii=eachindex(regions)] <= val_up[ii]) 

    #     println(P)
    #     return 0,0

    #     P̲ᵤ, P̅ᵤ = prob_unsafe_lower(Xⱼ), prob_unsafe_upper(Xⱼ)
    #     val_low, val_up = min(P̲ᵤ, P̅ᵤ), max(P̲ᵤ, P̅ᵤ)
    #     @variable(model, Pᵤ, lower_bound=val_low, upper_bound=val_up)    

    #     # Create probability decision variables β
    #     @variable(model, β)

    #     # Constraint ∑i=1 →k pᵢ + Pᵤ == 1
    #     @constraint(model, sum(P) + Pᵤ == 1)

    #     # Constraint martingale
    #     @constraint(model, dot(B, P̅) + P̅ᵤ <= Bⱼ + βⱼ)

    #     # Define optimization objective
    #     @objective(model, Max, β)
    
    #     # Optimize model
    #     JuMP.optimize!(model)
    
    #     # Print optimal values
    #     @inbounds β_parts[jj] = max(value(β), ϵ)

    #     p_values = [value.(P); [value(Pᵤ)]]
    #     @inbounds p_distribution[:, jj] = p_values
    # end

    # # Add RHS dual constraint
    # rhs = @expression(model, Bⱼ + βⱼ)

    # # Construct identity matrix     H → dim([#num hypercubes + 1]) to account for Pᵤ
    # H = [-I;
    #     I;
    #     -ones(1, length(B) + 1);
    #     ones(1, length(B) + 1)]

    # # Setup c vector: [b; 1]
    # c = [B; 1]

    # h = [-P̲; -P̲ᵤ; P̅; P̅ᵤ; [-1]; [1]]

    # # Define assynmetric constraint [Dual approach]
    
    # # Constraint martingale
    # @constraint(model, dot(B, P̅) + P̅ᵤ <= Bⱼ + βⱼ)

end



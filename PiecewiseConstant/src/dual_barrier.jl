""" Piecewise barrier function construction

    © Rayan Mazouz

"""

# Optimization function
function dual_constant_barrier(probabilities::MatlabFile)
    # Load probability matrices
    prob_lower = read(probabilities, "matrix_prob_lower")
    prob_upper = read(probabilities, "matrix_prob_upper")
    prob_unsafe_lower = read(probabilities, "matrix_prob_unsafe_lower")
    prob_unsafe_upper = read(probabilities, "matrix_prob_unsafe_upper")

    return dual_constant_barrier(prob_lower, prob_upper, prob_unsafe_lower, prob_unsafe_upper)
end

function dual_constant_barrier(prob_lower, prob_upper, prob_unsafe_lower, prob_unsafe_upper, initial_regions=round(Int, length(prob_unsafe_upper) / 2), obstacle_regions=nothing; time_horizon=1, ϵ=1e-6)

    # Number of hypercubes
    number_hypercubes = length(prob_unsafe_upper)

    # Using HiGHS as the LP solver
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # Create optimization variables
    @variable(model, b[1:number_hypercubes] >= ϵ)

    # Obstacle barrier
    if !isnothing(obstacle_regions)
        @constraint(model, b[obstacle_regions] == 1)
    end

    # Initial set
    @variable(model, η, lower_bound = ϵ)
    @constraint(model, b[initial_regions] .≤ η)

    # Create probability decision variables β
    @variable(model, β_parts_var[1:number_hypercubes], lower_bound = ϵ, upper_bound = 1 - ϵ)
    @variable(model, β)
    @constraint(model, β_parts_var .<= β)

    # Construct barriers
    @inbounds for jj in eachindex(b)
        probability_bounds = [prob_lower[jj, :],
            prob_upper[jj, :],
            prob_unsafe_lower[jj],
            prob_unsafe_upper[jj]]
        dual_expectation_constraint!(model, b, probability_bounds, b[jj], β_parts_var[jj])
    end

    # println("Synthesizing barries ... ")

    # Define optimization objective
    @objective(model, Min, η + β * time_horizon)

    # println("Objective made ... ")

    # Optimize model
    JuMP.optimize!(model)

    # Barrier certificate
    b = value.(b)
    # for Bⱼ in b
    #     println(Bⱼ)
    # end

    # Print optimal values
    β_values = value.(β_parts_var)
    max_β = maximum(β_values)
    η = value(η)
    println("Solution: [η = $(value(η)), β = $max_β]")

    # Print model summary and number of constraints
    # println("")
    # println(solution_summary(model))
    # println("")
    # println(" Number of constraints ", sum(num_constraints(model, F, S) for (F, S) in list_of_constraint_types(model)))


    return b, β_values

end

function dual_expectation_constraint!(model, b, probability_bounds, Bⱼ, βⱼ) 

    """ Barrier martingale condition
    * ∑B[f(x)]*p(x) + Pᵤ <= B(x) + β: expanded in summations
    """

    (prob_lower,
        prob_upper,
        prob_unsafe_lower,
        prob_unsafe_upper) = probability_bounds

    # Add RHS dual constraint
    rhs = Bⱼ + βⱼ

    # Construct identity matrix     H → dim([#num hypercubes + 1]) to account for Pᵤ
    H = [-I;
        I;
        -ones(1, length(b) + 1);
        ones(1, length(b) + 1)]

    # Setup c vector: [b; 1]
    c = [b; 1]

    h = [-prob_lower; -prob_unsafe_lower;
        prob_upper; prob_unsafe_upper;
        [-1];
        [1]]

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



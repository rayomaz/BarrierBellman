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

function dual_constant_barrier(prob_lower, prob_upper, prob_unsafe_lower, prob_unsafe_upper)
    
    # Number of hypercubes
    number_hypercubes = length(prob_unsafe_upper)

    # Number of hypercubes
    initial_state_partition = Int(round(number_hypercubes/2))

    # Using HiGHS as the LP solver
    optimizer = optimizer_with_attributes(HiGHS.Optimizer)
    model = Model(optimizer)

    # Create optimization variables
    ϵ = 1e-6
    @variable(model, b[1:number_hypercubes])    
    @constraint(model, b .>= ϵ)
    @constraint(model, b .<= 1 - ϵ)

    # Create probability decision variables β
    @variable(model, ϵ <= β_parts_var[1:number_hypercubes] <= 1 - ϵ)
    @variable(model, β)
    @constraint(model, β_parts_var .<= β)

    # Construct barriers
    for jj = 1:number_hypercubes
          probability_bounds = [prob_lower[jj, :],
                                prob_upper[jj, :],
                                prob_unsafe_lower[jj],
                                prob_unsafe_upper[jj]]
          dual_expectation_constraint!(model, b, jj, probability_bounds, β_parts_var[jj])
    end

    println("Synthesizing barries ... ")

    # Define optimization objective
    time_horizon = 1
    η = b[initial_state_partition]
    @objective(model, Min, η + β * time_horizon)

    println("Objective made ... ")

    # Optimize model
    JuMP.optimize!(model)

    # Barrier certificate
    for jj in 1:number_hypercubes
        certificate = value.(b[jj])
        println(certificate)
    end

    # Print optimal values
    β_values = value.(β_parts_var)
    max_β = maximum(β_values)
    η = value.(b[initial_state_partition])
    println("Solution: [η = $(value(η)), β = $max_β]")

    # Print model summary and number of constraints
    println("")
    println(solution_summary(model))
    println("")
    println(" Number of constraints ", sum(num_constraints(model, F, S) for (F, S) in list_of_constraint_types(model)))


    return value.(b), β_values

end

function dual_expectation_constraint!(model, b, jj, probability_bounds, βⱼ) 

    """ Barrier martingale condition
    * ∑B[f(x)]*p(x) + Pᵤ <= B(x) + β: expanded in summations
    """

    (prob_lower, 
     prob_upper, 
     prob_unsafe_lower, 
     prob_unsafe_upper) = probability_bounds

    # Barrier jth partition
    Bⱼ = b[jj]

    # Add RHS dual constraint
    rhs = Bⱼ + βⱼ

    # Construct identity matrix     H → dim([#num hypercubes + 1]) to account for Pᵤ
    H = [-Matrix(1.0I, length(b) + 1, length(b) + 1);
         Matrix(1.0I, length(b) + 1, length(b) + 1);
         ones(1, length(b) + 1);
         -ones(1, length(b) + 1)]

    # Setup c vector: [b 1]
    c = [b; 1]

    h = [-prob_lower; -prob_unsafe_lower; 
         prob_upper; prob_unsafe_upper;
         [1];
         [-1]]
    
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



""" Piecewise barrier function construction

    © Rayan Mazouz

"""

poly_type = SOSPoly
cone_type = SOSCone

# Optimization function
function piecewise_barrier(system::AdditiveGaussianPolynomialSystem{T, N}, bounds, initial_state_partition) where {T, N}
    # Using Mosek as the SDP solver
    optimizer = optimizer_with_attributes(Mosek.Optimizer,
        "MSK_DPAR_INTPNT_TOL_STEP_SIZE" => 1e-6,
        "MSK_IPAR_OPTIMIZER" => 0,
        "MSK_IPAR_BI_CLEAN_OPTIMIZER" => 0,
        "MSK_IPAR_NUM_THREADS" => 16,
        "MSK_IPAR_PRESOLVE_USE" => 0)
    model = SOSModel(optimizer)

    # Hyperspace
    lower_partitions = read(bounds, "lower_partition")
    upper_partitions = read(bounds, "upper_partition")
    state_partitions = hcat(lower_partitions, upper_partitions)
    state_partitions = [Hyperrectangle(low=[low], high=[high]) for (low, high) in eachrow(state_partitions)]
    number_state_hypercubes = length(state_partitions)

    # Create probability decision variables eta
    @variable(model, η >= ϵ)

    # Create optimization variables
    @variable(model, A[1:number_state_hypercubes, 1:N])
    @variable(model, b[1:number_state_hypercubes])    
    @variable(model, ϵ <= β_parts_var[1:number_state_hypercubes] <= 1 - ϵ)
    @variable(model, β)

    @constraint(model, β_parts_var .<= β)

    # Construct barriers
    B = [barrier_construct(system, A[jj, :], b[jj]) for jj in eachindex(state_partitions)]

    # Extract probability bounds
    lower_prob_A = read(bounds, "lower_probability_bounds_A")
    lower_prob_b = read(bounds, "lower_probability_bounds_b")
    
    upper_prob_A = read(bounds, "upper_probability_bounds_A")
    upper_prob_b = read(bounds, "upper_probability_bounds_b")

    # Construct piecewise constraints
    for (jj, region) in enumerate(state_partitions)

        nonnegativity_constraint!(model, B[jj], system, region, lagrange_degree)

        if jj == initial_state_partition
            initial_constraint!(model, B[jj], system, region, η, lagrange_degree)
        end

        current_state_partition = state_partitions[jj]

        """ Probability bounds
            - Bounds are loaded from .mat file
            - Structure of type:
                A = [i, j, p, x]
                b = [i, j, p]
            - i: each index of all state partitions
            - j: current state partition
            - Both lower and upper bounds on A and b are included
        """
        probability_bounds = [lower_prob_A[:, jj, 1, :], 
                              lower_prob_b[:, jj, 1],
                              upper_prob_A[:, jj, 1, :], 
                              upper_prob_b[:, jj, 1]]

        #! safe_prob_bounds = define here
        #! input into expectation constraint                      

        expectation_constraint!(model, B, B[jj], system, probability_bounds,
                                β_parts_var[jj], current_state_partition, lagrange_degree)

    end

    # Define optimization objective
    time_horizon = 1
    @objective(model, Min, η + β*time_horizon)
    println("Objective made")

    # Optimize model
    optimize!(model)

    # Barrier certificate
    for jj in 1:number_state_hypercubes
        certificate = piecewise_barrier_certificate(B[jj])
        println(certificate)
    end

    # Print optimal values
    β_values = value.(β_parts_var)
    max_β = maximum(β_values)
    println("Solution: [η = $(value(η)), β = $max_β]")

    println("")
    println(solution_summary(model))
end

function nonnegativity_constraint!(model, barrier, system, region, lagrange_degree)
    """ Barrier condition: nonnegativity
        * B(x) >= 0
    """
    x = variables(system)
    positive_set = 0.0

    lower_state = low(region)
    upper_state = high(region)
    product_set = (upper_state - x) .* (x - lower_state)

    for (xi, dim_set) in zip(x, product_set)
        # Lagragian multiplier
        monos = monomials(xi, 0:lagrange_degree)
        lag_poly_positive = @variable(model, variable_type=poly_type(monos))

        # Specify initial range
        positive_set += lag_poly_positive * dim_set
    end

    barrier_set_nonnegative = polynomial(barrier) - positive_set

    # Non-negative in Xᵢ ⊂ ℝⁿ 
    @constraint(model, barrier_set_nonnegative in cone_type())
end

function initial_constraint!(model, barrier, system, region, η, lagrange_degree)
    """ Barrier condition: initial
        * B(x) <= η
    """
    x = variables(system)
    initial_state = 0.0

    lower_state = low(region)
    upper_state = high(region)
    product_set = (upper_state - x) .* (x - lower_state)

    for (xi, dim_set) in zip(x, product_set)
        # Lagragian multiplier
        monos = monomials(xi, 0:lagrange_degree)
        lag_poly_initial = @variable(model, variable_type=poly_type(monos))

        # Specify initial range
        initial_state += lag_poly_initial * dim_set
    end

    # Add constraint to model
    _barrier_initial = -polynomial(barrier) + η - initial_state
    @constraint(model, _barrier_initial in cone_type())
end

function expectation_constraint!(model, barriers, Bⱼ, system::AdditiveGaussianPolynomialSystem{T, N}, probability_bounds, 
                                 βⱼ, current_state_partition, lagrange_degree) where {T, N}
    """ Barrier martingale condition
        * E[B(f(x))] <= B(x) + β: expanded in summations
    """

    x = variables(system)
    fx = dynamics(system)

    # Martingale terms
    @polyvar P[1:(length(barriers) + 1)]        # + 1 P to constraint Pᵤ
    @polyvar E[1:N, 1:length(barriers)]

    # Current state partition
    x_k_lower = low(current_state_partition)
    x_k_upper = high(current_state_partition)
    product_set = (x_k_upper - x) .* (x - x_k_lower)

    # Semi-algebraic set for current partition only
    hCubeSOS_X = 0

    for (xi, dim_set) in zip(x, product_set)
        monos = monomials(xi, 0:lagrange_degree)

        # Generate SOS Lagragian for partition bounds
        lag_poly_X = @variable(model, variable_type=poly_type(monos))

        # Generate SOS polynomials for bounds
        hCubeSOS_X += lag_poly_X * dim_set
    end

    # Construct piecewise martingale constraint
    martingale = 0
    hCubeSOS_P = 0
    hCubeSOS_E = 0

    (lower_probability_A, lower_probability_b, upper_probability_A, upper_probability_b) = probability_bounds

    #! safe_prob_bounds = define here as tuple
    #! Bounds on Pj - Ps (using martingale)
    # Pᵤ = P[end]
    #! Notice: to get Pu --> 1 - Ps

    # Bounds on Eij
    exponential_terms = exponential_bounds(system, current_state_partition)

    e_min = exponential_terms[1]
    e_max = exponential_terms[2]

    constant = 1/(2^(N - 1) * sqrt(2 * π))

    for (ii, Bᵢ) in enumerate(barriers)

        # Bounds on Pij
        lower_probability_bound = dot(lower_probability_A[ii], x) + lower_probability_b[ii]
        upper_probability_bound = dot(upper_probability_A[ii], x) + upper_probability_b[ii]
        probability_set = (upper_probability_bound - P[ii]) * (P[ii] - lower_probability_bound)
        
        # Generate probability Lagrangian
        monos_P = monomials(P[ii], 0:lagrange_degree)
        lag_poly_P = @variable(model, variable_type=poly_type(monos_P))
        hCubeSOS_P += lag_poly_P * probability_set

        # Compute B(f(x))
        barrier_fx = subs(polynomial(Bᵢ), x => fx)

        # Martingale
        martingale -= barrier_fx * P[ii]

        #! note term is 1 for the 1D case
        #! for N > 1, this term becomes the bounds on the modified product of erf functions
        term = 1

        lower_expectation_bound = [constant*dot(e_min, term)] + fx .* lower_probability_bound
        upper_expectation_bound = [constant*dot(e_max, term)] + fx .* upper_probability_bound
        expectation_product_set = (upper_expectation_bound - E[:,ii]) .* (E[:,ii] - lower_expectation_bound)

        # Generate expecation Lagrangian
        for (Ekk, dim_set) in zip(E[:,ii], expectation_product_set)
            monos_E = monomials(Ekk, 0:lagrange_degree)
            lag_poly_E = @variable(model, variable_type=poly_type(monos_E))
            hCubeSOS_E += lag_poly_E * polynomial(dim_set)
        end
    
        # Martingale
        martingale -= dot(Bᵢ.A, E[:,ii])
    end

    #! Pᵤ constraint [total probability == 1, later!)
    # """ Process:
    #     # Generate Polynomial Lagragian for bounds on Pᵤ
    #     # Constraint dot product(L, sum(P) -1)
    #     # Subtract from martingale
    # """
    # monos_Pᵤ = monomials(P, 0:lagrange_degree)
    # lag_poly_Pᵤ = @variable(model, variable_type=Poly(monos_Pᵤ))
    # sum_prob_constraint = dot(lag_poly_Pᵤ, sum(P) - 1)
    
    # Constraint martingale
    #! martingale_condition_multivariate = martingale - Pᵤ + polynomial(Bⱼ) + βⱼ - hCubeSOS_X - hCubeSOS_P - hCubeSOS_E - sum_prob_constraint
    martingale_condition_multivariate = martingale + polynomial(Bⱼ) + βⱼ - hCubeSOS_X - hCubeSOS_P - hCubeSOS_E
    certificate = InterregionRemovedNewton(cone_type(), MB.MonomialBasis, x, vcat(P, vec(E)))
    @constraint(model, martingale_condition_multivariate in cone_type(), certificate = certificate)
end
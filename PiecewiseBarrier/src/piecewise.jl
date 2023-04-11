""" Piecewise barrier function construction

    © Rayan Mazouz

"""

# Optimization function
function piecewise_barrier(system::AdditiveGaussianPolynomialSystem{T, N}, state_partitions, initial_state_partition) where {T, N}
    # Using Mosek as the SDP solver
    optimizer = optimizer_with_attributes(Mosek.Optimizer,
        "MSK_DPAR_INTPNT_TOL_STEP_SIZE" => 1e-6,
        "MSK_IPAR_OPTIMIZER" => 0,
        "MSK_IPAR_BI_CLEAN_OPTIMIZER" => 0,
        "MSK_IPAR_NUM_THREADS" => 16,
        "MSK_IPAR_PRESOLVE_USE" => 0)
    model = SOSModel(optimizer)

    # Hyperspace
    number_state_hypercubes = length(state_partitions)

    # Create probability decision variables eta
    @variable(model, η >= ϵ)

    # Create optimization variables
    @variable(model, A[1:number_state_hypercubes, 1:N])
    @variable(model, b[1:number_state_hypercubes])
    @variable(model, ϵ <= β_parts_var[1:number_state_hypercubes] <= 1 - ϵ)
    @variable(model, β)

    # Construct piecewise constraints
    for (jj, region) in enumerate(state_partitions)

        # Construct partition barrier
        Bⱼ = barrier_construct(system, A[jj, :], b[jj])

        nonnegativity_constraint!(model, Bⱼ)

        if jj == initial_state_partition
            initial_constraint!(model, Bⱼ, system, region, η, lagrange_degree)
        end

        expectation_constraint!(model, Bⱼ, system, β_parts_var, β, state_partitions, lagrange_degree)

    end

    # Define optimization objective
    time_horizon = 1
    @objective(model, Min, η + β*time_horizon)
    println("Objective made")

    # Optimize model
    optimize!(model)

    # Barrier certificate
    for jj in 1:number_state_hypercubes
        certificate = piecewise_barrier_certificate(system, A[jj, :], b[jj])
        println(certificate)
    end

    # Print optimal values
    β_values = value.(β_parts_var)
    max_β = maximum(β_values)
    println("Solution: [η = $(value(η)), β = $max_β]")

    println("")
    println(solution_summary(model))
end

function nonnegativity_constraint!(model, barrier)
    """ Barrier condition: nonnegativity
        * B(x) >= 0
    """
    # Non-negative in ℝⁿ 
    @constraint(model, barrier >= 0)
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
        lag_poly_initial = @variable(model, variable_type=SOSPoly(monos))

        # Specify initial range
        initial_state += lag_poly_initial * dim_set
    end

    # Add constraint to model
    _barrier_initial = -barrier + η - initial_state
    @constraint(model, _barrier_initial >= 0)
end

function expectation_constraint!(model, barrier, system::AdditiveGaussianPolynomialSystem{T, N}, β_parts_var, β, state_partitions, lagrange_degree) where {T, N}
    """ Barrier martingale condition
        * E[B(f(x,u))] <= B(x) + β
    """

    x = variables(system)
    fx = dynamics(system)

    # Create noise variable
    @polyvar z[1:N]
    
    # Create constraints for X (Partition)
    for state in eachindex(state_partitions)

        # Current state partition
        current_state_partition = state_partitions[state]

        # Semi-algebraic sets
        hCubeSOS_X = 0

        x_k_lower = low(current_state_partition)
        x_k_upper = high(current_state_partition)
        product_set = (x_k_upper - x) .* (x - x_k_lower)

        # Loop over state dimensions
        for (xi, dim_set) in zip(x, product_set)
            # Generate Lagragian for partition bounds
            monos = monomials(xi, 0:lagrange_degree)
            lag_poly_X = @variable(model, variable_type=SOSPoly(monos))

            # Generate SOS polynomials for bounds
            hCubeSOS_X += lag_poly_X * dim_set
        end

        # Compute expectation
        _e_barrier = barrier
        exp_evaluated = subs(_e_barrier, x => fx + z)

        # Extract noise term
        σ_noise = noise_distribution(system)
        exp = expectation_noise(exp_evaluated, σ_noise, z)

        # Constraint for hypercube
        martingale_condition_multivariate = -exp + barrier + β_parts_var[state] - hCubeSOS_X
        @constraint(model, martingale_condition_multivariate >= 0)

        # Non-negative constraint
        non_negative = barrier - hCubeSOS_X
        @constraint(model, non_negative >= 0)

        # Add constraint for maximum beta approach
        @constraint(model, β_parts_var[state] <= β)

    end
end
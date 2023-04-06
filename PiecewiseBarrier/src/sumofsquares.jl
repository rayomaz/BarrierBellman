""" SOS barrier function construction

    © Rayan Mazouz

"""

# Sum of squares optimization function
function sos_barrier(system_dimension, state_space,
                     state_partitions,
                     initial_state_partition)                               

    # Using Mosek as the SDP solver
    optimizer = optimizer_with_attributes(Mosek.Optimizer,
        "MSK_DPAR_INTPNT_TOL_STEP_SIZE" => 1e-6,
        "MSK_IPAR_OPTIMIZER" => 0,
        "MSK_IPAR_BI_CLEAN_OPTIMIZER" => 0,
        "MSK_IPAR_NUM_THREADS" => 16,
        "MSK_IPAR_PRESOLVE_USE" => 0)
    model = SOSModel(optimizer)
    
    # Create state space variables
    @polyvar x[1:system_dimension]

    # Hyperspace
    number_state_hypercubes = length(state_partitions)

    # Create probability decision variables eta
    @variable(model, η >= ϵ)

    # Create barrier candidate
    barrier_monomials = monomials(x, 0:barrier_degree)
    @variable(model, BARRIER, SOSPoly(barrier_monomials))

    sos_initial_constraint!(model, BARRIER, x, state_partitions[initial_state_partition], η, lagrange_degree)
    sos_unsafe_constraint!(model, BARRIER, x, state_space, lagrange_degree)

    """ Barrier martingale condition
        * E[B(f(x,u))] <= B(x) + β
    """
    martingale = true
    if martingale
        # Optimization variables beta
        @variable(model, ϵ <= β_parts_var[1:number_state_hypercubes] <= 1 - ϵ)

        sos_expectation_constraint!(model, BARRIER, x, β_parts_var, state_partitions, lagrange_degree)
    end

    # Define optimization objective
    if martingale
        time_horizon = 1
        #! The objective ought to be @objective(model, Min, η + max(β_parts_var) * time_horizon).
        #! Can be implemented using more constraints.
        @objective(model, Min, η + sum(β_parts_var) * time_horizon)
    else
        @objective(model, Min, η)
    end

    # Optimize model
    optimize!(model)

    # Lagrange values
    # print(value(lag_vars_initial[1]), " ", value(lag_vars_initial[2]), " ", value(lag_vars_initial[3]))
    # return 0,0
    # print(value(lag_vars_unsafe_upper[1,1]), " " , value(lag_vars_unsafe_upper[1,2]), " " , value(lag_vars_unsafe_upper[1,3]))

    # Barrier certificate
    certificate = polynomial(value(BARRIER))

    # Run barrier certificate validation tests
    # nonnegative_barrier(certificate, state_space, system_dimension)
    # unsafe_barrier(certificate, state_space, system_dimension)

    # Print barrier
    println("B(x) = $certificate")

    # Print optimal values
    β_values = value.(β_parts_var)
    max_β = maximum(β_values)
    println("Solution: [η = $(value(η)), β = $max_β]")

    # # Return optimal values
    # eta = value(η)
    # if martingale
    #     β_values = value.(β_parts_var)
    #     return certificate, eta, β_values
    # else
    #     return certificate, eta, 0
    # end

end

function sos_initial_constraint!(model, barrier, x, region, η, lagrange_degree)
    """ Barrier condition: initial
    * B(x) <= η
    """

    lower_state = low(region)
    upper_state = high(region)
    product_set = (upper_state - x) .* (x - lower_state)

    for (xi, dim_set) in zip(x, product_set)
        # Lagragian multiplier
        monos = monomials(xi, 0:lagrange_degree)
        lag_poly_initial = @variable(model, variable_type=SOSPoly(monos))

        # Extract lower and upper bound
        # Specify initial range
        initial_state = lag_poly_initial * dim_set

        # Add constraint to model
        _barrier_initial = -barrier + η - initial_state
        @constraint(model, _barrier_initial >= 0)
    end
end

function sos_unsafe_constraint!(model, barrier, x, state_space, lagrange_degree)
    """ Barrier unsafe region conditions
    * B(x) >= 1
    """
    product_set_lower = state_space[1] - x
    product_set_upper = x - state_space[2]

    for (xi, dim_set_lower, dim_set_upper) in zip(x, product_set_lower, product_set_upper)

        # Lagragian multiplier
        monos = monomials(xi, 0:lagrange_degree)
        lag_poly_i_lower = @variable(model, variable_type=SOSPoly(monos))
        lag_poly_i_upper = @variable(model, variable_type=SOSPoly(monos))

        # Specify constraints for initial and unsafe set
        _barrier_unsafe_lower = barrier - lag_poly_i_lower * dim_set_lower - 1.0
        _barrier_unsafe_upper = barrier - lag_poly_i_upper * dim_set_upper - 1.0

        # Add constraints to model
        @constraint(model, _barrier_unsafe_lower >= 0)
        @constraint(model, _barrier_unsafe_upper >= 0)

    end
end

function sos_expectation_constraint!(model, barrier, x, β_parts_var, state_partitions, lagrange_degree)

    system_dimension = length(x)

    # Create constraints for X (Partition), μ (Mean Dynamics) and σ (Noise Variable)
    for state in eachindex(state_partitions)

        # Current state partition
        current_state_partition = state_partitions[state]

        # Semi-algebraic sets
        hCubeSOS_X = 0

        # Partition bounds
        x_k_lower = low(current_state_partition)
        x_k_upper = high(current_state_partition)
        product_set = (x_k_upper - x) .* (x - x_k_lower)

        # Loop over state dimensions
        for (xk, dim_set) in zip(x, product_set)
            # Generate Lagragian for partition bounds
            monos = monomials(xk, 0:lagrange_degree)
            lag_poly_X = @variable(model, variable_type=SOSPoly(monos))

            # Generate SOS polynomials for bounds
            hCubeSOS_X += lag_poly_X * dim_set
        end

        # Create noise variable
        @polyvar z[1:system_dimension]

        # Compute expectation
        _e_barrier = barrier
        exp_evaluated = _e_barrier
        
        # Dummy system
        for zz in 1:system_dimension
            exp_evaluated = subs(exp_evaluated, x[zz] => 0.95*x[zz] + z[zz])
        end

        # Extract noise term
        exp = expectation_noise(exp_evaluated, σ_noise, z)

        # Constraint for hypercube
        martingale_condition_multivariate = -exp + barrier + β_parts_var[state] - hCubeSOS_X
        @constraint(model, martingale_condition_multivariate >= 0)
    end
end
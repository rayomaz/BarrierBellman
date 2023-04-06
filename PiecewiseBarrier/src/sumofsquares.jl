""" SOS barrier function construction

    © Rayan Mazouz

"""

# Sum of squares optimization function
function sos_barrier(state_space,
                     state_partitions,
                     σ_noise,
                     barrier_degree_input,
                     initial_state_partition)                               

    # System Specifications
    system_dimension = Int(1)

    # Using Mosek as the SDP solver
    model = SOSModel(optimizer_with_attributes(Mosek.Optimizer,
                                               "MSK_DPAR_INTPNT_TOL_STEP_SIZE" => 1e-6,
                                               "MSK_IPAR_OPTIMIZER" => 0,
                                               "MSK_IPAR_BI_CLEAN_OPTIMIZER" => 0,
                                               "MSK_IPAR_NUM_THREADS" => 16,
                                               "MSK_IPAR_PRESOLVE_USE" => 0))
    
    # Create state space variables
    @polyvar x[1:system_dimension]

    # Numerical precision
    ϵ = 1e-6

    # Hyperspace
    number_state_hypercubes = Int(length(state_partitions))   

    # Create probability decision variables eta
    @variable(model, η)
    @constraint(model, η >= ϵ )

    # Create barrier polynomial and specify degree Lagrangian polynomials
    barrier_degree::Int64 = barrier_degree_input
    lagrange_degree = 2
    length_per_lagrange_func::Int64 = length_polynomial(x::Array{PolyVar{true},1}, lagrange_degree::Int64)

    # Create barrier candidate
    barrier_monomial::MonomialVector{true} = monomials(x, 0:barrier_degree)
    @variable(model, c[1:Integer(length(barrier_monomial))])
    BARRIER::DynamicPolynomials.Polynomial{true, AffExpr} = barrier_polynomial(c, barrier_monomial)

    # Non-negative in ℜ^n
    add_constraint_to_model(model, BARRIER)

    # One initial condition and unsafe conditions
    @variable(model, lag_vars_initial[1:system_dimension, 1:length_per_lagrange_func])
    @variable(model, lag_vars_unsafe_lower[1:system_dimension, 1:length_per_lagrange_func])
    @variable(model, lag_vars_unsafe_upper[1:system_dimension, 1:length_per_lagrange_func])

    """ Barrier condition: initial
        * B(x) <= η
    """
    initial_condition_state_partition = split(state_partitions[initial_state_partition])
    initial_condition_state_partition = parse.(Float64, initial_condition_state_partition)
    if system_dimension > 1
        initial_condition_state_partition = reshape(initial_condition_state_partition, (system_dimension, system_dimension))
    end

    for ii = 1:system_dimension

        # Lagragian multiplier
        lag_poly_initial::DynamicPolynomials.Polynomial{true, AffExpr} = sos_polynomial(lag_vars_initial[ii,:], x[ii], lagrange_degree::Int64)
        add_constraint_to_model(model, lag_poly_initial)

        # Extract lower and upper bound
        lower_state = initial_condition_state_partition[1, ii]
        upper_state = initial_condition_state_partition[2, ii]

        # Specify initial range
        initial_state = lag_poly_initial * (upper_state - x[ii]) * (x[ii] - lower_state)

        # Add constraint to model
        _barrier_initial = -BARRIER + η - initial_state
        add_constraint_to_model(model, _barrier_initial)

    end

    """ Barrier unsafe region conditions
        * B(x) >= 1
    """

    for ii = 1:system_dimension

        lag_poly_i_lower::DynamicPolynomials.Polynomial{true, AffExpr} =  sos_polynomial(lag_vars_unsafe_lower[ii, :], x[ii], lagrange_degree::Int64)
        lag_poly_i_upper::DynamicPolynomials.Polynomial{true, AffExpr} =  sos_polynomial(lag_vars_unsafe_upper[ii, :], x[ii], lagrange_degree::Int64)

        add_constraint_to_model(model, lag_poly_i_lower)
        add_constraint_to_model(model, lag_poly_i_upper)

        # State space ranges
        if system_dimension == 1
            x_i_lower = state_space[1, ii]
            x_i_upper = state_space[2, ii]
        else
            x_i_lower = state_space[ii][1]
            x_i_upper = state_space[ii][2]
        end

        # Specify constraints for initial and unsafe set
        _barrier_unsafe_lower = BARRIER - lag_poly_i_lower * (x_i_lower - x[ii]) - 1
        _barrier_unsafe_upper = BARRIER - lag_poly_i_upper * (x[ii] - x_i_upper) - 1

        # Add constraints to model
        add_constraint_to_model(model, lag_poly_i_lower)
        add_constraint_to_model(model, lag_poly_i_upper)
        add_constraint_to_model(model, _barrier_unsafe_lower)
        add_constraint_to_model(model, _barrier_unsafe_upper)

    end

    """ Barrier martingale condition
        * E[B(f(x,u))] <= B(x) + β
    """
    martingale = true
    if martingale == true

        # Optimization variables beta
        @variable(model, β_parts_var[1:number_state_hypercubes])

        for states = 1:number_state_hypercubes
            @constraint(model, β_parts_var[states] >= ϵ)
            @constraint(model, β_parts_var[states] <= (1 - ϵ))
        end

        # Specify standard Lagrangian optimization variables
        @variable(model, lag_vars_X[1:number_state_hypercubes, 1:system_dimension, 1:length_per_lagrange_func])          # Lagrange to bound hyperspace

        # Specify constraints per loop
        number_constraints_per_loop = system_dimension

        # Create constraints for X (Partition), μ (Mean Dynamics) and σ (Noise Variable)
        for state ∈ eachindex(state_partitions)

            # Current state partition
            current_state_partition = split(state_partitions[state])
            current_state_partition = parse.(Float64, current_state_partition)
            if system_dimension > 1
                current_state_partition = reshape(current_state_partition, (system_dimension, system_dimension))
            end

            # Setup constraints array
            constraints = Array{DynamicPolynomials.Polynomial{true, AffExpr}}(undef, Integer(1), number_constraints_per_loop)

            # Semi-algebraic sets
            hCubeSOS_X = 0

            # Loop over state dimensions
            for kk = 1:system_dimension

                # Partition bounds
                x_k_lower::Float64 = current_state_partition[1, kk]
                x_k_upper::Float64 = current_state_partition[2, kk]

                # Generate Lagragian for partition bounds
                lag_poly_X = sos_polynomial(lag_vars_X[state, kk, :], x[kk], lagrange_degree::Int64)
                constraints[Integer(1), kk] = lag_poly_X

                # Generate SOS polynomials for bounds
                hCubeSOS_X += lag_poly_X*(x_k_upper - x[kk])*(x[kk] - x_k_lower)

            end

            # Create noise variable
            @polyvar z[1:system_dimension]

            # Compute expectation
            _e_barrier::DynamicPolynomials.Polynomial{true, AffExpr} = BARRIER
            exp_evaluated::DynamicPolynomials.Polynomial{true, AffExpr} = _e_barrier
            
            # Dummy system
            for zz = 1:system_dimension
                exp_evaluated = subs(exp_evaluated, x[zz] => 0.5*x[1] + z[zz])
            end

            # Extract noise term
            exp_poly, noise = expectation_noise(exp_evaluated, barrier_degree::Int64, σ_noise, z::Vector{PolyVar{true}})

            # Full expectation term
            exp_current = exp_poly + noise

            # Constraint for hypercube
            martingale_condition_multivariate = - exp_current + BARRIER + β_parts_var[state] - hCubeSOS_X

            # Constraint for hypercube
            constraints[Integer(1), number_constraints_per_loop] = martingale_condition_multivariate

            # Add constraints to model
            @constraint(model, constraints .>= 0)

        end

    end

    # Define optimization objective
    if martingale == true
        time_horizon = 1
        @objective(model, Min, η + sum(β_parts_var)*time_horizon)
    else
        @objective(model, Min, η)
    end

    # Optimize model
    set_silent(model)
    optimize!(model)

    # Lagrange values
    # print(value(lag_vars_initial[1]), " ", value(lag_vars_initial[2]), " ", value(lag_vars_initial[3]))
    # return 0,0
    # print(value(lag_vars_unsafe_upper[1,1]), " " , value(lag_vars_unsafe_upper[1,2]), " " , value(lag_vars_unsafe_upper[1,3]))

    # Barrier certificate
    certificate = sos_barrier_certificate(barrier_monomial, c)
    print("\n", certificate, "\n")

    # Optimal values
    eta = value(η)
    if martingale == true
        β_values = value.(β_parts_var)
        return certificate, eta, β_values
    else
        return certificate, eta, 0
    end

end
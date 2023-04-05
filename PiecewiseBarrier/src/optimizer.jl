""" Barrier optimization function

    © Rayan Mazouz

"""

# Optimization function
function barrier_bellman_sos(system_dimension, state_space, state_partitions, initial_state_partition)

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
    lagrange_degree = 2
    length_per_lagrange_func::Int64 = length_polynomial(x::Array{PolyVar{true},1}, lagrange_degree::Int64)

    # Create optimization variables
    @variable(model, A[1:(system_dimension*number_state_hypercubes)])
    @variable(model, b[1:number_state_hypercubes])
    @variable(model, β_parts_var[1:number_state_hypercubes])

    # One initial condition and unsafe conditions
    @variable(model, lag_vars_initial[1:system_dimension, 1:length_per_lagrange_func])
    @variable(model, lag_vars_unsafe_lower[1:number_state_hypercubes, 1:system_dimension, 1:length_per_lagrange_func])
    @variable(model, lag_vars_unsafe_upper[1:number_state_hypercubes, 1:system_dimension, 1:length_per_lagrange_func])

    # Specify standard Lagrangian optimization variables for martingale
    @variable(model, lag_vars_X[1:number_state_hypercubes, 1:system_dimension, 1:length_per_lagrange_func])          # Lagrange to bound hyperspace

    # Construct piecewise constraints
    
    martingale = true

    for jj ∈ eachindex(state_partitions)

        # Construct partition barrier
        A_j = A[(1+system_dimension*(jj-1):(system_dimension*jj))]
        BARRIER_j = barrier_construct(system_dimension, A_j, b[jj], x)

        """ Barrier condition: initial
            * B(x) >= 0
        """
        # Non-negative in ℜ^n
        add_constraint_to_model(model, BARRIER_j)


        """ Barrier condition: initial
            * B(x) <= η
        """
        if jj == initial_state_partition

            initial_state = 0.0

            for ii = 1:system_dimension

                # Extract initial
                initial_condition_state_partition = split(state_partitions[initial_state_partition])
                initial_condition_state_partition = parse.(Float64, initial_condition_state_partition)
                if system_dimension > 1
                    initial_condition_state_partition = reshape(initial_condition_state_partition, (system_dimension, system_dimension))
                end

                # Lagragian multiplier
                lag_poly_initial::DynamicPolynomials.Polynomial{true, AffExpr} = sos_polynomial(lag_vars_initial[ii,:], x[ii], lagrange_degree::Int64)
                add_constraint_to_model(model, lag_poly_initial)
        
                # Extract lower and upper bound
                lower_state = initial_condition_state_partition[1, ii]
                upper_state = initial_condition_state_partition[2, ii]
        
                # Specify initial range
                initial_state += lag_poly_initial * (upper_state - x[ii]) * (x[ii] - lower_state)
        
            end

            # Add constraint to model
            _barrier_initial = -BARRIER_j + η - initial_state
            add_constraint_to_model(model, _barrier_initial)
            
        end

        """ Barrier martingale condition
            * E[B(f(x,u))] <= B(x) + β
        """

        if martingale == true

            # Optimization variables beta
            for states = 1:number_state_hypercubes
                @constraint(model, β_parts_var[states] >= ϵ)
                @constraint(model, β_parts_var[states] <= (1 - ϵ))
            end

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
                _e_barrier::DynamicPolynomials.Polynomial{true, AffExpr} = BARRIER_j
                exp_evaluated::DynamicPolynomials.Polynomial{true, AffExpr} = _e_barrier
                
                # Dummy system
                for zz = 1:system_dimension
                    exp_evaluated = subs(exp_evaluated, x[zz] => 0.5*x[1]^2 + z[zz])
                end

                # Extract noise term
                barrier_degree = Int(1)
                σ_noise = 0.01
                exp_poly, noise = expectation_noise(exp_evaluated, barrier_degree::Int64, σ_noise, z::Vector{PolyVar{true}})

                # Full expectation term
                exp_current = exp_poly + noise

                # Constraint for hypercube
                martingale_condition_multivariate = - exp_current + BARRIER_j + β_parts_var[state] - hCubeSOS_X

                # Constraint for hypercube
                constraints[Integer(1), number_constraints_per_loop] = martingale_condition_multivariate

                # Add constraints to model
                @constraint(model, constraints .>= 0)

            end

        end

    end


    # Define optimization objective
    time_horizon = 1
    if martingale == true
        @objective(model, Min, η + sum(β_parts_var)*time_horizon)
    else
        @objective(model, Min, η)
    end
    print("Objective made\n")

    # Optimize model
    optimize!(model)

    # Barrier certificate
    for jj = 1:number_state_hypercubes
        A_j = A[(1+system_dimension*(jj-1):(system_dimension*jj))]
        certificate = piecewise_barrier_certificate(system_dimension, A_j, b[jj], x)
        print("\n", certificate, "\n")
    end

end
            # Compute transition probability
        # transition_probabilities = probability_distribution(hypercubes, covariance_matrix, system_dimension, system_flag, neural_flag)
#         total_expectation = total_law_of_expectation()

#         # return transition_probabilities

#     # Specify Covariance Matrix (Gaussian)
#     covariance_matrix = random_covariance_matrix(system_dimension) 

#         if jj == hypercube_initial 
#             @variable(model, eta)
#             @constraint(model, eta >= 1e-6)
#             @constraint(model, eta <= (1 - 1e-6))
#         end


    # # Expectation constraint
    # # Variables g and h for Lagrange multipliers
    # lagrange_monomial_length::Int64 = length_polynomial(x::Array{PolyVar{true},1}, lagrange_degree::Int64)
    # number_of_variables_exp::Int64 = system_dimension * lagrange_monomial_length
    # @variable(model, g[1:number_of_variables_exp])
    # @variable(model, h[1:number_of_variables_exp])

    # if neural_flag == true
    #     number_constraints_per_loop = (2*system_dimension) + 1
    # else
    #     number_constraints_per_loop = (system_dimension + 1)
    # end
    # constraints = Array{DynamicPolynomials.Polynomial{true, AffExpr}}(undef, 1, number_constraints_per_loop)
  
    # # Counters
    # counter_lag::Int64 = 0




    # for parts = 1:number_of_hypercubes
     
    #     # Create SOS polynomials for X (Partition) and Y (Bounds)
    #     hCubeSOS_X::DynamicPolynomials.Polynomial{true, AffExpr} = 0
    #     hCubeSOS_Y::DynamicPolynomials.Polynomial{true, AffExpr} = 0

    #     # Loop of state space and dynamics bounds
    #     for kk = 1:system_dimension
         
    #         # Partition bounds
    #         x_k_lower::Float64 = -3.0
    #         x_k_upper::Float64 = 3.0

    #         # Dynamics bounds
    #         if neural_flag == true
    #             y_k_lower_explicit = -1
    #             y_k_upper_explicit = +1
    #         end

    #         # Generate Lagrange polynomial for kth dimension
    #         lag_poly_X::DynamicPolynomials.Polynomial{true, AffExpr} = sos_polynomial(g::Vector{VariableRef}, x::Array{PolyVar{true},1}, (counter_lag + kk - 1)::Int64, lagrange_degree::Int64)

    #         if neural_flag == true
    #             lag_poly_Y::DynamicPolynomials.Polynomial{true, AffExpr} = sos_polynomial(h::Vector{VariableRef}, y::Array{PolyVar{true},1}, (counter_lag + kk - 1)::Int64, lagrange_degree::Int64)
    #         end

    #         # Add Lagrange polynomial to constraints vector for the state space
    #         constraints[parts, kk] = lag_poly_X

    #         if neural_flag == true
    #             constraints[parts, kk + system_dimension] = lag_poly_Y
    #         end

    #         # Generate SOS polynomials for state space
    #         hCubeSOS_X::DynamicPolynomials.Polynomial{true, AffExpr} += lag_poly_X*(x_k_upper - x[kk])*(x[kk] - x_k_lower)
            
    #         if neural_flag == true
    #             hCubeSOS_Y::DynamicPolynomials.Polynomial{true, AffExpr} += lag_poly_Y*(y_k_upper_explicit - y[kk])*(y[kk] - y_k_lower_explicit)
    #         end

    #     end

    #     # Update system counter
    #     counter_lag += system_dimension

    #     # Compute expectation
    #     _e_barrier::DynamicPolynomials.Polynomial{true, AffExpr} = BARRIER
    #     exp_evaluated::DynamicPolynomials.Polynomial{true, AffExpr} = _e_barrier

    #     # System properties
    #     m1 = 0.50
    #     m2 = 0.95
    #     m3 = 0.50

    #     if neural_flag == true
    #         for zz = 1:system_dimension
    #             exp_evaluated = subs(exp_evaluated, x[zz] => y[zz] + z)
    #         end
    #     else

    #         exp_current = b

    #         for zz = 1:system_dimension

    #             # Dynamics: population growth
    #             if system_flag == "population_growth"
    #                 if zz == 1
    #                     u_k = 0
    #                     y_dynamics = m3*x[2] + u_k

    #                     exp_current += A[1]*y_dynamics

    #                 elseif zz == 2
    #                     y_dynamics = m1*x[1] + m2*x[2] #+ sigma*z

    #                     exp_current += A[2]*y_dynamics

    #                 end
            
    #             end

    #             # exp_evaluated = subs(exp_evaluated, x[zz] => y_dynamics)
                
    #         end
    #     end

    #     # Extract noise term
    #     # barrier_degree = 1
    #     # exp_poly, noise = expectation_noise(exp_evaluated, barrier_degree::Int64, sigma::Float64, z::PolyVar{true})

    #     # Full expectation term
    #     # exp_current = exp_poly + noise

      
    #     # Constraint for expectation
    #     if neural_flag == true
    #         hyper_constraint = - exp_current + BARRIER/alpha + beta - hCubeSOS_X - hCubeSOS_Y
    #     else
    #         hyper_constraint = - exp_current + BARRIER/alpha + beta - hCubeSOS_X
    #     end

    #     # Add to model
    #     constraints[parts, number_constraints_per_loop] = hyper_constraint

    # end

    # # Add constraints to model as a vector of constraints
    # @time begin
    #     @constraint(model, constraints .>= 0)
    # end
    # print("Constraints made\n")






""" Piecewise barrier function construction

    © Rayan Mazouz

"""

# ACCUMULATION OF BUGS FOUND:
# 3. Incorrect barrier in expectation (j, not i)
# 4. A martingale constraint per pair (i, j) not per j
# 5. Missing unsafe set constraint B(x) >= 1 for all x in Xᵤ 

# TO DO LIST:

# 1. Indexing dynamics


# Optimization function
function piecewise_barrier(system_dimension, state_space, state_partitions, initial_state_partition)

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

    # Create optimization variables
    @variable(model, A[1:number_state_hypercubes, 1:system_dimension])
    @variable(model, b[1:number_state_hypercubes])
    @variable(model, ϵ <= β_parts_var[1:number_state_hypercubes] <= 1 - ϵ)
    @variable(model, β)

    # Construct piecewise constraints
    for (jj, region) in enumerate(state_partitions)

        # Construct partition barrier
        Bⱼ = barrier_construct(system_dimension, A[jj, :], b[jj], x)

        nonnegativity_constraint!(model, Bⱼ)

        #! No unsafe set constraint

        if jj == initial_state_partition
            initial_constraint!(model, Bⱼ, x, region, η, lagrange_degree)
        end

        expectation_constraint!(model, Bⱼ, x, β_parts_var, β, state_partitions, lagrange_degree)

    end

    # Define optimization objective
    time_horizon = 1
    @objective(model, Min, η + β*time_horizon)
    println("Objective made")

    # Optimize model
    optimize!(model)

    # Barrier certificate
    for jj in 1:number_state_hypercubes
        certificate = piecewise_barrier_certificate(system_dimension, A[jj, :], b[jj], x)
        println(certificate)
    end

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

function initial_constraint!(model, barrier, x, region, η, lagrange_degree)
    """ Barrier condition: initial
    * B(x) <= η
    """
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

function expectation_constraint!(model, barrier, x, β_parts_var, β, state_partitions, lagrange_degree)
    """ Barrier martingale condition
    * E[B(f(x,u))] <= B(x) + β
    """

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

        system_dimension = length(x)

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

        # Add constraint for maximum beta approach
        maximum_beta_constraint!(model, β_parts_var[state], β)

        #! There should only be one martingale constraint for each j, not for each pair (i, j)
        #! In the expectation, the barrier Bᵢ should be used.
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






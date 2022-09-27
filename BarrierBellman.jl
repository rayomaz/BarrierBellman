# Control Barrier Functions rapid computation using Bellman's Equation
# Rayan Mazouz, University of Colorado Boulder, Aerospace Engineering Sciences

# Initialization Statement
print("\n Computing Piecewise Barriers based on Bellman Operator")

# Import packages
using SumOfSquares
using DynamicPolynomials
using MosekTools
using MAT
using Polynomials
using StatsBase
using Combinatorics
using LinearAlgebra
using GLPK
using Optim



# # Function to compute number of decision variables per Lagrange function
# function length_polynomial(var::Array{PolyVar{true},1}, degree::Int64)::Int64
#     sos_polynomial::MonomialVector{true}  = monomials(var, 0:degree)
#     length_polynomial::Int64 = length(sos_polynomial)
#     return length_polynomial
# end

# Function to add constraints to the model
function add_constraint_to_model(model::Model, expression)
    @constraint(model, expression >= 0)
end

# # Function to compute the expecation and noise element
# function expectation_noise(exp_evaluated::DynamicPolynomials.Polynomial{true, AffExpr}, barrier_degree::Int64, standard_deviation::Float64, z::PolyVar{true})

# end


# Control hypercube optimization 
function barrier_bellman_linear(system_flag, system_dimension, identifier, partitions, part_of_initial, M_h_ii, M_l_ii, B_h_ii, B_l_ii)

    # Initialize solver
    model = Model()
    set_optimizer(model, GLPK.Optimizer)

    # Create optimization variables 
    @variable(model, x[1:system_dimension])
    @variable(model, x_prime[1:system_dimension])
    @variable(model, A[1:system_dimension*system_dimension])
    @variable(model, b[1:system_dimension])

    # Probability constraints [initial set]
    if part_of_initial == true
        @variable(model, eta)
        @constraint(model, eta >= 1e-6)
        @constraint(model, eta <= (1 - 1e-6))
    end
    
    # Probability constraints [c-martingale]
    @variable(model, beta)
    @constraint(model, beta >= 1e-6)

    # Create barrier candidate
    # A_matrix = zeros(system_dimension, system_dimension)
    bar = 1
    range_decision_vars = ((bar-1)*system_dimension + 1):(bar*system_dimension)
    print(A[range_decision_vars]*x')
    return 0
    BARRIER = A[range_decision_vars]*x' + b[bar] 

    print(BARRIER)
    # BARRIER = []
    # for bar = 1:system_dimension
    #     range_decision_vars = ((bar-1)*system_dimension + 1):(bar*system_dimension)
    #     BARRIER += A[range_decision_vars]*x + b[bar] 
    # end
    # print("\n", A_matrix)
    #\BARRIER = Ax + b



    # # Add constraints to model for positive barrier, eta and beta
    # add_constraint_to_model(model, BARRIER)

    return 0


    # Initial set constraint


    # Non-negative constraint


    # Unsafe region constraint


    # C-martingale constraint

    # Specify noise element (Gaussian)
    standard_deviation::Float64 = 0.01
    
    # Free variable bounds
    @constraint(lp_control_model, x .== y - y_prime)    

    # Loop over hcube higher bound
    hyper_matrix_higher = M_h_ii * (y-y_prime) + B_h_ii
    
    # Loop over hcube lower bound
    hyper_matrix_lower = M_l_ii * (y-y_prime) + B_l_ii
    

    # Create constraints
    for ctrl = 1:system_dimension

        x_k_hcube_bound_jj = partitions[identifier, :, ctrl]
        x_k_lower_jj = x_k_hcube_bound_jj[1, 1]
        x_k_upper_jj = x_k_hcube_bound_jj[2, 1]
        @constraint(lp_control_model,  y[ctrl] - y_prime[ctrl] >=  x_k_lower_jj)
        @constraint(lp_control_model,  y[ctrl] - y_prime[ctrl] <=  x_k_upper_jj)

        @constraint(lp_control_model, x_prime[ctrl] - x_star[ctrl] <= theta[ctrl])
        @constraint(lp_control_model, x_star[ctrl] - x_prime[ctrl] <= theta[ctrl])

        y_k_upper_explicit = hyper_matrix_higher[ctrl]
        y_k_lower_explicit = hyper_matrix_lower[ctrl]
        @constraint(lp_control_model, x_prime[ctrl] >= y_k_lower_explicit + u[ctrl])
        @constraint(lp_control_model, x_prime[ctrl] <= y_k_upper_explicit + u[ctrl])

        @constraint(lp_control_model, u[ctrl] >= -10)
        @constraint(lp_control_model, u[ctrl] <= 10)

    end

    #Define Objective
    @objective(lp_control_model, Min, sum(theta))

    #Run the opimization
    set_silent(lp_control_model)
    optimize!(lp_control_model)

    # Return feedback control law
    feedback_law = value.(u)
    return feedback_law

end


# Full Piecewise Barrier Constructer
function optimization(number_hypercubes::Int64,
                      system_flag::String, 
                      layer_flag::String,
                      unsafety_flag::String,
                      objective_flag::String,
                      global_flag::Bool, 
                      eta_flag::Bool,
                      large_range_initial::Bool,
                      print_to_txt::Bool)

    # File reading
    filename = "/models/" * system_flag * "/" * "alpha" * "/" * layer_flag* "_layers/partition_data_"  * string(number_hypercubes) * ".mat"

    # Extract data
    file = matopen(pwd()*filename)
    partitions = read(file, "partitions")
    state_space = read(file, "state_space")
    system_dimension::Int64 = Integer(length(state_space[:,1]))

    # Extract Neural Network Bounds
    if global_flag == true
        G_lb = read(file, "G_lb")
        G_ub = read(file, "G_ub")
    else
        M_h = read(file, "M_h")
        M_l = read(file, "M_l")
        B_h = read(file, "B_h")
        B_l = read(file, "B_l")
    end

    # Parallel Barrier Construction
    # for jj = 1:number_hypercubes
    for jj = 1:1

        # Designate identifier
        identifier = jj

        # Define global or explicit upper and lower bound for kth dimension of partition parts
        if global_flag == true
            y_k_upper_global::Float64 = G_ub[identifier, kk]
            y_k_lower_global::Float64 = G_lb[identifier, kk]
        else
            M_h_ii = transpose(M_h[identifier, :, :])
            M_l_ii = transpose(M_l[identifier, :, :])
            B_h_ii = B_h[identifier, :]
            B_l_ii = B_l[identifier, :]
        end

        # Determine which partitions belong to initial set
        part_of_initial = false
        if jj == 1
            part_of_initial = true
        end

        # Construct piecewise barrier
        barrier_bellman(system_flag, system_dimension, identifier, partitions, part_of_initial, M_h_ii, M_l_ii, B_h_ii, B_l_ii)

       
    end

    return 0


    # One initial condition and unsafe conditions
    if system_flag == "pendulum" && large_range_initial == true 
        number_decision_vars = ((system_dimension)^2 + 1 + 1)*length(barrier_monomial)
    end
    @variable(model, l[1:number_decision_vars])

    barrier_constraints_unsafe_initial = system_dimension + 1

    for ii = 1:barrier_constraints_unsafe_initial

        # Barrier initial condition f(eta)
        if ii == barrier_constraints_unsafe_initial

            # Generate sos polynomial
            count_lag = 0

            # Optimize this code: not all x variables needed in lag_poly_i, lag_poly_theta (only 1 2 4 and 3, respectively)
            lag_poly_i::DynamicPolynomials.Polynomial{true, AffExpr} =  sos_polynomial(l::Vector{VariableRef}, x::Array{PolyVar{true},1}, count_lag::Int64, lagrange_degree::Int64)
            add_constraint_to_model(model, lag_poly_i)
            
            # Change the radius ball of theta
            if system_flag == "pendulum" && large_range_initial == true 
                lag_poly_theta_pen::DynamicPolynomials.Polynomial{true, AffExpr} =  sos_polynomial(l::Vector{VariableRef}, x::Array{PolyVar{true},1}, (count_lag+1)::Int64, lagrange_degree::Int64)
                add_constraint_to_model(model, lag_poly_theta_pen)
            end

            # Initial condition radius and ball
            x_initial_radius = (1e-8)
            x_initial_sums = x_initial_radius
            
            if system_flag == "pendulum" && large_range_initial == true 
                theta_radius = deg2rad(5.0)^2
                theta_initial_sums = x_initial_radius

                for jj = 1:length(x)
                    if jj != 3
                        x_initial_sums += -(x[jj] - x_init[jj])^2
                    else 
                        theta_initial_sums += -(x[jj] - x_init[jj])^2
                    end
                end
            end

            # Barrier constraint eta
            if system_flag == "pendulum" && large_range_initial == true
                _barrier_initial = - BARRIER + eta - lag_poly_i * x_initial_sums - lag_poly_theta_pen*theta_initial_sums
            else
                _barrier_initial = - BARRIER + eta - lag_poly_i * x_initial_sums
            end

            # Add constraint to model
            add_constraint_to_model(model, _barrier_initial)


        # Barrier unsafe region conditions (twodim)
        elseif system_flag == "pendulum" 
    
            if ii == 1

                # Generate sos polynomials
                    count_lag = 2*ii
                    lag_poly_i_lower =  sos_polynomial(l::Vector{VariableRef}, x::Array{PolyVar{true},1}, (count_lag - 1)::Int64, lagrange_degree::Int64)
                    lag_poly_i_upper =  sos_polynomial(l::Vector{VariableRef}, x::Array{PolyVar{true},1}, (count_lag)::Int64, lagrange_degree::Int64)
        
                    # State space ranges
                    x_i_lower = state_space[ii, 1]
                    x_i_upper = state_space[ii, 2]
        
                    # Specify constraints for initial and unsafe set
                    _barrier_unsafe_lower = BARRIER - lag_poly_i_lower * (x_i_lower - x[ii]) - 1
                    _barrier_unsafe_upper = BARRIER - lag_poly_i_upper * (x[ii] - x_i_upper) - 1
        
                    # Add constraints to model
                    add_constraint_to_model(model, lag_poly_i_lower)
                    add_constraint_to_model(model, lag_poly_i_upper)
                    add_constraint_to_model(model, _barrier_unsafe_lower)
                    add_constraint_to_model(model, _barrier_unsafe_upper)

            end

        else
            continue
        end

    end

    # Variables g and h for Lagrange multipliers

    counter_lag::Int64 = 0
    parts_count::Int64 = 0
    counter_beta::Int64 = 0

    for parts = 1:length(hcube_identifier)

        if hcube_identifier[parts] == 0.0
            continue
        else
            parts_count += 1
            identifier = Integer(hcube_identifier[parts])
        end

        # Create SOS polynomials for X (Partition) and Y (Bounds)
        hCubeSOS_X::DynamicPolynomials.Polynomial{true, AffExpr} = 0
        hCubeSOS_Y::DynamicPolynomials.Polynomial{true, AffExpr} = 0
        

        # Loop of state space and neural network bounds
        for kk = 1:system_dimension

            # Partition bounds
            x_k_hcube_bound::Vector{Float64} = partitions[identifier, :, kk]
            x_k_lower::Float64 = x_k_hcube_bound[1, 1]
            x_k_upper::Float64 = x_k_hcube_bound[2, 1]
          
            # Loop over hcube higher bound
            hyper_matrix_higher = M_h_ii * x + B_h_ii
            y_k_upper_explicit = hyper_matrix_higher[kk]

            # Loop over hcube lower bound
            hyper_matrix_lower = M_l_ii * x + B_l_ii
            y_k_lower_explicit = hyper_matrix_lower[kk]

            if control_flag == true
                if beta_vals_opt[parts] >= safety_threshold_beta
                    y_k_upper_explicit += feedback_control[kk]
                    y_k_lower_explicit += feedback_control[kk]
                end
            end

            # Generate Lagrange polynomial for kth dimension
            lag_poly_X::DynamicPolynomials.Polynomial{true, AffExpr} = sos_polynomial(g::Vector{VariableRef}, x::Array{PolyVar{true},1}, (counter_lag + kk - 1)::Int64, lagrange_degree::Int64)
            lag_poly_Y::DynamicPolynomials.Polynomial{true, AffExpr} = sos_polynomial(h::Vector{VariableRef}, y::Array{PolyVar{true},1}, (counter_lag + kk - 1)::Int64, lagrange_degree::Int64)

            # Add Lagrange polynomial to constraints vector for the state space
            constraints[parts_count, kk] = lag_poly_X
            constraints[parts_count, kk + system_dimension] = lag_poly_Y

            # Generate SOS polynomials for state space
            hCubeSOS_X::DynamicPolynomials.Polynomial{true, AffExpr} += lag_poly_X*(x_k_upper - x[kk])*(x[kk] - x_k_lower)

            if global_flag == true
                hCubeSOS_Y::DynamicPolynomials.Polynomial{true, AffExpr} += lag_poly_Y*(y_k_upper_global - y[kk])*(y[kk] - y_k_lower_global)
            else
                hCubeSOS_Y::DynamicPolynomials.Polynomial{true, AffExpr} += lag_poly_Y*(y_k_upper_explicit - y[kk])*(y[kk] - y_k_lower_explicit)
            end

        end

        # Update system counter
        counter_lag += system_dimension

        # SOS for beta partition
        if beta_partition == true
    
            lag_poly_beta::DynamicPolynomials.Polynomial{true, AffExpr} = sos_polynomial(delta::Vector{VariableRef}, w::Array{PolyVar{true},1}, (counter_beta)::Int64, lagrange_degree::Int64)

            constraints[parts_count, (2*system_dimension) + 1] = lag_poly_beta

            counter_beta += 1

        end

        # Compute expectation
        _e_barrier::DynamicPolynomials.Polynomial{true, AffExpr} = BARRIER
        exp_evaluated::DynamicPolynomials.Polynomial{true, AffExpr} = _e_barrier

        for zz = 1:system_dimension
            exp_evaluated = subs(exp_evaluated, x[zz] => y[zz] + z)
        end

        # Extract noise term
        exp_poly, noise = expectation_noise(exp_evaluated, barrier_degree::Int64, standard_deviation::Float64, z::PolyVar{true})

        # Full expectation term
        exp_current = exp_poly + noise

        # Constraint for hypercube
        if beta_partition == true
            hyper_constraint = - exp_current + BARRIER/alpha + beta_parts_var[parts_count] - hCubeSOS_X - hCubeSOS_Y

            w_min = 0
            w_max = 1

            beta_constraint = (- beta_parts_var[parts_count] + beta - lag_poly_beta)*(w_max - w[1])*(w[1] - w_min)

            constraints[parts_count, number_constraints_per_loop - 1] = beta_constraint

        else
            hyper_constraint = - exp_current + BARRIER/alpha + beta - hCubeSOS_X - hCubeSOS_Y
        end


        # Add to model
        constraints[parts_count, number_constraints_per_loop] = hyper_constraint

    end
    
    # Add constraints to model as a vector of constraints
    @time begin
        @constraint(model, constraints .>= 0)
    end
    print("Constraints made\n")

    # Define optimization objective
    time_horizon = 1

    if objective_flag == "min_max"
        @objective(model, Min, eta + beta*time_horizon)
    elseif objective_flag == "sum"
        @objective(model, Min, eta + sum(beta_parts_var)/(number_hypercubes)*time_horizon)
    elseif objective_flag == "min_max_sum"
        @objective(model, Min, eta + (beta + sum(beta_parts_var)/(number_hypercubes))*time_horizon)
    end
    print("Objective made\n")

    # Print number of partitions
    print("\n", "Optimizing for number of partitions = " * string(number_hypercubes), "\n")

    # Optimize model
    optimize!(model)

    # Barrier certificate
    certificate = barrier_certificate(barrier_monomial, c)

    # Argmin barrier certificate

    # Return beta values
    if beta_partition == true
        beta_values = value.(beta_parts_var)
        max_beta = maximum(beta_values)

        lag_Y_vals = value.(h)

        # Print probability values
        println("Solution: [eta = $(value(eta)), beta = $(value(max_beta)), total = $(value(eta) + value(max_beta)) ]")
        # end

    else

        lag_Y_vals = value.(h)

        # Print probability values
        println("Solution: [eta = $(value(eta)), beta = $(value(beta)), total = $(value(eta) + value(beta)) ]")
        
    end

    # Print beta values to txt file
    if control_flag == true
        if beta_partition == true

            # Print beta values to txt file
            if isfile("probabilities/beta_vals_controller.txt") == true
                rm("probabilities/beta_vals_controller.txt")
            end

            open("probabilities/beta_vals_controller.txt","a") do io
                println(io, beta_values)
            end

        else
            if isfile("probabilities/probs_"*system_flag*".txt") == true
                rm("probabilities/probs_"*system_flag*".txt")
            end
    
            open("probabilities/probs_"*system_flag*".txt", "a") do io
               println(io, "eta = $(value(eta)), beta = $(value(beta)), total = $(value(eta) + value(beta)) ")
            end
        end

    else
        if beta_partition == true

            # Print beta values to txt file
            if isfile("probabilities/beta_vals_certificate.txt") == true
                rm("probabilities/beta_vals_certificate.txt")
            end

            open("probabilities/beta_vals_certificate.txt","a") do io
                println(io, beta_values)
            end
        else
            if isfile("probabilities/probs_"*system_flag*".txt") == true
                rm("probabilities/probs_"*system_flag*".txt")
            end
    
            open("probabilities/probs_"*system_flag*".txt", "a") do io
               println(io, "eta = $(value(eta)), beta = $(value(beta)), total = $(value(eta) + value(beta)) ")
            end
        end
    end

    # Return optimization results
    if beta_partition == true
        if control_flag == true
            return certificate
        else
            return certificate,  beta_values
        end
    else
        return certificate
    end

end

# Optimization flags
system_flag = "pendulum"
layer_flag = "1"
unsafety_flag = "outside"
objective_flag = "min_max"
global_flag = false
decision_eta_flag = true
large_range_initial = true
print_to_txt = true

# Optimize certificate
number_hypercubes = 120


# Optimize controller
@time controller = optimization(number_hypercubes,
                                system_flag,
                                layer_flag,
                                unsafety_flag,
                                objective_flag,
                                global_flag,
                                decision_eta_flag,
                                large_range_initial,
                                print_to_txt)
                                        



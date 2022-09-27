# Dynamical Neural Network Verification using Control Barrier Functions
# Rayan Mazouz, University of Colorado Boulder, Aerospace Engineering Sciences

# Initialization Statement
print("\n Computing Control Barrier Certificate & Controller based on Neural Network Bounds ")

# Import packages
using SumOfSquares
using DynamicPolynomials
using MosekTools
using MAT
using Polynomials
using StatsBase
using Combinatorics
using LinearAlgebra
using JuMP
using GLPK
using Optim

# Create Control Barrier Polynomial
function barrier_polynomial(c::Vector{VariableRef}, barrier_monomial::MonomialVector{true})::DynamicPolynomials.Polynomial{true, AffExpr}
    barrier_poly = 0
    for cc in 1:Integer(length(barrier_monomial))
        barrier_poly += c[cc] * barrier_monomial[cc]
    end
    return barrier_poly
end

# Create SOS polynomial function
function sos_polynomial(k::Vector{VariableRef}, var::Array{PolyVar{true},1}, k_count::Int64, lagrange_degree::Int64)::DynamicPolynomials.Polynomial{true, AffExpr}
    sos_polynomial::MonomialVector{true}  = monomials(var, 0:lagrange_degree)
    sos_poly_t = 0
    for sos in 1:Integer(length(sos_polynomial))
        sos_poly_t += k[sos + k_count*length(sos_polynomial)] * sos_polynomial[sos]
    end
    return sos_poly_t
end

# Function to compute number of decision variables per Lagrange function
function length_polynomial(var::Array{PolyVar{true},1}, degree::Int64)::Int64
    sos_polynomial::MonomialVector{true}  = monomials(var, 0:degree)
    length_polynomial::Int64 = length(sos_polynomial)
    return length_polynomial
end

# Function to add constraints to the model
function add_constraint_to_model(model::Model, expression::DynamicPolynomials.Polynomial{true, AffExpr})
    @constraint(model, expression >= 0)
end

# Function to compute the expecation and noise element
function expectation_noise(exp_evaluated::DynamicPolynomials.Polynomial{true, AffExpr}, barrier_degree::Int64, standard_deviation::Float64, z::PolyVar{true})

    exp_poly::DynamicPolynomials.Polynomial{true, AffExpr} = 0
    noise::DynamicPolynomials.Polynomial{true, AffExpr} = 0

    for zz in 1:length(exp_evaluated)
        z_occurs = occursin('z', string(exp_evaluated[zz]))

        if z_occurs == false
            exp_poly = exp_poly + exp_evaluated[zz]
        end

        if z_occurs == true
            for z_deg = 2:2:barrier_degree
                even_order_z = contains(string(exp_evaluated[zz]), "z^$z_deg")
                if even_order_z == true
                    exp_deep_evaluated::Term{true, AffExpr} = exp_evaluated[zz]
                    z_coefficients::Term{true, AffExpr} = subs(exp_deep_evaluated, z => 1)

                    noise_exp = z_coefficients * (doublefactorial(z_deg - 1) * standard_deviation^barrier_degree)
                    noise = noise + noise_exp
                end
            end
        end
    end
    return exp_poly, noise
end

# Compute the final barrier certificate
function barrier_certificate(barrier_monomial, c)

    # Control Barrier Certificate
    barrier_certificate = 0
    for cc in 1:Integer(length(barrier_monomial))
        barrier_certificate += value(c[cc]) * barrier_monomial[cc]
    end

    return barrier_certificate

end

# Control hypercube optimization 
function controller_convex(system_flag, x_star, system_dimension, identifier, controller_dimension, partitions, M_h_ii, M_l_ii, B_h_ii, B_l_ii, global_flag)

    # Optimize code later: only include controlled dimensions 

    # Initialize solver
    lp_control_model = Model()
    set_optimizer(lp_control_model, GLPK.Optimizer)

    # Create feedback law
    @variable(lp_control_model, theta[1:system_dimension])
    @variable(lp_control_model, x[1:system_dimension])
    @variable(lp_control_model, x_prime[1:system_dimension])
    @variable(lp_control_model, y[1:system_dimension])
    @variable(lp_control_model, y_prime[1:system_dimension])
    @variable(lp_control_model, u[1:system_dimension])

    # Argmin of barrier 
    # x_star = zeros(1, system_dimension)
    
    # Free variable bounds
    @constraint(lp_control_model, x .== y - y_prime)
    @constraint(lp_control_model, y .>= 0)
    @constraint(lp_control_model, y_prime .>= 0)

    if global_flag == true
        hyper_matrix_higher = M_h_ii
        hyper_matrix_lower = M_l_ii 
    else
        # Loop over hcube higher bound
        hyper_matrix_higher = M_h_ii * (y-y_prime) + B_h_ii
        
        # Loop over hcube lower bound
        hyper_matrix_lower = M_l_ii * (y-y_prime) + B_l_ii
    end

    # Control bounds
    if system_flag == "cartpole"
        u_min = -1
        u_max = 1
    else
        u_min = -1
        u_max = 1
    end

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

        # Controller logic
        if controller_dimension[ctrl] == 1
            @constraint(lp_control_model, u[ctrl] >= -1)
            @constraint(lp_control_model, u[ctrl] <= 1)
        else controller_dimension[ctrl] == 0
            @constraint(lp_control_model, u[ctrl] == 0)
        end

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

# Sum of squares optimization function
function optimization(number_hypercubes::Int64,
                      barrier_degree_input::Int64, 
                      controller_degree_input::Int64,
                      safety_threshold_eta::Float64,
                      safety_threshold_beta::Float64,
                      system_flag::String, 
                      neural_network_bound::String, 
                      epsilon_flag::String, 
                      model_flag::String,
                      layer_flag::String,
                      unsafety_flag::String,
                      objective_flag::String,
                      beta_partition::Bool, 
                      global_flag::Bool, 
                      eta_flag::Bool,
                      subset_flag::Bool,
                      large_range_initial::Bool,
                      print_to_txt::Bool,
                      control_flag::Bool, 
                      certificate_no,
                      eta_certificate,
                      x_star,
                      minimum_interferance::Bool, 
                      beta_vals_opt)

    # File reading
    filename = "/models/" * system_flag * "/" * neural_network_bound  * "/" * layer_flag* "_layers/partition_data_"  * string(number_hypercubes) * ".mat"

    layer_flag
    file = matopen(pwd()*filename)

    # Extract hypercube data (avoid using float64 for precision issues)
    partitions = read(file, "partitions")
    state_space = read(file, "state_space")

    # Number of hypercubes
    number_hypercubes_constraints = number_hypercubes
    hcube_identifier = 1:number_hypercubes
    
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

    # Define system and control dimensions
    system_dimension::Int64 = Integer(length(state_space[:,1]))

    if control_flag == true
        control_dimension::Int64 = Integer(2)
    end

    # Using Mosek as the SDP solver
    model = SOSModel(optimizer_with_attributes(Mosek.Optimizer,
                                               "MSK_DPAR_INTPNT_TOL_STEP_SIZE" => 1e-6,
                                               "MSK_IPAR_OPTIMIZER" => 0,
                                               "MSK_IPAR_BI_CLEAN_OPTIMIZER" => 0,
                                               "MSK_IPAR_NUM_THREADS" => 16,
                                               "MSK_IPAR_PRESOLVE_USE" => 0))

    # Create state space variables
    @polyvar x[1:system_dimension]

    # Create noise variable
    @polyvar z

    # Create global CROWN bounds variables
    @polyvar y[1:system_dimension]

    # Create dummy variable for beta in SOS
    @polyvar w[1:2]

    # Create probability decision variables eta  and beta
    if eta_flag == true
        @variable(model, eta)
    else
        eta = safety_threshold_eta
    end

    if beta_partition == true
        @variable(model, beta_parts_var[1:number_hypercubes_constraints])
        @variable(model, beta)
    else
        @variable(model, beta)
    end

    # Create barrier polynomial, specify degree Lagrangian polynomials
    barrier_degree::Int64 = barrier_degree_input
    controller_degree::Int64 = controller_degree_input
    alpha::Float64 = 1
    lagrange_degree::Int64 = 2

    # Specify noise element (Gaussian)
    standard_deviation::Float64 = 0.1

    # Specify initial state
    x_init::Array{Float64, 2} = zeros(1, system_dimension)

    # Placeholder for empty function return
    empty_return = zeros(1, system_dimension)

    # Create barrier candidate
    barrier_monomial::MonomialVector{true} = monomials(x, 0:barrier_degree)
    @variable(model, c[1:Integer(length(barrier_monomial))])
    BARRIER::DynamicPolynomials.Polynomial{true, AffExpr} = barrier_polynomial(c, barrier_monomial)

    # Add constraints to model for positive barrier, eta and beta
    add_constraint_to_model(model, BARRIER)
    if eta_flag == true
        @constraint(model, eta >= 1e-6)
        @constraint(model, eta <= safety_threshold_eta)
    end
    if beta_partition == true
        if control_flag == true
            for betas = 1:number_hypercubes_constraints
                @constraint(model, beta_parts_var[betas] >= 1e-6)
                if eta_flag == false
                    @constraint(model, beta_parts_var[betas] <= safety_threshold_beta)
                end
            end
            @constraint(model, beta >= 1e-6)
        else
            for betas = 1:number_hypercubes_constraints
                @constraint(model, beta_parts_var[betas] >= 1e-6)
                if eta_flag == false
                    @constraint(model, beta_parts_var[betas] <= (1 - 1e-6))
                end
            end
            @constraint(model, beta >= 1e-6)
        end
    else
        @constraint(model, beta >= 1e-6)
    end

    # One initial condition and unsafe conditions
    if system_flag == "cartpole" && large_range_initial == true 
        number_decision_vars = ((system_dimension)^2 + 1 + 1)*length(barrier_monomial)
    elseif system_flag == "husky" && large_range_initial == true 
        number_decision_vars = ((system_dimension)^2 + 1 + 1 + 1)*length(barrier_monomial)
    elseif system_flag == "husky5d" && large_range_initial == true 
        number_decision_vars = ((system_dimension)^2 + 1 + 1 + 1)*length(barrier_monomial)
    elseif system_flag == "acrobat" && large_range_initial == true 
        number_decision_vars = ((system_dimension)^2 + 1 + 1 + 1 + 1)*length(barrier_monomial)
    elseif system_flag == "pendulum" && large_range_initial == true 
        number_decision_vars = ((system_dimension)^2 + 1 + 1)*length(barrier_monomial)
    elseif system_flag == "pendulum3d" && large_range_initial == true 
        number_decision_vars = ((system_dimension)^2 + 1 + 1)*length(barrier_monomial)
    else
        number_decision_vars = ((system_dimension)^2 + 1)*length(barrier_monomial)
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
            if system_flag == "cartpole" && large_range_initial == true 
                lag_poly_theta::DynamicPolynomials.Polynomial{true, AffExpr} =  sos_polynomial(l::Vector{VariableRef}, x::Array{PolyVar{true},1}, (count_lag+1)::Int64, lagrange_degree::Int64)
                add_constraint_to_model(model, lag_poly_theta)
            elseif (system_flag == "pendulum" && large_range_initial == true) || (system_flag == "pendulum3d" && large_range_initial == true )
                lag_poly_theta_pen::DynamicPolynomials.Polynomial{true, AffExpr} =  sos_polynomial(l::Vector{VariableRef}, x::Array{PolyVar{true},1}, (count_lag+1)::Int64, lagrange_degree::Int64)
                add_constraint_to_model(model, lag_poly_theta_pen)
            elseif  (system_flag == "husky" || system_flag == "husky5d") && large_range_initial == true
                lag_poly_x1 =  sos_polynomial(l::Vector{VariableRef}, x::Array{PolyVar{true},1}, (count_lag+1)::Int64, lagrange_degree::Int64)
                lag_poly_x2 =  sos_polynomial(l::Vector{VariableRef}, x::Array{PolyVar{true},1}, (count_lag+2)::Int64, lagrange_degree::Int64)
            elseif  (system_flag == "acrobat") && large_range_initial == true
                lag_poly_x1 =  sos_polynomial(l::Vector{VariableRef}, x::Array{PolyVar{true},1}, (count_lag+1)::Int64, lagrange_degree::Int64)
                lag_poly_x2 =  sos_polynomial(l::Vector{VariableRef}, x::Array{PolyVar{true},1}, (count_lag+2)::Int64, lagrange_degree::Int64)
            end

            # Initial condition radius and ball
            x_initial_radius = (1e-8)
            x_initial_sums = x_initial_radius

            if system_flag == "cartpole" && large_range_initial == true 
                theta_radius = deg2rad(5.0)^2
                theta_initial_sums = x_initial_radius

                for jj = 1:length(x)
                    if jj != 3
                        x_initial_sums += -(x[jj] - x_init[jj])^2
                    else 
                        theta_initial_sums += -(x[jj] - x_init[jj])^2
                    end
                end
            
            elseif system_flag == "pendulum" && large_range_initial == true 
                theta_radius = deg2rad(5.0)^2
                theta_initial_sums = x_initial_radius

                for jj = 1:length(x)
                    if jj != 3
                        x_initial_sums += -(x[jj] - x_init[jj])^2
                    else 
                        theta_initial_sums += -(x[jj] - x_init[jj])^2
                    end
                end

            elseif system_flag == "pendulum3d" && large_range_initial == true 
                theta_radius = deg2rad(0.1)^2
                theta_initial_sums = x_initial_radius

                for jj = 1:length(x)
                    if jj != 3
                        x_initial_sums += -(x[jj] - x_init[jj])^2
                    else 
                        theta_initial_sums += -(x[jj] - x_init[jj])^2
                    end
                end

            elseif (system_flag == "husky" || system_flag == "husky5d") && large_range_initial == true 
                x1_initial_radius = 0.1^2
                x2_initial_radius = 0.1^2

                for jj = 1:length(x)
                    if jj == 1
                        x1_initial_radius += -(x[jj] - x_init[jj])^2
                    elseif jj == 2
                        x2_initial_radius += -(x[jj] - x_init[jj])^2    
                    else 
                        x_initial_sums += -(x[jj] - x_init[jj])^2
                    end
                end
                
            elseif (system_flag == "acrobat") && large_range_initial == true 
                x1_initial_radius = 0.1^2
                x2_initial_radius = 0.1^2

                for jj = 1:length(x)
                    if jj == 1
                        x1_initial_radius += -(x[jj] - x_init[jj])^2
                    elseif jj == 2
                        x2_initial_radius += -(x[jj] - x_init[jj])^2    
                    else 
                        x_initial_sums += -(x[jj] - x_init[jj])^2
                    end
                end    

            else
                for jj = 1:length(x)
                    x_initial_sums += -(x[jj] - x_init[jj])^2
                end
            end

            # Barrier constraint eta
            if system_flag == "cartpole" && large_range_initial == true
                _barrier_initial = - BARRIER + eta - lag_poly_i * x_initial_sums - lag_poly_theta*theta_initial_sums
            elseif (system_flag == "husky" || system_flag == "husky5d") && large_range_initial == true 
                _barrier_initial = - BARRIER + eta - lag_poly_i * x_initial_sums - lag_poly_x1*x1_initial_radius - lag_poly_x2*x2_initial_radius
            elseif (system_flag == "acrobat") && large_range_initial == true 
                _barrier_initial = - BARRIER + eta - lag_poly_i * x_initial_sums - lag_poly_x1*x1_initial_radius - lag_poly_x2*x2_initial_radius
            elseif system_flag == "pendulum" && large_range_initial == true
                _barrier_initial = - BARRIER + eta - lag_poly_i * x_initial_sums - lag_poly_theta_pen*theta_initial_sums
            elseif system_flag == "pendulum3d" && large_range_initial == true
                _barrier_initial = - BARRIER + eta - lag_poly_i * x_initial_sums - lag_poly_theta_pen*theta_initial_sums
            else
                _barrier_initial = - BARRIER + eta - lag_poly_i * x_initial_sums
                
                
                # x0_low::Float64 = -0.1
                # x0_up::Float64 = 0.1
                # _barrier_initial = - BARRIER + eta - lag_poly_i * (x0_up - x[1])*(x[1] - x0_low)
            end

            # Add constraint to model
            add_constraint_to_model(model, _barrier_initial)

        # Barrier unsafe region conditions (cartpole)
        elseif system_flag == "cartpole" 
            
            if ii == 1 || ii == 3  

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

        # Barrier unsafe region conditions (husky)
        elseif (system_flag == "husky" || system_flag == "husky5d")

            if ii == 1 || ii == 2

                # Generate sos polynomials
                if large_range_initial == true 
                    count_lag = 2*ii + 1
                else
                    count_lag = 2*ii
                end

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

        # Barrier unsafe region conditions (acrobat)
        elseif (system_flag == "acrobat")

            if ii == 2 || ii == 4

                # Generate sos polynomials
                if large_range_initial == true 
                    count_lag = 2*ii + 1
                else
                    count_lag = 2*ii
                end

                lag_poly_i_lower =  sos_polynomial(l::Vector{VariableRef}, x::Array{PolyVar{true},1}, (count_lag - 1)::Int64, lagrange_degree::Int64)
                lag_poly_i_upper =  sos_polynomial(l::Vector{VariableRef}, x::Array{PolyVar{true},1}, (count_lag)::Int64, lagrange_degree::Int64)

                # # State space ranges
                # if ii == 5
                #     x_i_lower = -2/9*pi
                #     x_i_upper = 2/9*pi
                # elseif ii == 6
                #     x_i_lower = -9/9*pi
                #     x_i_upper = 9/9*pi
                # end

                # State space ranges
                x_i_lower = -0.6
                x_i_upper = 0.6

                # Specify constraints for initial and unsafe set
                _barrier_unsafe_lower = BARRIER - lag_poly_i_lower * (x_i_lower - x[ii]) - 1
                _barrier_unsafe_upper = BARRIER - lag_poly_i_upper * (x[ii] - x_i_upper) - 1

                # Add constraints to model
                add_constraint_to_model(model, lag_poly_i_lower)
                add_constraint_to_model(model, lag_poly_i_upper)
                add_constraint_to_model(model, _barrier_unsafe_lower)
                add_constraint_to_model(model, _barrier_unsafe_upper)

            end


        # Barrier unsafe region conditions (twodim)
        elseif system_flag == "pendulum" || system_flag == "pendulum3d"
    
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
            
        # Barrier unsafe region conditions (twodim)
        elseif system_flag == "twodim" 
            
            if ii == 1 || ii == 2

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
    lagrange_monomial_length::Int64 = length_polynomial(x::Array{PolyVar{true},1}, lagrange_degree::Int64)
    number_of_variables_exp::Int64 = number_hypercubes_constraints * (system_dimension) * lagrange_monomial_length
    @variable(model, g[1:number_of_variables_exp])
    @variable(model, h[1:number_of_variables_exp])
    # @variable(model, pol[1:12])

    # Partition beta to extract ith beta values
    if beta_partition == true

        # Variables for beta in SOS
        num_vars_beta_lagrangian = number_hypercubes_constraints * lagrange_monomial_length
        @variable(model, delta[1:num_vars_beta_lagrangian])

        # Number of constraints
        number_constraints_per_loop = (2*system_dimension) + 1 + 1 + 1
        constraints = Array{DynamicPolynomials.Polynomial{true, AffExpr}}(undef, number_hypercubes_constraints, number_constraints_per_loop)

    else

        number_constraints_per_loop = (2*system_dimension) + 1
        constraints = Array{DynamicPolynomials.Polynomial{true, AffExpr}}(undef, number_hypercubes_constraints, number_constraints_per_loop)

    end

    # Counters
    counter_lag::Int64 = 0
    parts_count::Int64 = 0
    counter_beta::Int64 = 0
    count_bad::Int64 = 0
        
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

            # Define global or explicit upper and lower bound for kth dimension of partition parts
            if global_flag == true
                y_k_upper_global::Float64 = G_ub[identifier, kk]
                y_k_lower_global::Float64 = G_lb[identifier, kk]
            else
                M_h_ii = transpose(M_h[identifier, :, :])
                M_l_ii = transpose(M_l[identifier, :, :])
                B_h_ii = B_h[identifier, :]
                B_l_ii = B_l[identifier, :]

                # Loop over hcube higher bound
                hyper_matrix_higher = M_h_ii * x + B_h_ii
                y_k_upper_explicit::DynamicPolynomials.Polynomial{true,Float32} = hyper_matrix_higher[kk]

                # Loop over hcube lower bound
                hyper_matrix_lower = M_l_ii * x + B_l_ii
                y_k_lower_explicit::DynamicPolynomials.Polynomial{true,Float32} = hyper_matrix_lower[kk]
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

    # Return beta values
    eta_val = value(eta)
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
    if print_to_txt == true
        if control_flag == true
            
            if isfile("probabilities/probs_cont_"*system_flag*string(number_hypercubes)*".txt") == true
                rm("probabilities/probs_cont_"*system_flag*string(number_hypercubes)*".txt")
            end

            open("probabilities/probs_cont_"*system_flag*string(number_hypercubes)*".txt", "a") do io
            println(io, "eta = $(value(eta)), beta = $(value(beta)), total = $(value(eta) + value(beta)) ")
            end

            if beta_partition == true

                # Print beta values to txt file
                if isfile("probabilities/beta_vals_controller.txt") == true
                    rm("probabilities/beta_vals_controller.txt")
                end

                open("probabilities/beta_vals_controller.txt","a") do io
                    println(io, beta_values)
                end

            end

        else

            if isfile("probabilities/probs_cert_"*system_flag*string(number_hypercubes)*".txt") == true
                rm("probabilities/probs_cert_"*system_flag*string(number_hypercubes)*".txt")
            end

            open("probabilities/probs_cert_"*system_flag*string(number_hypercubes)*".txt", "a") do io
            println(io, "eta = $(value(eta)), beta = $(value(beta)), total = $(value(eta) + value(beta)) ")
            end

            if beta_partition == true

                # Print beta values to txt file
                if isfile("probabilities/beta_vals_certificate.txt") == true
                    rm("probabilities/beta_vals_certificate.txt")
                end

                open("probabilities/beta_vals_certificate.txt","a") do io
                    println(io, beta_values)
                end
            end
        end
    end

    # Return optimization results
    if beta_partition == true
        if control_flag == true
            return certificate, count_bad
        else
            return certificate,  eta_val, beta_values
        end
    else
        return certificate
    end

end


# Sum of squares optimization function
function control_loop(number_hypercubes::Int64,
                      barrier_degree_input::Int64, 
                      controller_degree_input::Int64,
                      safety_threshold_eta::Float64,
                      safety_threshold_beta::Float64,
                      system_flag::String, 
                      neural_network_bound::String, 
                      epsilon_flag::String, 
                      model_flag::String,
                      layer_flag::String,
                      unsafety_flag::String,
                      objective_flag::String,
                      beta_partition::Bool, 
                      global_flag::Bool, 
                      eta_flag::Bool,
                      subset_flag::Bool,
                      large_range_initial::Bool,
                      print_to_txt::Bool,
                      control_flag::Bool, 
                      certificate,
                      eta_certificate,
                      x_star,
                      minimum_interferance::Bool, 
                      beta_vals_opt)

    # File reading
    filename = "/models/" * system_flag * "/" * neural_network_bound  * "/" * layer_flag* "_layers/partition_data_"  * string(number_hypercubes) * ".mat"
    file = matopen(pwd()*filename)

    # Extract hypercube data (avoid using float64 for precision issues)
    partitions = read(file, "partitions")
    state_space = read(file, "state_space")

    # Number of hypercubes
    number_hypercubes_constraints = number_hypercubes
    hcube_identifier = 1:number_hypercubes
    
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

    # Define system and control dimensions
    system_dimension::Int64 = Integer(length(state_space[:,1]))

    # Using Mosek as the SDP solver
    model = SOSModel(optimizer_with_attributes(Mosek.Optimizer,
                                               "MSK_DPAR_INTPNT_TOL_STEP_SIZE" => 1e-6,
                                               "MSK_IPAR_OPTIMIZER" => 0,
                                               "MSK_IPAR_BI_CLEAN_OPTIMIZER" => 0,
                                               "MSK_IPAR_NUM_THREADS" => 16,
                                               "MSK_IPAR_PRESOLVE_USE" => 0))

    # Create state space variables
    @polyvar x[1:system_dimension]

    # Create noise variable
    @polyvar z

    # Create global CROWN bounds variables
    @polyvar y[1:system_dimension]

    # Create dummy variable for beta in SOS
    @polyvar w[1:2]

    # Create probability decision variables eta  and beta
    eta = eta_certificate
    if beta_partition == true
        @variable(model, beta_parts_var[1:number_hypercubes_constraints])
        @variable(model, beta)
    else
        @variable(model, beta)
    end

    # Create barrier polynomial, specify degree Lagrangian polynomials
    alpha::Float64 = 1
    lagrange_degree::Int64 = 2

    # Specify noise element (Gaussian)
    standard_deviation::Float64 = 0.1

    # Barrier
    BARRIER = certificate

    # Add constraints to model for positive barrier, eta and beta
    if beta_partition == true
        if control_flag == true
            for betas = 1:number_hypercubes_constraints
                @constraint(model, beta_parts_var[betas] >= 1e-6)
            end
            @constraint(model, beta >= 1e-6)
        else
            for betas = 1:number_hypercubes_constraints
                @constraint(model, beta_parts_var[betas] >= 1e-6)
            end
            @constraint(model, beta >= 1e-6)
            
        end
    else
        @constraint(model, beta >= 1e-6)
    end

    # Variables g and h for Lagrange multipliers
    lagrange_monomial_length::Int64 = length_polynomial(x::Array{PolyVar{true},1}, lagrange_degree::Int64)
    number_of_variables_exp::Int64 = number_hypercubes_constraints * (system_dimension) * lagrange_monomial_length
    @variable(model, g[1:number_of_variables_exp])
    @variable(model, h[1:number_of_variables_exp])

    # Partition beta to extract ith beta values
    if beta_partition == true

        # Variables for beta in SOS
        num_vars_beta_lagrangian = number_hypercubes_constraints * lagrange_monomial_length
        @variable(model, delta[1:num_vars_beta_lagrangian])

        # Number of constraints
        number_constraints_per_loop = (2*system_dimension) + 1 + 1 + 1
        constraints = Array{DynamicPolynomials.Polynomial{true, AffExpr}}(undef, number_hypercubes_constraints, number_constraints_per_loop)

    else

        number_constraints_per_loop = (2*system_dimension) + 1
        constraints = Array{DynamicPolynomials.Polynomial{true, AffExpr}}(undef, number_hypercubes_constraints, number_constraints_per_loop)

    end

    # Counters
    counter_lag::Int64 = 0
    parts_count::Int64 = 0
    counter_beta::Int64 = 0
    count_bad::Int64 = 0
    feedback_control_store = zeros(system_dimension, number_hypercubes)

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
        
   
        # Call on feedback law
        if control_flag == true
            if system_flag == "cartpole"
                control_dimensions = [0, 1, 0, 1]
            elseif system_flag == "husky"
                control_dimensions = [0, 0, 1, 1]
            elseif system_flag == "husky5d"
                control_dimensions = [0, 0, 0, 1, 1]
            elseif system_flag == "acrobat"
                control_dimensions = [0, 0, 0, 0, 1, 1]
            elseif system_flag == "pendulum"
                control_dimensions = [0 ,1]
            elseif system_flag == "twodim"
                control_dimensions = [1, 1]
            else
                print("System not defined ...")
            end
            if beta_vals_opt[parts] >= safety_threshold_beta
                count_bad += 1
                if global_flag == true
                    feedback_control = controller_convex(system_flag, x_star, system_dimension, identifier, control_dimensions, partitions, G_ub, G_lb, false, false, global_flag)
                    feedback_control_store[:, parts] = feedback_control
                else
                    feedback_control = controller_convex(system_flag, x_star, system_dimension, identifier, control_dimensions, partitions, M_h_ii, M_l_ii, B_h_ii, B_l_ii, global_flag)
                    feedback_control_store[:, parts] = feedback_control
                end
            end
        end

        # Loop of state space and neural network bounds
        for kk = 1:system_dimension

            # Partition bounds
            x_k_hcube_bound::Vector{Float64} = partitions[identifier, :, kk]
            x_k_lower::Float64 = x_k_hcube_bound[1, 1]
            x_k_upper::Float64 = x_k_hcube_bound[2, 1]

            # Define global or explicit upper and lower bound for kth dimension of partition parts
            if global_flag == true
                y_k_upper_global::Float64 = G_ub[identifier, kk]
                y_k_lower_global::Float64 = G_lb[identifier, kk]
            else
                M_h_ii = transpose(M_h[identifier, :, :])
                M_l_ii = transpose(M_l[identifier, :, :])
                B_h_ii = B_h[identifier, :]
                B_l_ii = B_l[identifier, :]

                # Loop over hcube higher bound
                hyper_matrix_higher = M_h_ii * x + B_h_ii
                y_k_upper_explicit::DynamicPolynomials.Polynomial{true,Float32} = hyper_matrix_higher[kk]

                # Loop over hcube lower bound
                hyper_matrix_lower = M_l_ii * x + B_l_ii
                y_k_lower_explicit::DynamicPolynomials.Polynomial{true,Float32} = hyper_matrix_lower[kk]

            end

            if control_flag == true
                if beta_vals_opt[parts] >= safety_threshold_beta
                    if global_flag == true
                        y_k_upper_global += feedback_control[kk]
                        y_k_lower_global += feedback_control[kk]
                    else
                        y_k_upper_explicit += feedback_control[kk]
                        y_k_lower_explicit += feedback_control[kk]
                    end
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
        barrier_degree = barrier_degree_input
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
    @objective(model, Min, beta*time_horizon) 
    if objective_flag == "min_max"
        @objective(model, Min, beta*time_horizon) 
    elseif objective_flag == "sum"
        @objective(model, Min, sum(beta_parts_var)/(number_hypercubes)*time_horizon)
    elseif objective_flag == "min_max_sum"
        @objective(model, Min, (beta + sum(beta_parts_var)/(number_hypercubes))*time_horizon)
    end
    print("Objective made\n")

    # Print number of partitions
    print("\n", "Optimizing for number of partitions = " * string(number_hypercubes), "\n")

    # Optimize model
    optimize!(model)

    # Return beta values
    if beta_partition == true
        beta_values = value.(beta_parts_var)
        max_beta = maximum(beta_values)

        # Print probability values
        println("Solution: [eta = $(value(eta)), beta = $(value(max_beta)), total = $(value(eta) + value(max_beta)) ]")
        # end

        # Print controller values to mat file
        if isfile("controllers/controller_magnitudes.txt") == true
            rm("controllers/controller_magnitudes.txt")
        end

        open("controllers/controller_magnitudes.txt","a") do io
            println(io, feedback_control_store)
        end


        # Print beta values to txt file
        if isfile("probabilities/beta_vals_controller.txt") == true
            rm("probabilities/beta_vals_controller.txt")
        end

        open("probabilities/beta_vals_controller.txt","a") do io
            println(io, beta_values)
        end

    else

        # Print probability values
        println("Solution: [eta = $(value(eta)), beta = $(value(beta)), total = $(value(eta) + value(beta)) ]")
        
    end

    # Return optimization results
    if beta_partition == true
        if control_flag == true
            return certificate, count_bad
        else
            return certificate, beta_values
        end
    else
        return certificate
    end

end

# Optimization flags
# system_flag =  "cartpole"
# neural_network_bound = "alpha" # "lirpa" #"crown" 
# epsilon_flag =  "variable" # "fixed"
# model_flag = # "Model5" #"" 
# layer_flag = "2"
# unsafety_flag = "outside" #"none"  ## "obstacle"  # "both "
# objective_flag = "min_max" #"min_max_sum" # "sum"
# beta_partition = true
# global_flag = true
# decision_eta_flag = true
# subset_flag = false
# large_range_initial = true
# print_to_txt = false

# # Optimize certificate
# system_dimension = 4
# number_hypercubes = 3840
# barrier_degree_input = 4
# controller_degree_input = 4
# safety_threshold_eta = (1 - 1e-6) # norm = (1 - 1e-6)
# safety_threshold_beta = 0.05

# Optimization flags
system_flag =  "husky5d"
neural_network_bound =  "alpha" #"lirpa" #"crown" 
epsilon_flag =  "variable" # "fixed"
model_flag = "Model5" #"" 
layer_flag = "1"
unsafety_flag = "outside" #"none"  ## "obstacle"  # "both "
objective_flag = "min_max" #"min_max_sum" #"sum" 
beta_partition = true
global_flag = false
decision_eta_flag = true
subset_flag = false
large_range_initial = true
adjust_barrier_monomial = false
print_to_txt = true

# Optimize certificate
system_dimension = 5
number_hypercubes = 1728
barrier_degree_input = 4
controller_degree_input = 4
safety_threshold_eta = 0.049 #(1 - 1e-6) # norm = (1 - 1e-6)
safety_threshold_beta = 10.0

# # Optimization flags
# system_flag = "pendulum"
# neural_network_bound = "alpha" #"lirpa" #"crown" 
# epsilon_flag =  "" #"variable" # "fixed"
# model_flag = "2d" #"Model5" #"3d" 
# layer_flag = "2"
# unsafety_flag = "outside" #"none"  ## "obstacle"  # "both "
# objective_flag = "min_max" #"min_max_sum" # "sum"
# beta_partition = true
# global_flag = false
# decision_eta_flag = true
# subset_flag = false
# large_range_initial = true
# print_to_txt = true

# # Optimize certificate
# number_hypercubes = 480
# barrier_degree_input = 4
# controller_degree_input = 4
# safety_threshold_eta = (1 - 1e-6) # norm = (1 - 1e-6)
# safety_threshold_beta = 0.05

# Optimization flags
# system_flag = "pendulum"
# neural_network_bound = "alpha" #"lirpa" #"crown" 
# epsilon_flag =  "" #"variable" # "fixed"
# model_flag = "2d" #"Model5" #"3d" 
# layer_flag = "5"
# unsafety_flag = "outside" #"none"  ## "obstacle"  # "both "
# objective_flag = "min_max" #"min_max_sum" # "sum"
# beta_partition = true
# global_flag = true
# decision_eta_flag = false
# subset_flag = false
# large_range_initial = true
# print_to_txt = true

# # Optimize certificate
# system_dimension = 2
# number_hypercubes = 1920
# barrier_degree_input = 4
# controller_degree_input = 4
# safety_threshold_eta = 0.049 #(1 - 1e-6) # norm = (1 - 1e-6)
# safety_threshold_beta = 0.05

# Two dimensional system
# system_flag =  "twodim"
# neural_network_bound =  "alpha" 
# epsilon_flag = "fixed"
# model_flag = "" 
# layer_flag = "1"
# unsafety_flag = "outside" #"none"  ## "obstacle"  # "both "
# objective_flag = "min_max" #"min_max_sum" # "sum"
# beta_partition = true
# global_flag = false
# decision_eta_flag = true
# subset_flag = false
# large_range_initial = false
# print_to_txt = true

# # Optimize certificate
# number_hypercubes = 1225
# barrier_degree_input = 2
# controller_degree_input = 2
# safety_threshold_eta = (1 - 1e-6)
# safety_threshold_beta = 0.02

# Optimization flags
# system_flag =  "acrobat"
# neural_network_bound =  "alpha" #"lirpa" #"crown" 
# epsilon_flag =  "variable" # "fixed"
# model_flag = "Model5" #"" 
# layer_flag = "1"
# unsafety_flag = "outside" #"none"  ## "obstacle"  # "both "
# objective_flag = "min_max" #"min_max_sum" #"sum" 
# beta_partition = true
# global_flag = false
# decision_eta_flag = true
# subset_flag = false
# large_range_initial = true
# adjust_barrier_monomial = false
# print_to_txt = true

# # Optimize certificate
# system_dimension = 6
# number_hypercubes = 400
# barrier_degree_input = 4
# controller_degree_input = 4
# safety_threshold_eta = 1e-6 #(1 - 1e-6) # norm = (1 - 1e-6)
# safety_threshold_beta = 0.049

# Optimize certificate
@time certificate,  eta_certificate, certificate_beta_vals = optimization(number_hypercubes,
                                                                        barrier_degree_input,
                                                                        controller_degree_input,
                                                                        safety_threshold_eta,
                                                                        safety_threshold_beta,
                                                                        system_flag,
                                                                        neural_network_bound,
                                                                        epsilon_flag,
                                                                        model_flag,
                                                                        layer_flag,
                                                                        unsafety_flag,
                                                                        objective_flag,
                                                                        beta_partition,
                                                                        global_flag,
                                                                        decision_eta_flag,
                                                                        subset_flag,
                                                                        large_range_initial,
                                                                        print_to_txt,
                                                                        false,
                                                                        false,
                                                                        false,
                                                                        false,
                                                                        false,
                                                                        false)

# Find the argmin of the barrier 
h = Meta.parse(string(certificate))
res = optimize((@eval x -> $h),  zeros(system_dimension))
x_star = Optim.minimizer(res)

# # Optimize controller
minimum_interferance = true
@time controller, counts = control_loop(number_hypercubes,
                                        barrier_degree_input,
                                        controller_degree_input,
                                        safety_threshold_eta,
                                        safety_threshold_beta,
                                        system_flag,
                                        neural_network_bound,
                                        epsilon_flag,
                                        model_flag,
                                        layer_flag,
                                        unsafety_flag,
                                        objective_flag,
                                        beta_partition,
                                        global_flag,
                                        decision_eta_flag,
                                        subset_flag,
                                        large_range_initial,
                                        print_to_txt,
                                        true,
                                        certificate,
                                        eta_certificate,
                                        x_star,
                                        minimum_interferance,
                                        certificate_beta_vals)



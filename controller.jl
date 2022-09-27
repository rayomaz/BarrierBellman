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

# Create Control Barrier Polynomial
function barrier_polynomial(c::Vector{VariableRef}, barrier_monomial::MonomialVector{true})::DynamicPolynomials.Polynomial{true, AffExpr}
    barrier_poly = 0
    for cc in 1:Integer(length(barrier_monomial))
        barrier_poly += c[cc] * barrier_monomial[cc]
    end
    return barrier_poly
end

# Create SOS polynomial function
function sos_polynomial(k::Vector{VariableRef}, var, k_count::Int64, lagrange_degree::Int64)::DynamicPolynomials.Polynomial{true, AffExpr}

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
function expectation_noise(exp_evaluated, barrier_degree::Int64, standard_deviation::Float64, z::PolyVar{true})

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

# Sum of squares optimization function
function controlfunc(number_hypercubes::Int64,
                    x_init::Float64,
                    controller_degree_input::Int64,
                    system_dimension::Int64,
                    control_dimension::Int64)

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

    # Create control variable
    @polyvar u

    # Create probability decision variables eta  and beta
    @variable(model, beta)
    @variable(model, eta)

    # Create barrier polynomial, specify degree Lagrangian polynomials
    controller_degree::Int64 = controller_degree_input
    alpha::Float64 = 1
    lagrange_degree::Int64 = 2

    # Specify noise element (Gaussian)
    standard_deviation::Float64 = 0.01

    # Create barrier candidate
    cv = [0.01529731507423992; - 1.408487700689928; 49.11009976365797; - 766.8062132094691; 4514.598436659872]
    # cv = [0.191; -16.478; 532.641; -7.651e3; 4.121e4]
    BARRIER = cv[1]*x[1]^4 + cv[2]*x[1]^3 + cv[3]*x[1]^2 + cv[4]*x[1] + cv[5]

    # Create controller candidate
    controller_monomial::MonomialVector{true} = monomials(x, 0:controller_degree)
    @variable(model, d1[1:Integer(length(controller_monomial)*number_hypercubes)])
    decision_vars_controller = length(controller_monomial)

    # Obtain controller expression
    monomial_string = Vector{String}()
    for mon = 1:length(controller_monomial)
        push!(monomial_string, string(controller_monomial[mon]))
    end

    # Add constraints to model for positive barrier, eta and beta
    @constraint(model, eta >= 1e-6)
    @constraint(model, eta <= (1 - 1e-6))
    @constraint(model, beta >= 1e-6)

    # One initial condition and unsafe conditions
    number_decision_vars = ((system_dimension)^2 + 1)*length(BARRIER)
    @variable(model, l[1:number_decision_vars])

    barrier_constraints_unsafe_initial = system_dimension + 1

    for ii = 1:barrier_constraints_unsafe_initial

        # Barrier initial condition f(eta)
        if ii == barrier_constraints_unsafe_initial

            # Generate sos polynomial
            count_lag = 0

            # Optimize this code: not all x variables needed in lag_poly_i, lag_poly_theta (only 1 2 4 and 3, respectively)
            lag_poly_i::DynamicPolynomials.Polynomial{true, AffExpr} =  sos_polynomial(l::Vector{VariableRef}, x, count_lag::Int64, lagrange_degree::Int64)

            add_constraint_to_model(model, lag_poly_i)
            
            # Initial condition radius and ball
            x_initial_radius = 0.5^2
            x_initial_sums = x_initial_radius

            for jj = 1:length(x)
                x_initial_sums += -(x[jj] - x_init[jj])^2
            end

            # Barrier constraint ete
            _barrier_initial = - BARRIER + eta - lag_poly_i * x_initial_sums

            # Add constraint to model
            add_constraint_to_model(model, _barrier_initial)
   
        else

            # Generate sos polynomials
            lag_poly_i_lower =  sos_polynomial(l::Vector{VariableRef}, x, (1)::Int64, lagrange_degree::Int64)
            lag_poly_i_upper =  sos_polynomial(l::Vector{VariableRef}, x, (2)::Int64, lagrange_degree::Int64)

            # State space ranges
            x_i_lower_min = 0
            x_i_lower_max = 20

            x_i_upper_min = 23
            x_i_upper_max = 45

            # Specify constraints for initial and unsafe set
            _barrier_unsafe_lower = BARRIER - lag_poly_i_lower * (x_i_lower_max - x[ii])*(x[ii] - x_i_lower_min) - 1
            _barrier_unsafe_upper = BARRIER - lag_poly_i_upper * (x_i_upper_max - x[ii])*(x[ii] - x_i_upper_min) - 1
        
            # Add constraints to model
            add_constraint_to_model(model, lag_poly_i_lower)
            add_constraint_to_model(model, lag_poly_i_upper)
            add_constraint_to_model(model, _barrier_unsafe_lower)
            add_constraint_to_model(model, _barrier_unsafe_upper)

        end

    end

    # Variables g and h for Lagrange multipliers
    lagrange_monomial_length::Int64 = length_polynomial(x, lagrange_degree::Int64)
    number_of_variables_exp::Int64 = number_hypercubes * (system_dimension^2) * lagrange_monomial_length
    @variable(model, g[1:number_of_variables_exp])

    # Number of constraints per loop
    number_constraints_per_loop = (system_dimension) + 1
    constraints = Array{DynamicPolynomials.Polynomial{true, AffExpr}}(undef, number_hypercubes, number_constraints_per_loop)

    for parts = 1

        # Create SOS polynomials for X (Partition) and Y (Bounds)
        hCubeSOS_X::DynamicPolynomials.Polynomial{true, AffExpr} = 0

        # Loop of state space and neural network bounds
        for kk = 1:system_dimension

            # Partition bounds
            x_k_lower::Float64 = 20
            x_k_upper::Float64 = 23

            # Generate Lagrange polynomial for kth dimension
            lag_poly_X::DynamicPolynomials.Polynomial{true, AffExpr} = sos_polynomial(g::Vector{VariableRef}, x, (0)::Int64, lagrange_degree::Int64)

            # Add Lagrange polynomial to constraints vector for the state space
            constraints[1, kk] = lag_poly_X

            # Generate SOS polynomials for state space
            hCubeSOS_X::DynamicPolynomials.Polynomial{true, AffExpr} += lag_poly_X*(x_k_upper - x[kk])*(x[kk] - x_k_lower)

        end

        # Compute expectation
   
        # Affine controller loop
        Ts = 5
        Th = 55
        Te = 15
        ae = 8 * 10^-3
        aH = 3.6 * 10^-3

        # CONTROLLER_1 = barrier_polynomial(d1[1:decision_vars_controller], controller_monomial)

        # print("\n", CONTROLLER_1)

        # return 0,0

        y_func = x[1] + Ts*( ae * (Te - x[1]) + aH * (Th - x[1]) *  u) + 0.1*z

        exp_evaluated = subs(BARRIER, x[1] => y_func)

        # Extract noise term
        exp_poly, noise = expectation_noise(exp_evaluated, 4, standard_deviation::Float64, z::PolyVar{true})

        # Full expectation term
        exp_current = exp_poly + noise

        print("\n", exp_current)

        return 0,0 

        # Constraint for hypercube
        hyper_constraint = - exp_current + (BARRIER/alpha) - (u - CONTROLLER_1) + beta - hCubeSOS_X

        constraints[1, 2] = hyper_constraint

        @constraint(model, CONTROLLER_1 == u)

        add_constraint_to_model(model, CONTROLLER_1)
        add_constraint_to_model(model, - CONTROLLER_1 + 1)

        #add_constraint_to_model(model, u[1] - CONTROLLER_1)

        # add_constraint_to_model(model, CONTROLLER_1 - u[1])
        #  


    end
    
    # Add constraints to model as a vector of constraints
    @time begin
        @constraint(model, constraints .>= 0)
    end
    print("Constraints made\n")

    # Define optimization objective
    time_horizon = 1
    @objective(model, Min, eta + beta*time_horizon)
    print("Objective made\n")

    # Print number of partitions
    print("\n", "Optimizing for number of partitions = " * string(number_hypercubes), "\n")

    # Optimize model
    optimize!(model)

    # Barrier certificate
    certificate = BARRIER

    # Print probability values
    println("Solution: [eta = $(value(eta)), beta = $(value(beta)), total = $(value(eta) + value(beta)) ]")

    # Print controller polynomial to txt file
    control_vars_1 = value.(d1)

    if isfile("controllers/control_lag_1.txt") == true
        rm("controllers/control_lag_1.txt")
    end

    open("controllers/control_lag_1.txt","a") do io
        println(io, control_vars_1)
    end

    # Return optimization results
    return certificate

end

# Optimize certificate
number_hypercubes = 1
initial_state = 21.5
system_dimension = 1
control_dimension = 1
controller_degree_input = 4

# Optimize certificate
@time control_poly = controlfunc(number_hypercubes,
                                initial_state,
                                controller_degree_input,
                                system_dimension,
                                control_dimension)






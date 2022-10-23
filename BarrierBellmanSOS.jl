# Piecewise Barrier Functions based on Bellman's Equation
# Rayan Mazouz, University of Colorado Boulder, Aerospace Engineering Sciences

# Initialization Statement
print("\n Computing Piecewise Barriers based on Bellman Equation")

# Import packages
using SumOfSquares
using DynamicPolynomials
using MosekTools
using Polynomials
using LinearAlgebra
using Random
using SpecialFunctions
using PyCall
using NLopt
using Plots

# Not used
# using MAT
# using StatsBase
# using Combinatorics
# using GLPK
# using Optim
# using Meshes

# Create Linear Barrier Function
function barrier_construct(system_dimension, A, b, x) #::DynamicPolynomials.Polynomial{true, AffExpr}

    barrier_linear = b
    for ii = 1:system_dimension
        barrier_linear += A[ii]*x[ii]
    end

    return barrier_linear
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
function add_constraint_to_model(model::Model, expression)
    @constraint(model, expression >= 0)
end

# Compute the final barrier certificate
function barrier_certificate(system_dimension, A, b, x)

    # Control Barrier Certificate
    barrier_certificate = value(b)
    for ii = 1:system_dimension
        barrier_certificate += value(A[ii])*x[ii]
    end

    return barrier_certificate

end

# Function: partition a space
function partition_space(state_space, partitions_eps)

    # py"""
    # def sumMyArgs (i, j):
    #     return i+j
    # def getNElement (n):
    #     a = [0,1,2,3,4,5,6,7,8,9]
    #     return a[n]
    # """

    # a = py"sumMyArgs"(3,4)          # 7
    # b = py"sumMyArgs"([3,4],[5,6])  # [8,10]

    # pyimport(utilities)

    hypercubes = [[[-3, 0], [0, 3]],
                  [[0, 3], [0, 3]],
                  [[0, 3], [0, -3]],
                  [[-3, 0], [-3, 0]]]

    return hypercubes

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

# Covariance matrix
function random_covariance_matrix(n::Int)

    # Set random seed
    Random.seed!(1234)

    # Create positive definite symmetric matrix
    x = randn(n, n)
    covariance_matrix = 0.5(x + x') + n*I

    return covariance_matrix 
end

# Transformation function (based on proposition 1, http://dx.doi.org/10.1145/3302504.3311805)
function transformation(covariance_matrix)

    eigenvalues = eigvals(covariance_matrix)
    eigenvector_matrix = eigvecs(covariance_matrix)

    Gamma_a = Diagonal(eigenvalues)
    V_a = eigenvector_matrix

    T_a = (Gamma_a^(-1/2))*(transpose(V_a))

    return T_a

end

function system_dynamics(x, T_a, system_flag, neural_flag)

    if system_flag == "population_growth"

        # System properties
        m1 = 0.50
        m2 = 0.95
        m3 = 0.50

        # Define if system is Neural Network
        if neural_flag == true
            print("Complete code for Neural Network Systems ....")
            return 0
        end

        # Dynamics definion
        F_a = [0 m3; m1 m2]
        x_next_step = F_a*x

        # Modified hyperspace (see Eq. 17: http://dx.doi.org/10.1145/3302504.3311805)
        y = T_a*x_next_step

        return y

    end

end

# Function vertices: plot hypercube posterior
function hypercube_posterior(hypercubes, probability_distribution)

    print("hi")

end

# Transition probability, T(q | x, a), based on proposition 1, http://dx.doi.org/10.1145/3302504.3311805
function probability_distribution(hypercubes, covariance_matrix, system_dimension, system_flag, neural_flag)

    # Initiate transformation function
    T_a = transformation(covariance_matrix)

    # Test if proper matrix
    proper_test = Matrix(1.0I, system_dimension, system_dimension) - T_a*covariance_matrix*transpose(T_a)
    proper_threshold = 1e-9
    proper_elements_test = length(proper_test[proper_test .> proper_threshold])

    if proper_elements_test > 0
        print("\n", "Error: this is not a proper matrix, violation of Proposition 1 in http://dx.doi.org/10.1145/3302504.3311805")
        return 0
    end

    # Gradient descent on log-concave function: 
    model_gradient = Model(NLopt.Optimizer)
    set_optimizer_attribute(model_gradient, "algorithm", :LD_MMA)

    # Define y variable: altered hyperspace
    @variable(model_gradient, x_gradient[1:system_dimension])
    y = system_dynamics(x_gradient, T_a, system_flag, neural_flag)

    # Loop for f(y, q), Proposition 3, http://dx.doi.org/10.1145/3302504.3311805
    # Product of erf functions to obtain probability
    P_vector = []
    vector_erf_vars = []
    m = system_dimension
    hyper = length(hypercubes)

    for qq = 1:hyper

        # Identify current hypercube
        hypercube_j = hypercubes[qq]

        for tt = 1:m

            # Hypercube bounds
            v_l = hypercube_j[tt][1]
            v_u = hypercube_j[tt][2]

            # Add constraints to original hypercube dimensions
            @constraint(model_gradient, x_gradient[tt] >= v_l)
            @constraint(model_gradient, x_gradient[tt] <= v_u)

            # Set start value at the middle of the hypercube
            set_start_value(x_gradient[tt], 1/2 * (v_l + v_u))

            # Error function range
            y_i = y[tt]
            erf_func_lo = (y_i - v_l) / sqrt(2) 
            erf_func_up = (y_i - v_u) / sqrt(2) 

            vector_erf_vars = push!(vector_erf_vars, [erf_func_lo, erf_func_up])
        
        end

        # Objective function to be minimized
        @NLobjective(model_gradient, Min, (1/(2^m))*prod(erf(vector_erf_vars[i][1] - vector_erf_vars[i][2]) for i in 1:m))

        # Optimize for minimum
        JuMP.optimize!(model_gradient)
        P_min = JuMP.objective_value(model_gradient)

        # Objective function to be maximized
        @NLobjective(model_gradient, Max, (1/(2^m))*prod(erf(vector_erf_vars[i][1] - vector_erf_vars[i][2]) for i in 1:m))

        # Optimize for maximum
        JuMP.optimize!(model_gradient)
        P_max = JuMP.objective_value(model_gradient)

        # Store in vector
        P_vector = push!(P_vector, [P_min, P_max])

    end

    print("\n", P_vector)

    return 0


end


# Total Law Expectation construction
function total_law_of_expectation()




end





# Control hypercube optimization 
function barrier_bellman_sos(system_dimension, partitions_eps, state_space, system_flag, neural_flag)

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

    # Hypercubes
    hypercubes = partition_space(state_space, partitions_eps)
    number_of_hypercubes = length(hypercubes)

    # Create optimization variables
    @variable(model, A[1:(system_dimension*number_of_hypercubes)])
    @variable(model, b[1:number_of_hypercubes])
    @variable(model, beta[1:number_of_hypercubes])

    # Specify Covariance Matrix (Gaussian)
    covariance_matrix = random_covariance_matrix(system_dimension) 

    # Specify initial state
    x_init::Array{Float64, 2} = zeros(1, system_dimension)

    # Barrier function
    alpha::Float64 = 1

    for jj = 1:number_of_hypercubes
        
        # Compute transition probability
        transition_probabilities = probability_distribution(hypercubes, covariance_matrix, system_dimension, system_flag, neural_flag)

        total_expectation = total_law_of_expectation()

        # return transition_probabilities

        return 0

        # Construct partition barrier
        A_j = A[(1+system_dimension*(jj-1):(system_dimension*jj))]
        b_j = b[jj]
        BARRIER_j = barrier_construct(system_dimension, A_j, b_j, x)

        # Constraint nonnegative
        add_constraint_to_model(model, BARRIER_j)

        # Decision variables
        beta_j = beta[jj]
        @constraint(model, beta_j >= 1e-6)

        if jj == hypercube_initial 
            @variable(model, eta)
            @constraint(model, eta >= 1e-6)
            @constraint(model, eta <= (1 - 1e-6))
        end

    end

    # # Specify degree Lagrangian polynomials and decision variables
    # lagrange_degree::Int64 = 2
    # lagrange_monomial::MonomialVector{true} = monomials(x, 0:lagrange_degree)
    # number_decision_vars = system_dimension*length(lagrange_monomial)
    # @variable(model, l[1:number_decision_vars])

    # # Constraint initial set and unsafe set
    # barrier_constraints_unsafe_initial = system_dimension

    # for ii = 1:barrier_constraints_unsafe_initial

    #     # Barrier unsafe region conditions
    #     if ii == 1

    #         # Generate sos polynomial
    #         count_lag = 0

    #         # Create Lagrangian
    #         lag_poly_i::DynamicPolynomials.Polynomial{true, AffExpr} =  sos_polynomial(l::Vector{VariableRef}, x::Array{PolyVar{true},1}, count_lag::Int64, lagrange_degree::Int64)
    #         add_constraint_to_model(model, lag_poly_i)

    #         # Specify set
    #         x_initial_radius = 2.0
    #         x_initial_sums = x_initial_radius
    #         for jj = 1:length(x)
    #             x_initial_sums += (x[jj] - x_init[jj])^2
    #         end

    #         # Constraint unsafe set
    #         _barrier_unsafe = BARRIER - lag_poly_i * x_initial_sums

    #         # Add constraint to model
    #         add_constraint_to_model(model, _barrier_unsafe)
     
    #     end

    #     # Barrier initial condition f(eta)
    #     if ii == barrier_constraints_unsafe_initial

    #         # Generate sos polynomial
    #         count_lag = 1

    #         # Create Lagrangian
    #         lag_poly_j::DynamicPolynomials.Polynomial{true, AffExpr} =  sos_polynomial(l::Vector{VariableRef}, x::Array{PolyVar{true},1}, count_lag::Int64, lagrange_degree::Int64)
    #         add_constraint_to_model(model, lag_poly_j)

    #         # Specify set
    #         x_initial_radius = 1.5
    #         x_initial_sums = x_initial_radius
    #         for jj = 1:length(x)
    #             x_initial_sums += -(x[jj] )^2
    #         end

    #         # Constraint eta
    #         _barrier_initial = - BARRIER + eta - lag_poly_j * x_initial_sums

    #         # Add constraint to model
    #         add_constraint_to_model(model, _barrier_initial)

    #     end

    # end

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


    # # Define optimization objective
    # time_horizon = 1
    # @objective(model, Min, eta + beta*time_horizon)
    # print("Objective made\n")

    # # Optimize model
    # optimize!(model)

    # # Barrier certificate
    # certificate = barrier_certificate(system_dimension, A, b, x)

    # # Print probability values
    # println("Solution: [eta = $(value(eta)), beta = $(value(beta)), total = $(value(eta) + value(beta)) ]")

    # # Return certificate
    # return certificate

end

# Optimize certificate
system_dimension = 2
partitions_eps = 0.5
state_space = ([-3, 3], [-3, 3])
system_flag = "population_growth"
neural_flag = false

# Optimize controller
@time X = barrier_bellman_sos(system_dimension,
                                 partitions_eps,
                                 state_space,
                                 system_flag,
                                 neural_flag)


# Plot hypercube
# pyplot()

# plot!(rand(10))
# plot!(rand(10))


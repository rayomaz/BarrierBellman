""" Extra functions: probably not useful here (delete later)

    © Rayan Mazouz

"""

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

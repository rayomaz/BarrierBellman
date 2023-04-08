""" Piecewise barrier utility functions

    © Rayan Mazouz

"""

# Compute the final barrier certificate
function piecewise_barrier_certificate(system_dimension, A, b, x)

    # Control Barrier Certificate
    barrier_certificate = value(b)
    for ii = 1:system_dimension
        barrier_certificate += value(A[ii])*x[ii]
    end

    return barrier_certificate

end

# Create Linear Barrier Function
function barrier_construct(system_dimension, A, b, x) #::DynamicPolynomials.Polynomial{true, AffExpr}

    barrier_linear = b
    for ii = 1:system_dimension
        barrier_linear += A[ii]*x[ii]
    end

    return barrier_linear
end

# Generate state space from bounds
function state_space_generation(state_partitions)

    # Identify lower bound
    lower_bound = low.(state_partitions)
    lower_bound = minimum(lower_bound)

    # Identify current bound
    upper_bound = high.(state_partitions)
    upper_bound = maximum(upper_bound)

    state_space = [lower_bound, upper_bound]

    return state_space

end

function maximum_beta_constraint(model, β_state, β)
    """ Adding constraint:
        - minimize for maximum β
    """

    # Create dummy variable 
    @polyvar w[1:2]

    w_min = 0
    w_max = 1

    product_set = (w_max - w[1]).*(w[1] - w_min)

    # Create Lagragian multiplier
    monos = monomials(w, 0:lagrange_degree)
    lag_poly_beta = @variable(model, variable_type=SOSPoly(monos))  

    # Specify beta bound
    beta_constraint = (- β_state + β - lag_poly_beta) * product_set
    @constraint(model, beta_constraint >= 0)

end
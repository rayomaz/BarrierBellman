""" Piecewise barrier utility functions

    © Rayan Mazouz

"""

# Create Control Barrier Polynomial
function barrier_polynomial(c::Vector{VariableRef}, barrier_monomial::MonomialVector{true})::DynamicPolynomials.Polynomial{true, AffExpr}
    barrier_poly = 0
    for cc in 1:Integer(length(barrier_monomial))
        barrier_poly += c[cc] * barrier_monomial[cc]
    end
    return barrier_poly
end

# Create SOS polynomial function
function sos_polynomial(vars, x, lagrange_degree::Int64)::DynamicPolynomials.Polynomial{true, AffExpr}
    sos_polynomial::MonomialVector{true}  = monomials(x, 0:lagrange_degree)

    sos_poly_t = 0
    for sos in 1:Integer(length(sos_polynomial))
        sos_poly_t += vars[sos] * sos_polynomial[sos]
    end

    return sos_poly_t::DynamicPolynomials.Polynomial{true, AffExpr}
    
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
function piecewise_barrier_certificate(system_dimension, A, b, x)

    # Control Barrier Certificate
    barrier_certificate = value(b)
    for ii = 1:system_dimension
        barrier_certificate += value(A[ii])*x[ii]
    end

    return barrier_certificate

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
    lower_bound = split(state_partitions[1])
    lower_bound = minimum(parse.(Float64, lower_bound))

    # Identify current bound
    upper_bound = split(state_partitions[end])
    upper_bound = maximum(parse.(Float64, upper_bound))

    state_space = [lower_bound, upper_bound]

    return state_space

end
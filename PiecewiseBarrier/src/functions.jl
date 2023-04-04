""" Piecewise barrier utility functions

    © Rayan Mazouz

"""

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

    hypercubes = [[[-3, 0], [0, 3]],
                  [[0, 3], [0, 3]],
                  [[0, 3], [0, -3]],
                  [[-3, 0], [-3, 0]]]

    return hypercubes

end

# Create Linear Barrier Function
function barrier_construct(system_dimension, A, b, x) #::DynamicPolynomials.Polynomial{true, AffExpr}

    barrier_linear = b
    for ii = 1:system_dimension
        barrier_linear += A[ii]*x[ii]
    end

    return barrier_linear
end

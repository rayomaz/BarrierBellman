""" Piecewise barrier utility functions

    © Rayan Mazouz

"""

vectorize(x::Vector) = x
vectorize(x::VariableRef) = [x]
vectorize(x::Number) = [x]
vectorize(x::AbstractPolynomialLike) = [x]

# Compute the final barrier certificate
piecewise_barrier_certificate(system, A, b) = barrier_construct(system, value.(A), value.(b))

# Create Linear Barrier Function
function barrier_construct(system, A, b) 
    x = variables(system)
    return dot(A, x) + b
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
""" Piecewise barrier utility functions

    © Rayan Mazouz

"""

function read_regions(partitions::MatlabFile, probabilities::MatlabFile)
    # Load partition matrices
    regions = read(partitions, "partitions")
    regions_lower = regions[:, 1, :]
    regions_upper = regions[:, 2, :]

    # Load probability matrices
    prob_lower = read(probabilities, "matrix_prob_lower")
    prob_upper = read(probabilities, "matrix_prob_upper")
    prob_unsafe_lower = read(probabilities, "matrix_prob_unsafe_lower")'
    prob_unsafe_upper = read(probabilities, "matrix_prob_unsafe_upper")'

    regions = [
        RegionWithProbabilities(
            Hyperrectangle(low=X̲, high=X̅), (copy(P̲), copy(P̅)), (copy(P̲ᵤ[1]), copy(P̅ᵤ[1]))
        ) for (X̲, X̅, P̲, P̅, P̲ᵤ, P̅ᵤ) in zip(
            eachrow(regions_lower),
            eachrow(regions_upper),
            eachrow(prob_lower),
            eachrow(prob_upper),
            eachrow(prob_unsafe_lower),
            eachrow(prob_unsafe_upper)
        )]

    return regions
end

vectorize(x::Vector) = x
vectorize(x::VariableRef) = [x]
vectorize(x::Number) = [x]
vectorize(x::AbstractPolynomialLike) = [x]

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

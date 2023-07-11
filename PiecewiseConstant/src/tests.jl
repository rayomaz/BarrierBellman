""" Piecewise barrier function construction

    © Rayan Mazouz

"""

export read_regions

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

function sum_probabilities(jj, probabilities::MatlabFile)
    # Bounds
    prob_lower = read(probabilities, "matrix_prob_lower")
    prob_upper = read(probabilities, "matrix_prob_upper")
    prob_unsafe_lower = read(probabilities, "matrix_prob_unsafe_lower")
    prob_unsafe_upper = read(probabilities, "matrix_prob_unsafe_upper")

    return sum_probabilities(jj, prob_lower, prob_upper, prob_unsafe_lower, prob_unsafe_upper)
end

function sum_probabilities(jj, prob_lower, prob_upper, prob_unsafe_lower, prob_unsafe_upper)
    
    # Sum all probabilities values for a given hypercube
    PU_upper = sum(prob_upper[jj, :]) + prob_unsafe_upper[jj]
    if PU_upper < 1
        print("Warning upper:" , jj)
    end

    PU_lower = sum(prob_lower[jj, :]) + prob_unsafe_lower[jj]
    if PU_lower > 1
        print("Warning lower:" , jj)
    end

    println("Sum upper/lower bounds: ", PU_upper, ", ", PU_lower)

end

function sum_barrier_probabilities(jj, b, beta, probabilities::MatlabFile)
    # Bounds
    prob_lower = read(probabilities, "matrix_prob_lower")
    prob_upper = read(probabilities, "matrix_prob_upper")
    prob_unsafe_lower = read(probabilities, "matrix_prob_unsafe_lower")
    prob_unsafe_upper = read(probabilities, "matrix_prob_unsafe_upper")

    return sum_barrier_probabilities(jj, b, beta, prob_lower, prob_upper, prob_unsafe_lower, prob_unsafe_upper)
end

function sum_barrier_probabilities(jj, b, beta, prob_lower, prob_upper, prob_unsafe_lower, prob_unsafe_upper)
  
    number_hypercubes = length(b)
    
    exp = 0
    for ii = 1:number_hypercubes
        exp += b[ii]*prob_upper[jj, ii]
    end
    exp += prob_unsafe_upper[jj]

    println("E[B]: ", exp, ", β: ", beta[jj], ", barrier: ", b[jj], ", ∑(β + b): ", beta[jj] + b[jj])

    return exp
end


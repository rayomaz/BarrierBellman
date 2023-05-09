
""" Extra function: exact transition probabilities
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

# Proper transformation under system dynamics
function system_dynamics(system::AdditiveGaussianPolynomialSystem{T, N}, T_a) where {T, N}

    if system_flag == "linear"

        # System properties
        fx = dynamics(system)

        # Dynamics definion
        x_next_step = fx

        # Modified hyperspace (see Eq. 17: http://dx.doi.org/10.1145/3302504.3311805)
        y = T_a*x_next_step

        return y

    end

end
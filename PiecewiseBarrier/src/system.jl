

abstract type AbstractDiscreteTimeStochasticSystem{N} end



struct AdditiveGaussianPolynomialSystem{T, N} <: AbstractDiscreteTimeStochasticSystem{N}
    # This struct represents a system with the dynamics x(k + 1) = f(x(k)) + v(k)
    # where the set of random variables (v(k))_{k ∈ ℕ} are independent and identically
    # Gaussian distributed.

    x
    fx
    σ
end

variables(system::AdditiveGaussianPolynomialSystem) = vectorize(system.x)
dynamics(system::AdditiveGaussianPolynomialSystem) = vectorize(system.fx)
noise_distribution(system::AdditiveGaussianPolynomialSystem) = vectorize(system.σ)

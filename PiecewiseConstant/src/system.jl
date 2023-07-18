"""
    - System structure

    © Frederik Baymler Mathiesen, Rayan Mazouz

"""

abstract type AbstractDiscreteTimeStochasticSystem{N} end

struct AdditiveGaussianPolynomialSystem{T, N} <: AbstractDiscreteTimeStochasticSystem{N}
    # This struct represents a system with the dynamics x(k + 1) = f(x(k)) + v(k)
    # where the set of random variables (v(k))_{k ∈ ℕ} are independent and identically
    # Gaussian distributed.

    x
    fx
    σ
end

AdditiveGaussianPolynomialSystem{N}(x, fx, σ) where {N} = AdditiveGaussianPolynomialSystem{Float64, N}(x, fx, σ)
AdditiveGaussianPolynomialSystem(x::MP.AbstractVariable, fx, σ) = AdditiveGaussianPolynomialSystem{1}(x, fx, σ)
AdditiveGaussianPolynomialSystem(x, fx, σ) = AdditiveGaussianPolynomialSystem{length(x)}(x, fx, σ)

dimensionality(system::AdditiveGaussianPolynomialSystem{T, N}) where {T, N} = N
variables(system::AdditiveGaussianPolynomialSystem) = vectorize(system.x)
dynamics(system::AdditiveGaussianPolynomialSystem) = vectorize(system.fx)
noise_distribution(system::AdditiveGaussianPolynomialSystem) = vectorize(system.σ)
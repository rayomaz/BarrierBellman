"""
    - System structure

    © Frederik Baymler Mathiesen, Rayan Mazouz

"""

abstract type AbstractDiscreteTimeStochasticSystem{N} end
abstract type AbstractAdditiveGaussianSystem{N} <: AbstractDiscreteTimeStochasticSystem{N} end

dimensionality(::AbstractDiscreteTimeStochasticSystem{N}) where {N} = N
noise_distribution(system::AbstractAdditiveGaussianSystem) = system.σ

struct AdditiveGaussianLinearSystem{T, N} <: AbstractAdditiveGaussianSystem{N}
    # This struct represents a system with the dynamics x(k + 1) = f(x(k)) + v(k)
    # where f is affine in x and the set of random variables (v(k))_{k ∈ ℕ} are
    # independent and identically Gaussian distributed.

    A::AbstractMatrix{T}
    b::AbstractVector{T}
    σ::AbstractVector{T}
end
AdditiveGaussianLinearSystem(A, b, σ) = AdditiveGaussianLinearSystem{eltype(A), length(σ)}(A, b, σ)
dynamics(system::AdditiveGaussianLinearSystem) = (system.A, system.b)


struct UncertainPWARegion{T, S<:LazySet{T}}
    X::S

    dyn::Vector{Tuple{Matrix{T}, Vector{T}}}
end
Base.iterate(X::UncertainPWARegion) = (X.X, Val(:dyn))
Base.iterate(X::UncertainPWARegion, ::Val{:dyn}) = (X.dyn, Val(:done))
Base.iterate(X::UncertainPWARegion, ::Val{:done}) = nothing
dynamics(X::UncertainPWARegion) = X.dyn
region(X::UncertainPWARegion) = X.X

struct AdditiveGaussianUncertainPWASystem{T, N, S<:UncertainPWARegion{T}} <: AbstractAdditiveGaussianSystem{N}
    # This struct represents a system with the dynamics x(k + 1) = f(x(k)) + v(k)
    # where f is an uncertain PWA function and the set of random variables (v(k))_{k ∈ ℕ} are
    # independent and identically Gaussian distributed.

    Xs::AbstractVector{S}
    σ::AbstractVector{T}
end
AdditiveGaussianUncertainPWASystem(Xs, σ) = AdditiveGaussianUncertainPWASystem{eltype(σ), length(σ), eltype(Xs)}(Xs, σ)
regions(system::AdditiveGaussianUncertainPWASystem) = map(X -> X.X, system.Xs)
dynamics(system::AdditiveGaussianUncertainPWASystem) = system.Xs
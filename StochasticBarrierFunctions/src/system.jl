abstract type AbstractDiscreteTimeStochasticSystem{N} end
abstract type AbstractAdditiveGaussianSystem{N} <: AbstractDiscreteTimeStochasticSystem{N} end

dimensionality(::AbstractDiscreteTimeStochasticSystem{N}) where {N} = N
noise_distribution(system::AbstractAdditiveGaussianSystem) = system.σ

struct AdditiveGaussianLinearSystem{T, N} <: AbstractAdditiveGaussianSystem{N}
    # This struct represents a system with the dynamics x(k + 1) = f(x(k)) + v(k)
    # where f is affine in x and the set of random variables (v(k))_{k ∈ ℕ} are
    # independent and identically Gaussian distributed with zero mean and diagonal
    # covariance. The stochasticity is represented by the diagonal std.dev.'s σ.

    A::AbstractMatrix{T}
    b::AbstractVector{T}
    σ::AbstractVector{T}

    state_space::Union{AbstractHyperrectangle{T}, Nothing}

    function AdditiveGaussianLinearSystem(A::AbstractMatrix{T}, b::AbstractVector{T}, σ::AbstractVector{T}, state_space::Union{AbstractHyperrectangle{T}, Nothing}=nothing) where {T}
        n = LinearAlgebra.checksquare(A)
        
        if size(b, 1) != n
            throw(DimensionMismatch("The number of elements in b must be equal to the number of rows/columns in A"))
        end

        if size(σ, 1) != n
            throw(DimensionMismatch("The number of elements in σ must be equal to the number of rows/columns in A"))
        end

        new{T, length(σ)}(A, b, σ, state_space)
    end
end
dynamics(system::AdditiveGaussianLinearSystem) = (system.A, system.b)


struct AdditiveGaussianPolySystem{T, N} <: AbstractAdditiveGaussianSystem{N}
    # This struct represents a system with the dynamics x(k + 1) = f(x(k)) + v(k)
    # where f is polynomial in x and the set of random variables (v(k))_{k ∈ ℕ} are
    # independent and identically Gaussian distributed with zero mean and diagonal
    # covariance. The stochasticity is represented by the diagonal std.dev.'s σ.

    f::Any
    σ::AbstractVector{T}

    state_space::Union{AbstractHyperrectangle{T}, Nothing}

    function AdditiveGaussianPolySystem(f::Any, σ::AbstractVector{T}, state_space::Union{AbstractHyperrectangle{T}, Nothing}=nothing) where {T}
        n = length(f)

        if length(σ) != n
            throw(DimensionMismatch("The number of elements in σ must be equal to the number of dimensions of f"))
        end

        new{T, length(σ)}(f, σ, state_space)
    end
end
dynamics(system::AdditiveGaussianPolySystem) = system.f


struct UncertainPWARegion{T, S<:LazySet{T}}
    X::S

    dyn::Vector{Tuple{Matrix{T}, Vector{T}}}

    function UncertainPWARegion(X::S, dyn::Vector{Tuple{Matrix{T}, Vector{T}}}) where {T, S}
        if length(dyn) == 0
            throw(ArgumentError("The dynamics must be non-empty"))
        end

        n = LazySets.dim(X)
        for (A, b) in dyn
            m = LinearAlgebra.checksquare(A)

            if m != n
                throw(DimensionMismatch("The number of rows/columns in A must be equal to the dimensionality of the region"))
            end

            if size(b, 1) != n
                throw(DimensionMismatch("The number of elements in b must be equal to the number of rows/columns in A"))
            end
        end
        
        new{T, S}(X, dyn)
    end
end
Base.iterate(X::UncertainPWARegion) = (X.X, Val(:dyn))
Base.iterate(X::UncertainPWARegion, ::Val{:dyn}) = (X.dyn, Val(:done))
Base.iterate(X::UncertainPWARegion, ::Val{:done}) = nothing
dynamics(X::UncertainPWARegion) = X.dyn
region(X::UncertainPWARegion) = X.X

struct AdditiveGaussianUncertainPWASystem{T, N, S<:UncertainPWARegion{T}} <: AbstractAdditiveGaussianSystem{N}
    # This struct represents a system with the dynamics x(k + 1) = f(x(k)) + v(k)
    # where f is an uncertain PWA function and the set of random variables (v(k))_{k ∈ ℕ} are
    # independent and identically Gaussian distributed with zero mean and diagonal
    # covariance. The stochasticity is represented by the diagonal std.dev.'s σ.

    Xs::AbstractVector{S}
    σ::AbstractVector{T}
end
AdditiveGaussianUncertainPWASystem(Xs, σ) = AdditiveGaussianUncertainPWASystem{eltype(σ), length(σ), eltype(Xs)}(Xs, σ)
regions(system::AdditiveGaussianUncertainPWASystem) = map(X -> X.X, system.Xs)
dynamics(system::AdditiveGaussianUncertainPWASystem) = system.Xs

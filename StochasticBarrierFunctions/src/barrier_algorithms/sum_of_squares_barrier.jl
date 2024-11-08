export SumOfSquaresAlgorithm, SumOfSquaresAlgResult

Base.@kwdef struct SumOfSquaresAlgorithm <: SumOfSquaresBarrierAlgorithm
    barrier_degree = 4
    lagrange_degree = 2
    sdp_solver = default_sdp_solver()
end

struct SumOfSquaresAlgResult <: BarrierResult
    B::SumOfSquaresBarrier
    η::Float64
    β::Float64
    synthesis_time::Float64  # Total time to solve the optimization problem in seconds
end

barrier(res::SumOfSquaresAlgResult) = res.B
eta(res::SumOfSquaresAlgResult) = res.η
beta(res::SumOfSquaresAlgResult) = res.β
total_time(res::SumOfSquaresAlgResult) = res.synthesis_time

# Sum of squares optimization function
function synthesize_barrier(alg::SumOfSquaresAlgorithm, system, initial_region::LazySet, obstacle_region::LazySet; time_horizon=1)
    synthesis_time = @elapsed begin
        model = SOSModel(alg.sdp_solver)
        set_silent(model)

        # Create decision variables eta and beta
        @variable(model, η, lower_bound=0.0)
        @variable(model, β, lower_bound=0.0)

        # Create barrier candidate
        @polyvar x[1:dimensionality(system)]
        barrier_monomials = monomials(x, 0:floor(Int64, alg.barrier_degree / 2))

        # Non-negative in ℝⁿ
        @variable(model, B, SOSPoly(barrier_monomials))

        sos_initial_constraint!(model, B, x, η, initial_region)
        sos_obstacle_constraint!(model, B, x, obstacle_region)
        sos_system_specific_constraints!(alg, model, B, x, β, system)

        # Define optimization objective
        @objective(model, Min, η + β * time_horizon)

        # Optimize model
        optimize!(model)

        # Barrier certificate
        B, η, β = MP.polynomial(value(B)), value(η), value(β)
    end

    res = SumOfSquaresAlgResult(SumOfSquaresBarrier(B), η, β, synthesis_time)

    @info "Solution Sum of Squares" η=eta(res) β=beta(res) Pₛ=psafe(res, time_horizon) time=total_time(res)

    return res
end

function check_hypercube_state_space(system)
    Xs = regions(system)

    state_space = box_approximation(UnionSetArray(Xs))

    # TODO: Check if ⋃Xs = state_space

    return state_space
end

sos_initial_constraint!(model, B, x, η, region::AbstractHyperrectangle) = sos_initial_constraint!(model, B, x, η, [region])
sos_initial_constraint!(model, B, x, η, region::UnionSet{T, <:AbstractHyperrectangle, <:AbstractHyperrectangle}) where {T} = sos_initial_constraint!(model, B, x, η, [region.X, region.Y])
sos_initial_constraint!(model, B, x, η, region::UnionSetArray{T, <:AbstractHyperrectangle}) where {T} = sos_initial_constraint!(model, B, x, η, region.array)
function sos_initial_constraint!(model, B, x, η, regions::Vector{<:AbstractHyperrectangle})
    """ Barrier condition: initial
        * B(x) <= η
    """
    for region in regions
        initial_domain = sos_hpoly_lag(model, x, region)
        @constraint(model, -B + η - initial_domain >= 0)
    end
end

sos_obstacle_constraint!(model, B, x, region::AbstractPolyhedron) = sos_obstacle_constraint!(model, B, x, [region])
sos_obstacle_constraint!(model, B, x, region::EmptySet) = nothing
sos_obstacle_constraint!(model, B, x, region::UnionSet{T, <:AbstractPolyhedron, <:AbstractPolyhedron}) where {T} = sos_obstacle_constraint!(model, B, x, [region.X, region.Y])
sos_obstacle_constraint!(model, B, x, region::UnionSetArray{T, <:AbstractPolyhedron}) where {T} = sos_obstacle_constraint!(model, B, x, region.array)
function sos_obstacle_constraint!(model, B, x, regions::Vector{<:AbstractPolyhedron})
    """ Barrier obstacle region conditions
        * B(x) >= 1
    """
    for region in regions
        obstacle_domain = sos_hpoly_lag(model, x, region)
        @constraint(model, B - 1 - obstacle_domain >= 0)
    end
end

function sos_system_specific_constraints!(alg, model, B, x, β, system::AdditiveGaussianUncertainPWASystem)
    # Unsafe set
    state_space = check_hypercube_state_space(system)
    sos_unsafe_constraint!(alg, model, B, x, state_space)

    # Expectation constraint
    for dyn_region in dynamics(system)
        sos_expectation_constraint!(alg, model, system, B, x, dyn_region, β)
    end
end

function sos_system_specific_constraints!(alg, model, B, x, β, system::AdditiveGaussianLinearSystem)
    # Unsafe set
    sos_unsafe_constraint!(alg, model, B, x, system.state_space)

    # Expectation constraint
    @polyvar z[eachindex(x)]

    A, b = dynamics(system)
    fx = subs(B, x => A * x + b + z)

    σ_noise = noise_distribution(system)
    exp = expectation_noise(fx, σ_noise, z)

    exp_domain = sos_hpoly_lag(model, x, system.state_space)

    # Add constraint
    @constraint(model, -exp + B + β - exp_domain >= 0)
end

function sos_system_specific_constraints!(alg, model, B, x, β, system::AdditiveGaussianPolySystem)
    # Unsafe set
    sos_unsafe_constraint!(alg, model, B, x, system.state_space)

    # Expectation constraint
    @polyvar z[eachindex(x)]

    f = dynamics(system)
    fx = subs(B, x => f + z)

    σ_noise = noise_distribution(system)
    exp = expectation_noise(fx, σ_noise, z)

    exp_domain = sos_hpoly_lag(model, x, system.state_space)

    # Add constraint
    @constraint(model, -exp + B + β - exp_domain >= 0)
end

function sos_unsafe_constraint!(alg, model, B, x, state_space::AbstractHyperrectangle)
    """ Barrier unsafe region conditions
        * B(x) >= 1
    """

    product_set_lower = low(state_space) - x
    product_set_upper = x - high(state_space)
    monos = monomials(x, 0:floor(Int64, alg.lagrange_degree / 2))

    for (dim_set_lower, dim_set_upper) in zip(product_set_lower, product_set_upper)
        # Lagragian multiplier
        lag_poly_lower = @variable(model, variable_type=SOSPoly(monos))
        domain_lower = lag_poly_lower * dim_set_lower

        lag_poly_upper = @variable(model, variable_type=SOSPoly(monos))
        domain_upper = lag_poly_upper * dim_set_upper

        @constraint(model, B - 1 - domain_lower >= 0)
        @constraint(model, B - 1 - domain_upper >= 0)
    end
end

function sos_hpoly_lag(model, x, region)
    # Quadratic H-polytope encoding
    H, h = tosimplehrep(region)
    halfspace_constraints = H * x - h

    domain = sum(Iterators.product(halfspace_constraints, halfspace_constraints)) do (constraint1, constraint2)
        # TODO: Skip if the `constraint1 * constraint2` is zero (i.e. the constraints are orthogonal)
        
        # Lagragian multiplier
        τ = @variable(model, lower_bound=0.0)
        return τ * constraint1 * constraint2
    end

    return domain
end

function sos_expectation_constraint!(alg, model, system, B, x, dyn_region, β)
    """ Barrier martingale condition
        * E[B(f(x,u))] <= B(x) + β
    """
    @polyvar z[eachindex(x)]
    @polyvar y[eachindex(x)]

    X, dyn = dyn_region
    
    # Current state partition
    exp_domain = sos_hpoly_lag(model, x, X)

    # Compute dynamics domain
    dyn_lower = dyn[1][1] * x + dyn[1][2]
    dyn_upper = dyn[2][1] * x + dyn[2][2]
    product_set = (dyn_upper - y) .* (y - dyn_lower)

    monos_y = monomials(y, 0:floor(Int64, alg.lagrange_degree / 2))
    dyn_domain = sum(product_set) do dim_set
        # Lagragian multiplier
        lag_poly = @variable(model, variable_type=SOSPoly(monos_y))
        return lag_poly * dim_set
    end

    # Compute expectation
    fx = subs(B, x => y + z)

    σ_noise = noise_distribution(system)
    exp = expectation_noise(fx, σ_noise, z)

    # Add constraint
    @constraint(model, -exp + B + β - exp_domain - dyn_domain >= 0)
end

# Function to compute the expecation and noise element
function expectation_noise(exp_evaluated, standard_deviations, zs)
    exp = 0

    for term in terms(exp_evaluated)
        z_degs = [MultivariatePolynomials.degree(term, z) for z in zs]
        z_occurs = sum(z_degs) > 0

        if z_occurs
            if all(iseven, z_degs)
                coeff = subs(term, zs => ones(length(zs)))
                exp_z = prod(splat(expected_univariate_noise), zip(z_degs, standard_deviations))

                exp += coeff * exp_z
            end
        else
            exp += term
        end
    end

    return exp
end

function expected_univariate_noise(z_deg, standard_deviation)
    if z_deg == 0
        return 1.0
    else
        return Int64(doublefactorial(z_deg - 1)) * standard_deviation^z_deg
    end
end
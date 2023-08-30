""" Stochastic barrier function construction (Sum of Squares)

    © Rayan Mazouz

"""

# Sum of squares optimization function
function synthesize_barrier(alg::SumOfSquaresAlgorithm, system::AdditiveGaussianUncertainPWASystem, initial_region::LazySet, obstacle_region::LazySet; time_horizon=1)
    model = SOSModel(alg.sdp_solver)

    # State space
    number_state_hypercubes = length(regions(system))
    state_space = check_hypercube_state_space(system)

    # Create probability decision variables eta
    @variable(model, η >= alg.ϵ)

    # Create barrier candidate
    @polyvar x[1:dimensionality(system)]
    barrier_monomials = monomials(x, 0:floor(Int64, alg.barrier_degree / 2))

    # Non-negative in ℝⁿ
    @variable(model, B, SOSPoly(barrier_monomials))

    # Barrier constraints and β variable
    @variable(model, β_parts_var[1:number_state_hypercubes] >= alg.ϵ)
    @variable(model, β)
    @constraint(model, β_parts_var .<= β)

    # Unsafe set
    sos_unsafe_constraint!(alg, model, B, x, state_space)

    for (dyn_region, βⱼ) in zip(dynamics(system), β_parts_var)
        X = region(dyn_region)
        # Initial set
        if !isdisjoint(initial_region, X)
            sos_initial_constraint!(alg, model, B, x, η, X)
        end

        # Obstacle
        if !isdisjoint(obstacle_region, X)
            sos_unsafe_constraint!(alg, model, B, x, X)
        end

        # Expectation constraint
        sos_expectation_constraint!(alg, model, system, B, x, dyn_region, βⱼ)
    end

    # Define optimization objective
    @objective(model, Min, η + β * time_horizon)

    # Optimize model
    optimize!(model)

    # Barrier certificate
    B = value(B)
    β_values = value.(β_parts_var)

    @info "Solution Gradient Descent" η=value(η) β=maximum(β_values) Pₛ=1 - (value(η) + maximum(β_values) * time_horizon)

    return SOSBarrier(B), β_values
end

function check_hypercube_state_space(system)
    Xs = regions(system)

    state_space = box_approximation(UnionSetArray(Xs))

    # TODO: Check if ⋃Xs = state_space

    return state_space
end

function sos_unsafe_constraint!(alg, model, B, x, state_space)
    """ Barrier unsafe region conditions
        * B(x) >= 1
    """

    product_set_lower = low(state_space) - x
    product_set_upper = x - high(state_space)

    for (xi, dim_set_lower, dim_set_upper) in zip(x, product_set_lower, product_set_upper)

        # Lagragian multiplier
        monos = monomials(xi, 0:alg.lagrange_degree)
        lag_poly_lower = @variable(model, variable_type=SOSPoly(monos))
        domain_lower = lag_poly_lower * dim_set_lower

        lag_poly_upper = @variable(model, variable_type=SOSPoly(monos))
        domain_upper = lag_poly_upper * dim_set_upper

        # Add constraints to model
        @constraint(model, B - 1 - domain_lower >= 0)
        @constraint(model, B - 1 - domain_upper >= 0)

    end
end

function sos_initial_constraint!(alg, model, B, x, η, region)
    """ Barrier condition: initial
        * B(x) <= η
    """
    initial_domain = sos_hyperrectangle_lag(alg, model, x, region)

    # Add constraint to model
    @constraint(model, -B + η - initial_domain >= 0)
end

function sos_obstacle_constraint!(alg, model, B, x, region)
    """ Barrier obstacle region conditions
        * B(x) >= 1
    """
    obstacle_domain = sos_hyperrectangle_lag(alg, model, x, region)

    # Add constraint to model
    @constraint(model, B - η - obstacle_domain >= 0)
end

function sos_hyperrectangle_lag(alg, model, x, region)
    lower_state = low(region)
    upper_state = high(region)
    product_set = (upper_state - x) .* (x - lower_state)

    domain = sum(zip(x, product_set)) do (xi, dim_set)
        # Lagragian multiplier
        monos = monomials(xi, 0:alg.lagrange_degree)
        lag_poly = @variable(model, variable_type=SOSPoly(monos))

        return lag_poly * dim_set
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
    exp_domain = sos_hyperrectangle_lag(alg, model, x, X)

    # Compute dynamics domain
    dyn_lower = dyn[1][1] * x + dyn[1][2]
    dyn_upper = dyn[2][1] * x + dyn[2][2]
    product_set = (dyn_upper - y) .* (y - dyn_lower)

    dyn_domain = sum(zip(x, product_set)) do (xi, dim_set)
        # Lagragian multiplier
        monos = monomials(xi, 0:alg.lagrange_degree)
        lag_poly = @variable(model, variable_type=SOSPoly(monos))

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

                noise_exp = coeff * exp_z
                exp += noise_exp
            end
        else
            exp += term
        end
    end

    return exp
end

function expected_univariate_noise(z_deg, standard_deviation)
    if z_deg == 0
        return 1
    else
        return Int64(doublefactorial(z_deg - 1)) * standard_deviation^z_deg
    end
end
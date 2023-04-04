""" Closed-loop expectation expressions for a Gaussian distribution:

    © Rayan Mazouz, Frederik Baymler Mathiesen

"""

# Function to compute the expecation and noise element
function expectation_noise(exp_evaluated::DynamicPolynomials.Polynomial{true, AffExpr}, barrier_degree::Int64, standard_deviation::Float64, zs::Vector{PolyVar{true}})
    exp_poly::DynamicPolynomials.Polynomial{true, AffExpr} = 0
    noise::DynamicPolynomials.Polynomial{true, AffExpr} = 0

    for term in terms(exp_evaluated)
        z_degs = [MultivariatePolynomials.degree(term, z) for z in zs]

        z_occurs = sum(z_degs) > 0

        if z_occurs == false
            exp_poly = exp_poly + term
        end

        if z_occurs == true
            all_even = all(iseven, z_degs)

            if all_even
                coeff = subs(term, zs => ones(length(zs)))
                exp_z = prod([expected_univariate_noise(z_deg, standard_deviation) for z_deg in z_degs])

                noise_exp = coeff * exp_z
                noise = noise + noise_exp
            end
        end
    end

    return exp_poly, noise
end

function expected_univariate_noise(z_deg, standard_deviation)
    if z_deg == 0
        return 1
    else
        return (doublefactorial(z_deg - 1) * standard_deviation^z_deg)
    end
end

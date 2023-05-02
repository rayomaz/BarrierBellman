struct InterregionRemovedNewton{
    CT<:SumOfSquares.SOSLikeCone,
    BT<:MB.AbstractPolynomialBasis,
} <: SumOfSquares.Certificate.SimpleIdealCertificate{CT,BT}
    cone::CT
    basis::Type{BT}
    state_variables
    aux_variables
end
function SumOfSquares.Certificate.gram_basis(certificate::InterregionRemovedNewton{CT,B}, poly) where {CT,B}
    monomials = MP.monomials(poly)
    monomial_groups = [
        [mono for mono in monomials if contains_only_vars(mono, [certificate.state_variables; [var]])]
        for var in certificate.aux_variables]

    reduced_monomials = [
        SumOfSquares.Certificate.monomials_half_newton_polytope(
            group,
            tuple(),
        ) for group in monomial_groups]

    reduced_monomials = vcat(reduced_monomials...)
    unique!(reduced_monomials)

    return MB.basis_covering_monomials(
        B,
        reduced_monomials,
    )
end
function SumOfSquares.Certificate.gram_basis_type(::Type{<:InterregionRemovedNewton{CT,BT}}) where {CT,BT}
    return BT
end

function contains_only_vars(monomial, vars)
    mono_vars = MP.effective_variables(monomial)

    return mono_vars ⊆ vars
end

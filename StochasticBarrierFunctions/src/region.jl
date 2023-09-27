struct RegionWithProbabilities{T, S<:LazySet{T}, VT<:AbstractVector{T}}
    region::S
    lower::VT
    upper::VT
    gap::VT

    sum_lower::T
    sum_upper::T

    function RegionWithProbabilities(region::S, transition_to_other_regions::Tuple{VT, VT}) where {T, S<:LazySet{T}, VT<:AbstractVector{T}}
        # Include custom constructor only for safety checks

        lower, upper = transition_to_other_regions
        joint_lower_bound = sum(lower)
        @assert joint_lower_bound <= 1 "The joint lower bound transition probability (is $joint_lower_bound) should be less than or equal to 1."

        joint_upper_bound = sum(upper)
        @assert joint_upper_bound >= 1 - 1e-9 "The joint upper bound transition probability (is $joint_upper_bound) should be greater than or equal to 1."

        return new{T, S, VT}(region, lower, upper, upper - lower, sum(lower), sum(upper))
    end
end

function RegionWithProbabilities(region::S, transition_to_other_regions::Tuple{VT, VT}, transition_to_unsafe::Tuple{T, T}) where {T, S<:LazySet{T}, VT<:AbstractVector{T}}
    # Include custom constructor only for safety checks

    lower, upper = vcat(transition_to_other_regions[1], transition_to_unsafe[1]),
                   vcat(transition_to_other_regions[2], transition_to_unsafe[2])

    return RegionWithProbabilities(region, (lower, upper))
end

region(X::RegionWithProbabilities) = X.region
prob_lower(X::RegionWithProbabilities) = X.lower[1:end - 1]
prob_upper(X::RegionWithProbabilities) = X.upper[1:end - 1]
prob_unsafe_lower(X::RegionWithProbabilities) = X.lower[end]
prob_unsafe_upper(X::RegionWithProbabilities) = X.upper[end]

function ivi_prob(X::RegionWithProbabilities{T}, q::AbstractVector{<:Integer}) where {T}
    v = Vector{T}(undef, length(q))
    return ivi_prob!(v, X, q)
end

function ivi_prob!(p, X::RegionWithProbabilities, q)
    copyto!(p, X.lower)
    
    remaining = 1.0 - X.sum_lower
    
    @inbounds for i in q
        @inbounds p[i] += X.gap[i]
        @inbounds remaining -= X.gap[i]
        if remaining < 0.0
            @inbounds p[i] += remaining
            remaining = 0.0
            break
        end
    end

    @assert remaining == 0.0 "The remaining probability should be zero, but is $remaining."

    return p
end

function update_regions(regions::Vector{<:RegionWithProbabilities}, p_distribution::Vector{<:AbstractVector})
    new_regions = Vector{RegionWithProbabilities}(undef, length(regions))

    Threads.@threads for jj in eachindex(regions)
        Xⱼ = regions[jj]
        p_values = p_distribution[jj]

        # Compute new transition probabilities
        new_p = Xⱼ.lower, p_values

        new_regions[jj] = RegionWithProbabilities(region(Xⱼ), new_p)
    end

    return new_regions
end

function update_regions(regions::Vector{<:RegionWithProbabilities}, p_distribution::Matrix{Float64})
    new_regions = Vector{RegionWithProbabilities}(undef, length(regions))

    Threads.@threads for jj in eachindex(regions)
        Xⱼ = regions[jj]
        p_values = p_distribution[:, jj]

        # Compute new transition probabilities
        new_p = Xⱼ.lower, p_values

        new_regions[jj] = RegionWithProbabilities(region(Xⱼ), new_p)
    end

    return new_regions
end

##### WARNING: This is type piracy #####
# This is a type piracy of the `LazySets` package.
# Please don't do this, but I needed to do this for the linear 2D system.
@commutative function LazySets.isdisjoint(H::AbstractHyperrectangle, B::Ball2)
    return LazySets._isdisjoint_convex_sufficient(H, B)
end
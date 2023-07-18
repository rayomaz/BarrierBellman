struct RegionWithProbabilities{T, S<:LazySet{T}}
    region::S
    transition_to_other_regions::Tuple{Vector{T}, Vector{T}}
    transition_to_unsafe::Tuple{T, T}

    function RegionWithProbabilities(region::S, transition_to_other_regions::Tuple{Vector{T}, Vector{T}}, transition_to_unsafe::Tuple{T, T}) where {T, S<:LazySet{T}}
        # Include custom constructor only for safety checks

        joint_lower_bound = sum(transition_to_other_regions[1]) + transition_to_unsafe[1]
        @assert joint_lower_bound <= 1 "The joint lower bound transition probability (is $joint_lower_bound) should be less than or equal to 1."

        joint_upper_bound = sum(transition_to_other_regions[2]) + transition_to_unsafe[2]
        @assert joint_upper_bound >= 1 "The joint upper bound transition probability (is $joint_upper_bound) should be greater than or equal to 1."

        return new{T, S}(region, transition_to_other_regions, transition_to_unsafe)
    end
end

region(X::RegionWithProbabilities) = X.region
prob_lower(X::RegionWithProbabilities) = X.transition_to_other_regions[1]
prob_upper(X::RegionWithProbabilities) = X.transition_to_other_regions[2]
prob_unsafe_lower(X::RegionWithProbabilities) = X.transition_to_unsafe[1]
prob_unsafe_upper(X::RegionWithProbabilities) = X.transition_to_unsafe[2]
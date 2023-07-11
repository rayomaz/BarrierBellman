export region, prob_lower, prob_upper, prob_unsafe_lower, prob_unsafe_upper

struct RegionWithProbabilities{T, S<:LazySet{T}}
    region::S
    transition_to_other_regions::Tuple{Vector{T}, Vector{T}}
    transition_to_unsafe::Tuple{T, T}
end

region(X::RegionWithProbabilities) = X.region
prob_lower(X::RegionWithProbabilities) = X.transition_to_other_regions[1]
prob_upper(X::RegionWithProbabilities) = X.transition_to_other_regions[2]
prob_unsafe_lower(X::RegionWithProbabilities) = X.transition_to_unsafe[1]
prob_unsafe_upper(X::RegionWithProbabilities) = X.transition_to_unsafe[2]
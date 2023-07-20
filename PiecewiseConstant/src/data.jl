
function create_probability_dataset(regions::Vector{<:Hyperrectangle}, P̲::AbstractDimArray, P̅::AbstractDimArray, P̲ᵤ::AbstractDimArray, P̅ᵤ::AbstractDimArray)
    n = length(regions)
    d = (LazySets.dim ∘ first)(regions)

    axlist = (Dim{:region}(1:n), Dim{:dir}(["lower", "upper"]), Dim{:dim}(1:d))
    l, h = stack(low.(regions); dims=1), stack(high.(regions); dims=1)
    regions = YAXArray(axlist, stack((l, h); dims=2))  # NOTE: Order of stacking is important here.
    @assert size(regions) == (n, 2, d)

    axlist = (Dim{:to}(1:n), Dim{:from}(1:n), Dim{:dir}(["lower", "upper"]))
    order(A) = permutedims(A, (:to, :from))
    prob = YAXArray(axlist, stack((order(P̲), order(P̅)); dims=3))  # NOTE: Order of stacking is important here.
    @assert size(prob) == (n, n, 2)

    axlist = (Dim{:from}(1:n), Dim{:dir}(["lower", "upper"]))
    prob_unsafe = YAXArray(axlist, stack((P̲ᵤ, P̅ᵤ); dims=2))  # NOTE: Order of stacking is important here.
    @assert size(prob_unsafe) == (n, 2)

    ds = YAXArrays.Dataset(regions=regions, prob=prob, prob_unsafe=prob_unsafe; properties=Dict("dim"=>d, "num_regions"=>n))
    return ds
end
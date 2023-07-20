
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

function load_probabilities(dataset::YAXArrays.Dataset)
    n = dataset.properties["num_regions"]

    regions = dataset.regions
    P̲ = dataset.prob[dir=At("lower")]
    P̅ = dataset.prob[dir=At("upper")]
    P̲ᵤ = dataset.prob_unsafe[dir=At("lower")]
    P̅ᵤ = dataset.prob_unsafe[dir=At("upper")]

    regions = [
        RegionWithProbabilities(
            Hyperrectangle(low=copy(regions[region=j, dir=At("lower")].data), high=copy(regions[region=j, dir=At("upper")].data)),
            (copy(P̲[from=j].data), copy(P̅[from=j].data)),
            (copy(P̲ᵤ[from=j].data[1]), copy(P̅ᵤ[from=j].data[1]))
        )
        for j in 1:n
    ]

    return regions
end

function load_regions(partitions::MatlabFile)
    regions = read(partitions, "partitions")
    regions_lower = regions[:, 1, :]
    regions_upper = regions[:, 2, :]

    return [Hyperrectangle(low=X̲, high=X̅) for (X̲, X̅) in zip(eachrow(regions_lower), eachrow(regions_upper))]
end

function load_dynamics(partitions::MatlabFile)
    # Extract hypercube data
    state_partitions = read(partitions, "partitions")

    # Extract Neural Network Bounds [CROWN]
    M_upper = read(partitions, "M_h")
    M_lower = read(partitions, "M_l")
    b_upper = read(partitions, "B_h")
    b_lower = read(partitions, "B_l")

    n = size(state_partitions, 1)

    Xs = [
        UncertainPWARegion(
            Hyperrectangle(low=state_partitions[ii, 1, :], high=state_partitions[ii, 2, :]),
            [(M_lower[ii, :, :], b_lower[ii, :]), (M_upper[ii, :, :], b_upper[ii, :])]
        ) for ii in 1:n
    ]

    return Xs
end
module Data

using StochasticBarrierFunctions
using LazySets, SparseArrays

# Helper functions for loading data
using YAXArrays, YAXArrayBase, DimensionalData
using MAT.MAT_v4, MAT.MAT_v5, MAT.MAT_HDF5

const MatlabFile = Union{MAT_v4.Matlabv4File, MAT_v5.Matlabv5File, MAT_HDF5.MatlabHDF5File}

export load_regions, load_dynamics, load_probabilities
export create_probability_dataset, create_sparse_probability_dataset
export generate_partitions

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

    ds = YAXArrays.Dataset(regions=regions, prob=prob, prob_unsafe=prob_unsafe; properties=Dict("format"=>"dense", "dim"=>d, "num_regions"=>n))
    return ds
end

function create_probability_dataset(regions::Vector{<:Hyperrectangle}, P̲::AbstractDimArray, P̅::AbstractDimArray)
    println(typeof(P̲))
    n = length(regions)
    d = (LazySets.dim ∘ first)(regions)

    axlist = (Dim{:region}(1:n), Dim{:dir}(["lower", "upper"]), Dim{:dim}(1:d))
    l, h = stack(low.(regions); dims=1), stack(high.(regions); dims=1)
    regions = YAXArray(axlist, stack((l, h); dims=2))  # NOTE: Order of stacking is important here.
    @assert size(regions) == (n, 2, d)

    axlist = (Dim{:to}(1:n + 1), Dim{:from}(1:n), Dim{:dir}(["lower", "upper"]))
    order(A) = permutedims(A, (:to, :from))
    prob = YAXArray(axlist, stack((order(P̲), order(P̅)); dims=3))  # NOTE: Order of stacking is important here.
    @assert size(prob) == (n + 1, n, 2)

    ds = YAXArrays.Dataset(regions=regions, prob=prob; properties=Dict("format"=>"dense", "dim"=>d, "num_regions"=>n))
    return ds
end

function create_sparse_probability_dataset(regions::Vector{<:Hyperrectangle},
    P̲::AbstractDimArray{T, N, D, <:AbstractSparseMatrix},
    P̅::AbstractDimArray{T, N, D, <:AbstractSparseMatrix}) where {T, N, D}
    
    n = length(regions)
    d = (LazySets.dim ∘ first)(regions)

    axlist = (Dim{:region}(1:n), Dim{:dir}(["lower", "upper"]), Dim{:dim}(1:d))
    l, h = stack(low.(regions); dims=1), stack(high.(regions); dims=1)
    regions = YAXArray(axlist, stack((l, h); dims=2))  # NOTE: Order of stacking is important here.
    @assert size(regions) == (n, 2, d)

    axlist = (Dim{:to}(1:n + 1), Dim{:from}(1:n))
    axes = collect(map(string ∘ DimensionalData.name, axlist))

    order(A) = permutedims(A, (:to, :from))
    P̲, P̅ = order(P̲), order(P̅)
    P̲, P̅ = DimensionalData.data(P̲), DimensionalData.data(P̅)

    lower_values = YAXArray((Dim{:val_lower}(1:nnz(P̲)),), nonzeros(P̲))
    lower_row_indices = YAXArray((Dim{:val_lower}(1:nnz(P̲)),), rowvals(P̲))
    lower_col_indices = YAXArray((Dim{:col_lower}(1:length(P̲.colptr)),), P̲.colptr)

    upper_values = YAXArray((Dim{:val_upper}(1:nnz(P̅)),), nonzeros(P̅))
    upper_row_indices = YAXArray((Dim{:val_upper}(1:nnz(P̅)),), rowvals(P̅))
    upper_col_indices = YAXArray((Dim{:col_upper}(1:length(P̅.colptr)),), P̅.colptr)

    ds = YAXArrays.Dataset(regions=regions, lower_values=lower_values, lower_row_indices=lower_row_indices, lower_col_indices=lower_col_indices,
                                            upper_values=upper_values, upper_row_indices=upper_row_indices, upper_col_indices=upper_col_indices; 
        properties=Dict("format"=>"sparse", "axes"=>axes, "dim"=>d, "num_regions"=>n))
    return ds
end

function load_probabilities(dataset::YAXArrays.Dataset)
    n = dataset.properties["num_regions"]
    d = dataset.properties["dim"]
    format = get(dataset.properties, "format", "dense")

    # Pre-load data for speed
    regions = yaxconvert(DimArray, dataset.regions)
    X̲, X̅ = regions[dir=At("lower")], regions[dir=At("upper")]

    if format == "dense"
        prob = yaxconvert(DimArray, dataset.prob)
        P̲, P̅ = prob[dir=At("lower")], prob[dir=At("upper")]

        if haskey(dataset.cubes, :prob_unsafe)
            prob_unsafe = yaxconvert(DimArray, dataset.prob_unsafe)
            P̲ᵤ, P̅ᵤ = prob_unsafe[dir=At("lower")], prob_unsafe[dir=At("upper")]

            regions = [
                RegionWithProbabilities(
                    Hyperrectangle(low=Vector(copy(X̲[region=j].data)), high=Vector(copy(X̅[region=j].data))),
                    (copy(P̲[from=j].data), copy(P̅[from=j].data)),
                    (P̲ᵤ[from=j], P̅ᵤ[from=j])   # This are already scalars, no need to copy.
                )
                for j in 1:n
            ]
        else
            regions = [
                RegionWithProbabilities(
                    Hyperrectangle(low=Vector(copy(X̲[region=j].data)), high=Vector(copy(X̅[region=j].data))),
                    (copy(P̲[from=j].data), copy(P̅[from=j].data))
                )
                for j in 1:n
            ]
        end
    elseif format == "sparse"
        @assert dataset.properties["axes"] == ["to", "from"]

        lower_values = yaxconvert(DimArray, dataset.lower_values) |> DimensionalData.data |> copy
        lower_row_indices = yaxconvert(DimArray, dataset.lower_row_indices) |> DimensionalData.data |> copy
        lower_col_indices = yaxconvert(DimArray, dataset.lower_col_indices) |> DimensionalData.data |> copy
        P̲ = SparseMatrixCSC(n + 1, n, lower_col_indices, lower_row_indices, lower_values)

        upper_values = yaxconvert(DimArray, dataset.upper_values) |> DimensionalData.data |> copy
        upper_row_indices = yaxconvert(DimArray, dataset.upper_row_indices) |> DimensionalData.data |> copy
        upper_col_indices = yaxconvert(DimArray, dataset.upper_col_indices) |> DimensionalData.data |> copy
        P̅ = SparseMatrixCSC(n + 1, n, upper_col_indices, upper_row_indices, upper_values)

        regions = [
            RegionWithProbabilities(
                Hyperrectangle(low=Vector(copy(X̲[region=j].data)), high=Vector(copy(X̅[region=j].data))),
                (P̲[:, j], P̅[:, j])
            )
            for j in 1:n
        ]
    else
        throw(ArgumentError("Unknown format $format"))
    end

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
            [(convert(Matrix{Float64}, transpose(M_lower[ii, :, :])), b_lower[ii, :]), 
             (convert(Matrix{Float64}, transpose(M_upper[ii, :, :])), b_upper[ii, :])]
        ) for ii in 1:n
    ]

    return Xs
end

function load_dynamics(dataset::YAXArrays.Dataset)
    # Extract hypercube data
    n = dataset.properties["num_regions"]

    # Pre-load data for speed
    regions = yaxconvert(DimArray, dataset.regions)
    A = yaxconvert(DimArray, dataset.nominal_dynamics_A)
    b = yaxconvert(DimArray, dataset.nominal_dynamics_b)

    # Give convenient names
    X̲, X̅ = regions[dir=At("lower")], regions[dir=At("upper")]
    A̲, A̅ = permutedims(A[dir=At("lower")], (:region, :y, :x)), permutedims(A[dir=At("upper")], (:region, :y, :x))
    b̲, b̅ = b[dir=At("lower")], b[dir=At("upper")]

    Xs = [
        UncertainPWARegion(
            Hyperrectangle(low=copy(X̲[region=j].data), high=copy(X̅[region=j].data)),
            [(copy(A̲[region=j].data), copy(b̲[region=j].data)), (copy(A̅[region=j].data), copy(b̅[region=j].data))]
        ) for j in 1:n
    ]

    return Xs
end

function generate_partitions(state_space::Hyperrectangle{Float64, Vector{Float64}, Vector{Float64}}, ϵ::Vector{Float64})
    # Define ranges
    ranges = [
    range(
        state_space.center[i] - state_space.radius[i],
        step=ϵ[i],
        length=Int(ceil((2 * state_space.radius[i]) / ϵ[i])) + 1
    )
    for i in 1:length(ϵ)
    ]

    # Generate a flat vector of Hyperrectangle objects for n-dimensions
    state_partitions = [
    Hyperrectangle(
        low=[low for (low, high) in point_pairs],
        high=[high for (low, high) in point_pairs]
    )
    for point_pairs in Iterators.product([zip(r[1:end-1], r[2:end]) for r in ranges]...)
    ] |> vec

    return state_partitions
end

end # module
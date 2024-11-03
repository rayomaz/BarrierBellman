using StochasticBarrierFunctions, LazySets
using YAXArrays, NetCDF, YAML

function barrier_synthesis(yaml_file::String)

    # Load config file
    config = YAML.load_file(yaml_file)

    # Extract parameters
    system_flag = config["system_flag"]
    dim = config["dim"]

    A = hcat(config["A"]...)
    b = config["b"]
    σ = config["σ"]

    # Establish System
    state_space = Hyperrectangle(low=config["state_space"]["low"], high=config["state_space"]["high"])
    barrier_type = config["barrier_settings"]["barrier_type"]
    if barrier_type == "SOS"
        system = AdditiveGaussianLinearSystem(A, b, σ, state_space)
    elseif barrier_type == "PWC"
        system = AdditiveGaussianLinearSystem(A, b, σ)
        
        # Generate all partitions in n-dimensions
        δ = config["δ"]
        ranges = [
        range(
            state_space.center[i] - state_space.radius[i],
            step=δ[i],
            length=Int(ceil((2 * state_space.radius[i]) / δ[i])) + 1
        )
        for i in 1:dim
        ]

        # Generate a flat vector of Hyperrectangle objects for n-dimensions
        state_partitions = [
        Hyperrectangle(
            low=[low for (low, high) in point_pairs],
            high=[high for (low, high) in point_pairs]
        )
        for point_pairs in Iterators.product([zip(r[1:end-1], r[2:end]) for r in ranges]...)
        ] |> vec

        probability_bounds = transition_probabilities(system, state_partitions)
        
        # Save to a .nc file
        filename = "data/linear/$(dim)D_probability_data_$(length(state_partitions))_sigma_$σ.nc"
        savedataset(probability_bounds; path=joinpath(@__DIR__, filename), driver=:netcdf, overwrite=true)  
    end

    # Initial and Obstacle Region
    initial_region = Hyperrectangle(low=config["initial_region"]["low"], high=config["initial_region"]["high"])
    if config["obstacle_region"] == "empty"
        obstacle_region = EmptySet(dim)
    else
        initial_region = Hyperrectangle(low=config["obstacle_region"]["low"], high=config["obstacle_region"]["high"])
    end

    # Call on optimization
    time_horizon = config["barrier_settings"]["time_horizon"]
    if barrier_type == "SOS"

        # Optimize: baseline 1 (sos)
        barrier_degree = config["barrier_settings"]["barrier_degree"]
        @time res = synthesize_barrier(SumOfSquaresAlgorithm(barrier_degree=barrier_degree), 
                                           system, initial_region, obstacle_region; time_horizon=time_horizon)
        
    elseif barrier_type == "PWC"
        dataset = open_dataset(joinpath(@__DIR__, filename))
        probabilities = load_probabilities(dataset)
        optimization_type = config["barrier_settings"]["optimization_type"]
        if optimization_type == "DUAL"
            # Optimize: method 2 (dual approach)
            @time res = synthesize_barrier(DualAlgorithm(), probabilities, initial_region, obstacle_region; time_horizon = time_horizon)
        elseif optimization_type == "CEGS"
            # Optimize: method 3 (iterative approach)
            @time res = synthesize_barrier(IterativeUpperBoundAlgorithm(), probabilities, initial_region, obstacle_region; time_horizon = time_horizon)
        elseif optimization_type == "GD"
            # Optimize: method 4 (project gradient descent approach)
            @time res = synthesize_barrier(GradientDescentAlgorithm(), probabilities, initial_region, obstacle_region; time_horizon = time_horizon)
        end

    end

    # Print to txt                                   
    file_path = "results/result.txt"
    open(file_path, "w") do file
        println(file, res)
    end
    
end

# Load the system setup from YAML
barrier_synthesis(yaml_file)

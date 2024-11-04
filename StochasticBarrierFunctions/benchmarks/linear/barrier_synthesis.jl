using StochasticBarrierFunctions, LazySets
using YAXArrays, NetCDF, YAML

abstract type SystemType end
    struct LINEAR <: SystemType end
    struct POLYNOMIAL <: SystemType end
    struct NONLINEAR <: SystemType end

abstract type BarrierType end
    struct SOS <: BarrierType end
    struct PWC <: BarrierType end

abstract type PWCType end
    struct DUAL_ALG <: PWCType end
    struct CEGS_ALG <: PWCType end
    struct GD_ALG   <: PWCType end

function get_system_type(system_type_str) :: SystemType
    
    # Map the string to the correct type
    return system_type_str == "LINEAR" ? LINEAR() :
           system_type_str == "POLYNOMIAL" ? POLYNOMIAL() :
           system_type_str == "NONLINEAR" ? NONLINEAR() :
           error("Unknown system type: $system_type_str")
end

function get_barrier_type(barrier_type_str) :: BarrierType
    
    # Map the string to the correct type
    return barrier_type_str == "SOS" ? SOS() :
           barrier_type_str == "PWC" ? PWC() :
           error("Unknown barrier type: $barrier_type_str")
end

function get_pwc_optimization_type(optimization_type_str) :: PWCType
    
    # Map the string to the correct type
    return optimization_type_str == "DUAL_ALG" ? DUAL_ALG() :
           optimization_type_str == "CEGS_ALG" ? CEGS_ALG() :
           optimization_type_str == "GD_ALG" ? GD_ALG() :
           error("Unknown optimization type: $optimization_type_str")
end

function barrier_synthesis(yaml_file::String)

    # Load config file
    config = YAML.load_file(yaml_file)

    # Define optimization type
    barrier_type_instance = get_barrier_type(config["barrier_settings"]["barrier_type"])
    call_barrier_method(config, barrier_type_instance)

end

function extract_system_parms(config)

    dim, A, b, σ =  dim = config["dim"], hcat(config["A"]...), config["b"], config["σ"]
    state_space = Hyperrectangle(low=config["state_space"]["low"], high=config["state_space"]["high"])

    # Initial Region
    initial_region = Hyperrectangle(low=config["initial_region"]["low"], high=config["initial_region"]["high"])

    # Obstacle region
    obstacle_region = haskey(config, "obstacle_region") ? 
                      Hyperrectangle(low=config["obstacle_region"]["low"], high=config["obstacle_region"]["high"]) :
                      EmptySet(dim)

    # Verification discrete-time horizon
    time_horizon = config["barrier_settings"]["time_horizon"]

    return dim, A, b, σ, state_space, initial_region, obstacle_region, time_horizon

end

function generate_partitions(dim, state_space, δ)
    # Define ranges
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

    return state_partitions
end

function call_barrier_method(config, barrier_type::SOS)
    # Establish System
    dim, A, b, σ, state_space, initial_region, obstacle_region, time_horizon = extract_system_parms(config)
    system = AdditiveGaussianLinearSystem(A, b, σ, state_space)

    # Optimize: baseline 1 (sos)
    barrier_degree, lagrange_degree = get_kwargs(config, barrier_type)
    @time res_sos = synthesize_barrier(SumOfSquaresAlgorithm(barrier_degree=barrier_degree, lagrange_degree = lagrange_degree), 
                                                             system, initial_region, obstacle_region; 
                                                             time_horizon=time_horizon)

    # Print results to txt
    system_flag = config["system_flag"]
    print_to_txt(system_flag, "SOS", res_sos)
end

function call_barrier_method(config, ::PWC)
    # Establish System
    dim, A, b, σ, state_space, initial_region, obstacle_region, time_horizon = extract_system_parms(config)
    δ = config["transition_probalities"]["δ"]
    system = AdditiveGaussianLinearSystem(A, b, σ)
    state_partitions = generate_partitions(dim, state_space, δ)

    # Check if probability bounds exist, else compute and save
    system_flag = config["system_flag"]
    filename = "data/$(system_flag)/$(dim)D_probability_data_$(length(state_partitions))_δ_$(δ)_sigma_$σ.nc"
    transition_probalities_path = config["transition_probalities"]["transition_probalities_path"]
    if isfile(filename) || isfile(transition_probalities_path )
        dataset = open_dataset(joinpath(@__DIR__, filename))
        probabilities = load_probabilities(dataset)
    else
        probability_bounds = transition_probabilities(system, state_partitions)
        savedataset(probability_bounds; path=joinpath(@__DIR__, filename), driver=:netcdf, overwrite=true) 
        probabilities = load_probabilities(open_dataset(joinpath(@__DIR__, filename)))
    end
    
    optimization_type_instance = get_pwc_optimization_type(config["barrier_settings"]["optimization_type"])

    res_pwc = pwc_optimization_call(time_horizon, optimization_type_instance)
    
    # Print results to txt
    print_to_txt(system_flag, "PWC", res_pwc)
end


function pwc_optimization_call(time_horizon, ::DUAL_ALG)
    # Optimize: method 2 (dual approach)
    @time res_pwc = synthesize_barrier(DualAlgorithm(), probabilities, initial_region, obstacle_region; time_horizon = time_horizon)
    return res_pwc
end

function pwc_optimization_call(time_horizon, ::CEGS_ALG)
    # Optimize: method 3 (iterative approach)
    @time res_pwc = synthesize_barrier(IterativeUpperBoundAlgorithm(), probabilities, initial_region, obstacle_region; time_horizon = time_horizon)
    return res_pwc
end

function pwc_optimization_call(time_horizon, ::GD_ALG)
    # Optimize: method 4 (project gradient descent approach)
    @time res_pwc = synthesize_barrier(GradientDescentAlgorithm(), probabilities, initial_region, obstacle_region; time_horizon = time_horizon)
    return res_pwc
end

function get_kwargs(config, barrier_type::SOS)
    barrier_degree = get(config["barrier_settings"], "barrier_degree", SumOfSquaresAlgorithm().barrier_degree)
    lagrange_degree = get(config["barrier_settings"], "lagrange_degree", SumOfSquaresAlgorithm().lagrange_degree) 
    return barrier_degree, lagrange_degree
end

function print_to_txt(system_flag, barrier_type, res)
    # Print to txt                                   
    file_path = "results/$(system_flag)/$(barrier_type)/result.txt"
    open(file_path, "w") do file
        println(file, res)
    end
end

# Load the system setup from YAML
barrier_synthesis(yaml_file)
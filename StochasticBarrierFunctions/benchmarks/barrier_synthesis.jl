using StochasticBarrierFunctions, StochasticBarrierFunctions.Data, LazySets
using DynamicPolynomials
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
    return system_type_str == "linear"     ? LINEAR() :
           system_type_str == "polynomial" ? POLYNOMIAL() :
           system_type_str == "nonlinear"  ? NONLINEAR() :
           error("Unknown system type: $system_type_str")
end

function get_barrier_type(barrier_type_str) :: BarrierType
    return barrier_type_str == "SOS" ? SOS() :
           barrier_type_str == "PWC" ? PWC() :
           error("Unknown barrier type: $barrier_type_str")
end

function get_pwc_optimization_type(optimization_type_str) :: PWCType
    return optimization_type_str == "DUAL_ALG" ? DUAL_ALG() :
           optimization_type_str == "CEGS_ALG" ? CEGS_ALG() :
           optimization_type_str == "GD_ALG"   ? GD_ALG() :
           error("Unknown optimization type: $optimization_type_str")
end

function barrier_synthesis(yaml_file::String)
    # Load config file
    config = YAML.load_file(yaml_file)

    # Define optimization type and make call
    system_type_instance  = get_system_type(config["system_flag"])
    barrier_type_instance = get_barrier_type(config["barrier_settings"]["barrier_type"])
    call_barrier_method(config, system_type_instance, barrier_type_instance)
end

function extract_system_parms(config, system_type_str::LINEAR)
    dim, A, b, σ = config["dim"], hcat(config["A"]...), vcat(config["b"]...), config["σ"]
    state_space = Hyperrectangle(low=config["state_space"]["low"], high=config["state_space"]["high"])
    return dim, A, b, σ, state_space
end

function extract_system_parms(config, system_type_str::POLYNOMIAL)
    dim, f, σ = config["dim"], config["f"], config["σ"]
    state_space = Hyperrectangle(low=config["state_space"]["low"], high=config["state_space"]["high"])
    return dim, f, σ, state_space
end

function extract_system_parms(config, system_type_str::NONLINEAR)
    dim, σ =  config["dim"], config["σ"]
    filename = "nonlinear/$(config["filename"])"
    dataset = open_dataset(filename)
    Xs = load_dynamics(dataset)
    return dim, σ, Xs
end

function create_initial_region(config, dim)
    # Initial region should never be empty
    if haskey(config, "initial_region")
        if haskey(config["initial_region"], "low") && haskey(config["initial_region"], "high")
            return Hyperrectangle(low=config["initial_region"]["low"], high=config["initial_region"]["high"])
        elseif haskey(config["initial_region"], "c") && haskey(config["initial_region"], "r")
            center = config["initial_region"]["c"]
            radius = config["initial_region"]["r"][1]
            return Hyperrectangle(low=center .- radius, high=center .+ radius)
        else
            error("Invalid initial region configuration: expected 'low' and 'high' or 'c' and 'r'")
        end
    else
        error("Initial region configuration is required and cannot be empty.")
    end
end

function create_obstacle_region(config, dim)
    if !haskey(config, "obstacle_region")
        return EmptySet(dim)
    end
    
    num_obstacles = config["obstacle_region"]["num_obstacles"]
    obstacles = []
    for i in 1:num_obstacles
        obstacle_key = "obstacle_$i"
        obstacle_data = config["obstacle_region"][obstacle_key]
        
        if haskey(obstacle_data, "c") && haskey(obstacle_data, "r")
            # Construct a Hyperrectangle to represent Ball2
            center = obstacle_data["c"]
            radius = obstacle_data["r"][1]
            push!(obstacles, Hyperrectangle(low = center .- radius, high = center .+ radius))
        
        elseif haskey(obstacle_data, "low") && haskey(obstacle_data, "high")
            # Construct Hyperrectangle directly from low and high bounds
            push!(obstacles, Hyperrectangle(low = obstacle_data["low"], high = obstacle_data["high"]))
        
        else
            error("Invalid configuration for $obstacle_key: expected 'c' and 'r' or 'low' and 'high'")
        end
    end

    return length(obstacles) > 1 ? UnionSet(obstacles...) : obstacles[1]
end


function call_barrier_method(config, system_type_instance, barrier_type::SOS)
    # Establish System
    if system_type_instance == LINEAR()
        dim, A, b, σ, state_space = extract_system_parms(config, system_type_instance::LINEAR)
        system = AdditiveGaussianLinearSystem(A, b, σ, state_space)

    elseif system_type_instance == POLYNOMIAL()
        dim, f, σ, state_space = extract_system_parms(config, system_type_instance::POLYNOMIAL)
        f = parse_polynomial_string(f, dim)
        system = AdditiveGaussianPolySystem(f, σ, state_space)  

    elseif system_type_instance == NONLINEAR()
        dim, σ, Xs = extract_system_parms(config, system_type_instance::NONLINEAR)
        system = AdditiveGaussianUncertainPWASystem(Xs, σ)  

    else
        error("Unsupported system type instance: $system_type_instance")
    end

    # Initial Region
    initial_region = create_initial_region(config, dim)

    # Obstacle region
    obstacle_region = create_obstacle_region(config, dim)

    # Optimize: baseline 1 (sos)
    barrier_degree, lagrange_degree, time_horizon = get_kwargs(config, barrier_type)
    @time res_sos = synthesize_barrier(SumOfSquaresAlgorithm(barrier_degree=barrier_degree, lagrange_degree = lagrange_degree), 
                                                             system, initial_region, obstacle_region; 
                                                             time_horizon=time_horizon)

    # Print results to txt
    system_flag = config["system_flag"]
    print_to_txt(system_flag, "SOS", res_sos)
end

function call_barrier_method(config, system_type_instance, ::PWC)
    # Establish System
    if system_type_instance == LINEAR()
        dim, A, b, σ, state_space = extract_system_parms(config, system_type_instance::LINEAR)
        ϵ = config["transition_probalities"]["ϵ"]
        system = AdditiveGaussianLinearSystem(A, b, σ)
        state_partitions = generate_partitions(state_space, ϵ)

        filename = "$(config["system_flag"])/data/$(dim)D_probability_data_$(length(state_partitions))_δ_$(ϵ)_sigma_$σ.nc"
        transition_probalities_path = config["transition_probalities"]["transition_probalities_path"]
        if isfile(filename) || isfile(transition_probalities_path )
            dataset = open_dataset(joinpath(@__DIR__, filename))
            probabilities = load_probabilities(dataset)
        else
            probability_bounds = transition_probabilities(system, state_partitions)
            savedataset(probability_bounds; path=joinpath(@__DIR__, filename), driver=:netcdf, overwrite=true) 
            probabilities = load_probabilities(open_dataset(joinpath(@__DIR__, filename)))
        end

    elseif system_type_instance == NONLINEAR()
        filename = "$(config["system_flag"])/data/$(config["probabilities"])"
        if isfile(filename)
            dim = config["dim"]
            dataset = open_dataset(joinpath(@__DIR__, filename))
            probabilities = load_probabilities(dataset)
        else
            dim, σ, Xs = extract_system_parms(config, system_type_instance::NONLINEAR)
            system = AdditiveGaussianUncertainPWASystem(Xs, σ)
            probability_bounds = transition_probabilities(system)
            savedataset(probability_bounds; path=joinpath(@__DIR__, filename), driver=:netcdf, overwrite=true) 
            probabilities = load_probabilities(open_dataset(joinpath(@__DIR__, filename)))
        end
    else
        error("Unsupported system type instance: $system_type_instance")
    end

    # Initial Region
    initial_region = create_initial_region(config, dim)

    # Obstacle region
    obstacle_region = create_obstacle_region(config, dim)

    # Call on DUAL, CEGS or GD
    optimization_type_instance = get_pwc_optimization_type(config["barrier_settings"]["optimization_type"])
    res_pwc = pwc_optimization_call(config, probabilities, initial_region, obstacle_region, optimization_type_instance)
    
    # Print results to txt
    print_to_txt(config["system_flag"], "PWC", res_pwc)
end

function pwc_optimization_call(config, probabilities, initial_region, obstacle_region, barrier_type::DUAL_ALG)
    # Optimize: method 2 (dual approach)
    time_horizon = get_kwargs(config, barrier_type::DUAL_ALG)
    @time res_pwc = synthesize_barrier(DualAlgorithm(), 
                                       probabilities, initial_region, obstacle_region; time_horizon = time_horizon)
    return res_pwc
end

function pwc_optimization_call(config, probabilities, initial_region, obstacle_region, barrier_type::CEGS_ALG)
    # Optimize: method 3 (iterative approach)
    δ, num_iterations, barrier_guided, distribution_guided, time_horizon = get_kwargs(config, barrier_type::CEGS_ALG)
    @time res_pwc = synthesize_barrier(IterativeUpperBoundAlgorithm(δ = δ, num_iterations = num_iterations, 
                                                                    barrier_guided = barrier_guided, 
                                                                    distribution_guided = distribution_guided), 
                                       probabilities, initial_region, obstacle_region; time_horizon = time_horizon)
    return res_pwc
end

function pwc_optimization_call(config, probabilities, initial_region, obstacle_region, barrier_type::GD_ALG)
    # Optimize: method 4 (project gradient descent approach)
    num_iterations, initial_lr, decay, momentum, time_horizon = get_kwargs(config, barrier_type::GD_ALG)
    @time res_pwc = synthesize_barrier(GradientDescentAlgorithm(num_iterations = num_iterations, initial_lr = initial_lr,
                                                                decay = decay, momentum = momentum), 
                                       probabilities, initial_region, obstacle_region; time_horizon = time_horizon)
    return res_pwc
end

function parse_polynomial_string(f, dim)
    # Create the polynomials in each dimension
    global x            # Hack to make meta parsing work
    @polyvar x[1:dim]
    poly = [sum(eval(Meta.parse(f[i]))) for i in 1:length(f)]
end

function get_kwargs(config, barrier_type::SOS)
    barrier_degree = get(config["barrier_settings"], "barrier_degree", SumOfSquaresAlgorithm().barrier_degree)
    lagrange_degree = get(config["barrier_settings"], "lagrange_degree", SumOfSquaresAlgorithm().lagrange_degree) 
    time_horizon = get(config["barrier_settings"], "time_horizon", 1) 
    return barrier_degree, lagrange_degree, time_horizon
end

function get_kwargs(config, barrier_type::DUAL_ALG)
    return get(config["barrier_settings"], "time_horizon", 1) 
end

function get_kwargs(config, barrier_type::CEGS_ALG)
    δ = get(config["barrier_settings"], "iteration_δ", IterativeUpperBoundAlgorithm().δ)
    num_iterations = get(config["barrier_settings"], "num_iterations", IterativeUpperBoundAlgorithm().num_iterations)
    barrier_guided = get(config["barrier_settings"], "barrier_guided", IterativeUpperBoundAlgorithm().barrier_guided)
    distribution_guided = get(config["barrier_settings"], "distribution_guided", IterativeUpperBoundAlgorithm().distribution_guided)
    time_horizon = get(config["barrier_settings"], "time_horizon", 1) 
    return δ, num_iterations, barrier_guided, distribution_guided, time_horizon
end

function get_kwargs(config, barrier_type::GD_ALG)
    num_iterations = get(config["barrier_settings"], "num_iterations", GradientDescentAlgorithm().num_iterations)
    initial_lr = get(config["barrier_settings"], "initial_lr", GradientDescentAlgorithm().initial_lr)
    decay = get(config["barrier_settings"], "decay", GradientDescentAlgorithm().decay)
    momentum = get(config["barrier_settings"], "momentum", GradientDescentAlgorithm().momentum)
    time_horizon = get(config["barrier_settings"], "time_horizon", 1) 
    return num_iterations, initial_lr, decay, momentum, time_horizon
end

function print_to_txt(system_flag, barrier_type, res)
    # Print to txt                                   
    file_path = "$(system_flag)/results/result.txt"
    open(file_path, "w") do file
        show(file, res)
    end
end

# Load the system setup from YAML
barrier_synthesis(yaml_file)
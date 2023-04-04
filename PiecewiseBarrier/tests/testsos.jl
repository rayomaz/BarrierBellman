"""
    - Thermostat Gaussian Process Model
    - Generation of Stochastic Barrier Functions

    © Rayan Mazouz

"""

cd(@__DIR__)

# Stochastic Barrier Verification for Thermostat Gaussian Process
using Revise
using Random
using DelimitedFiles

# Read data files
state_partitions = readdlm("/partitions/state_partitions.txt", ',')

# Optimization flags
barrier_degree_input = 4
decision_η_flag = true
min_β_strategy = "max" #"sum" #"max"

#===
Initial Optimization
===#
state_space = PiecewiseBarrier.state_space_generation(state_partitions)
beta_vals_matrix = zeros(length(state_partitions), length(control_partitions))
eta_vals_matrix = zeros(length(state_partitions), length(control_partitions))

σ_noise = 0.01

for ii = 1:1#length(state_partitions)
        
    initial_state_partition = ii
    initial_control_partition = jj

    eta, beta_values = optimization(system_flag,
                                    state_space,
                                    state_partitions,
                                    control_partitions,
                                    σ_noise,
                                    barrier_degree_input,
                                    min_β_strategy,
                                    decision_η_flag,
                                    initial_state_partition,
                                    initial_control_partition,
                                    verbose=false)
                                
    beta_vals_matrix[initial_state_partition, initial_control_partition] = maximum(beta_values)
    eta_vals_matrix[initial_state_partition, initial_control_partition] = eta
end

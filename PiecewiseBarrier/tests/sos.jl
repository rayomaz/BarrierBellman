"""

    - Generation of Stochastic Barrier Functions

    © Rayan Mazouz

"""

# Stochastic Barrier Verification for Thermostat Gaussian Process
using Revise
using PiecewiseBarrier
using DelimitedFiles

# Optimization flags
barrier_degree_input = 2
decision_η_flag = true
σ_noise = 0.1
initial_state_partition = Int(3)

# State partitions
state_partitions = readdlm("partitions/test/state_partitions.txt", ',')
state_space = PiecewiseBarrier.state_space_generation(state_partitions)

# Optimization
certificate, eta, beta = sos_barrier(state_space,
                                     state_partitions,
                                     σ_noise,
                                     barrier_degree_input,
                                     initial_state_partition)




                                      


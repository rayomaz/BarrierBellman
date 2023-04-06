"""

    - Generation of Stochastic Barrier Functions

    © Rayan Mazouz

"""

# Stochastic Barrier Verification for Thermostat Gaussian Process
using Revise, BenchmarkTools

using PiecewiseBarrier
using DelimitedFiles

# Optimization flags
initial_state_partition = Int(3)

# State partitions
state_partitions = readdlm("partitions/test/state_partitions.txt", ',')
state_space = state_space_generation(state_partitions)

# Optimization
# certificate, eta, beta = 
@time sos_barrier(state_space, state_partitions, initial_state_partition)




                                     

                                      


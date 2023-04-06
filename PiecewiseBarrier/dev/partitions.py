from utilities import *
import os

def hypercubes(dimension, state_space, epsilon):
    
    utility = utilities(system_dimension=dimension,
                        state_space=state_space,
                        epsilon=epsilon)

    hyper_matrix, states = utility.hypercubes()

    return hyper_matrix

def hypercube_array(hyper_matrix, system_dimension):
    
    arrays = np.array(hyper_matrix)
    hypercubes = []

    if system_dimension > int(1):
        for i in range(0, len(arrays), system_dimension):
            hypercubes.append(np.concatenate((arrays[i], arrays[i+1])))
    elif system_dimension == int(1):
        hypercubes = arrays

    return hypercubes

def create_hyperspace(space, epsilon):
    dimension = len(space)
    hyper_matrix = hypercubes(dimension, space, epsilon)
    hyper_matrix  = hyper_matrix.reshape(-1, hyper_matrix.shape[-1])
    return hypercube_array(hyper_matrix, dimension)


""" System properties 
    1. Test
"""
system = "test"

if system == "test":
    dimension = int(1)
    # state_space = np.array([[-5.0, -3.0]])
    # state_space = np.array([[13.0, 15.0]])
    state_space = np.array([[-1.0, 1.0]])
    epsilon_state = 0.20

# Create hyperspaces
state_hypercubes = create_hyperspace(state_space, epsilon_state)
print("Number of state hypercubes: ", len(state_hypercubes))

# Save
directory = "PiecewiseBarrier/tests/partitions/"
filename_state = "state_partitions.txt"
filepath = os.path.join(directory, system, filename_state)
np.savetxt(filepath, state_hypercubes)
# StochasticBarrier.jl

We present *StochasticBarrier.jl*, an open-source Julia-based toolbox for generating Stochastic Barrier Functions (SBFs) for safety verification of discrete-time stochastic systems with additive Gaussian noise. The tool supports linear, polynomial, and piecewise affine (PWA) uncertain dynamics. The toolbox implements a Sum-of-Squares (SOS) optimization approach (https://arxiv.org/abs/2206.07811), as well as  methods based on piecewise constant (PWC) functions (https://arxiv.org/abs/2404.16986). For the class of of PWC-SBFs, three engines are offered based on: (1) Dual Linear Programming, (2) Counter Example Guided Linear Programming, and (3) Projected Gradient Descent.

## Repeat Experiments
| **`Linux`** | **`Mac OS X`** | **`Windows`** |
|-----------------|---------------------|-------------------------|

Read the description below for repeatability of all the experiments.

### Docker Image
The Dockerfile is provided in the main folder. Build this docker file to obtain all the required Julia packages, as specified in the Project.toml. To build the docker image, navigate into the main folder and run the following command 
```sh
sudo docker build -t stochastic_barrier .
```

To start a container 

```sh
sudo docker run -it --name StochasticBarrier stochastic_barrier
```

## Run through bash

Use the following commands to run the optimization case studies through bash.

```sh
stochasticbarrier contraction   # To run the stochastic contraction map

```

## Run through Julia
Use the following commands to run the optimization case studies through Julia

Navigate to *```/StochasticBarrierFunctions```* \
In terminal call julia and run the following commands:
1. ```julia
      using Pkg
   ```
2. ```julia 
      Pkg.activate(".") 
   ```
3. ```julia 
      Pkg.precompile(".") 
   ```   

To run the Contraction Map for example, use the following command: 
```julia 
   yaml_file = "benchmarks/linear/systems/contraction/sos.yaml"; include("benchmarks/barrier_synthesis.jl")
```
The same command can be run for other benchmarks.


## Citing

If the package is useful in your research, and you would like to acknowledge it, please cite the following work:

```
@article{mazouz2024piecewise,
  title={Piecewise Stochastic Barrier Functions},
  author={Mazouz, Rayan and Mathiesen, Frederik Baymler and Laurenti, Luca and Lahijanian, Morteza},
  journal={arXiv preprint arXiv:2404.16986},
  year={2024}
}


```
@article{mazouz2022safety,
  title={Safety guarantees for neural network dynamic systems via stochastic barrier functions},
  author={Mazouz, Rayan and Muvvala, Karan and Ratheesh Babu, Akash and Laurenti, Luca and Lahijanian, Morteza},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={9672--9686},
  year={2022}
}


FROM julia:1.10.1

# Copy Julia StochasticBarrierFunctions to Docker Image
COPY ./StochasticBarrierFunctions /StochasticBarrierFunctions

# Make run.bash executable
RUN chmod +x /StochasticBarrierFunctions/run.bash

# Change the workdir to package root
WORKDIR /StochasticBarrierFunctions

# Precompile the Julia Package
RUN julia -e 'using Pkg;Pkg.activate("."); Pkg.instantiate(); Pkg.precompile()'

# Add Alias to run experiment 
RUN echo 'alias stochasticbarrier="/StochasticBarrierFunctions/run.bash"' >> ~/.bashrc

# Change Entrypoint to bash (Default: Julia)
ENTRYPOINT ["bash"]
using Distributions

# Define the function f(x)
f(x) = 0.5x + rand(Normal(0, 0.01))

# Define the integration limits
xi = [-0.4,-0.2]
# xj = [0.2, 0.4]

# Define the integration step size
step = 0.001

# Initialize the transition probability
transition_prob = 0.0

# Loop over the integration limits with the given step size
for x = xi[1]:step:xi[2]
    # Compute the value of f(x)
    fx = f(x)
    
    # Compute the probability density of x
    px = pdf(Normal(0, 0.01), x)
    
    # Compute the transition probability for this step
    transition_prob += (px * step)
end

# Output the transition probability
println("Transition probability from xi = $xi to xj = $xj: $transition_prob")

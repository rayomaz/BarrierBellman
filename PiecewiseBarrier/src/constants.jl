# Optimization constants
const ϵ = 1e-6                  # Precision parameter

### Commented out as they were silently introducing errors
### The problem is that if you use different σ_noise, barrier polynomial, or lagrange_degree, 
### by defining them somewere but using these constants elsewhere then the two methods are not
### comparable. Instead take them all as parameters and be sure they are well-defined there.
# const barrier_degree  = 2       # Barrier degree
# const lagrange_degree = 2       # Lagragian multiplier degree
# const σ_noise = 0.1             # Standard deviation

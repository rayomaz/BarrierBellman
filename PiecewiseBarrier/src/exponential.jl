
function exponential_bounds(system::AdditiveGaussianPolynomialSystem{T, N}, current_state_partition) where {T, N}

    #! This is written for 1D models, will expand to ND solution_summary
    if N > 1
        error("Not implemented for N-dimensional case")
    else
        
        fx = dynamics(system)
        
        # Partition bounds
        δ⁻ = low(current_state_partition)
        δ⁺ = high(current_state_partition)

        # Compute lower and upper bounds based for exp(-1/2*(̌z - f(x))^2) 
        f⁻_exp_low(x) = exp(-1/2*(δ⁻[1] - 0.95*x)^2)        #! To do automate the dynamics to be fx (mean)
        exp⁻_min = optimize(f⁻_exp_low, δ⁻[1], δ⁺[1])
        exp⁻_min = exp⁻_min.minimum

        f⁻_exp_up(x) = -1*exp(-1/2*(δ⁻[1] - 0.95*x)^2) 
        exp⁻_max = optimize(f⁻_exp_up, δ⁻[1], δ⁺[1])
        exp⁻_max = - exp⁻_max.minimum

        # Compute lower and upper bounds based for exp(-1/2*(̂z - f(x))^2) 
        f⁺_exp_low(x) = exp(-1/2*(δ⁺[1] - 0.95*x)^2)        #! To do automate the dynamics to be fx (mean)
        exp⁺_min = optimize(f⁺_exp_low, δ⁻[1], δ⁺[1])
        exp⁺_min = exp⁺_min.minimum

        f⁺_exp_up(x) = -1*exp(-1/2*(δ⁺[1] - 0.95*x)^2) 
        exp⁺_max = optimize(f⁺_exp_up, δ⁻[1], δ⁺[1])
        exp⁺_max = - exp⁺_max.minimum

        # print(exp⁻_min)
        # print(exp⁻_max)
        # print(exp⁺_min)
        # print(exp⁺_max)

        # Arithmetic
        exp_min = minimum([exp⁻_min, exp⁺_min])
        exp_max = maximum([exp⁻_max, exp⁺_max])

        return exp_min, exp_max

    end

end
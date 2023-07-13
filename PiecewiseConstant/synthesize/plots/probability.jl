using Plots

# Plotting ibp-interval bounds on erf functions

# Probability distribution on hyperspace
partitions = readdlm("../models/linear/state_partitions.txt")
data_hyper = load("../models/linear/probability_data_5_sigma_0.01.mat")
prob_lower = data_hyper["matrix_prob_lower"]
prob_upper = data_hyper["matrix_prob_upper"]
prob_unsafe_lower = data_hyper["matrix_prob_unsafe_lower"]
prob_unsafe_upper = data_hyper["matrix_prob_unsafe_upper"]

fig = plot()
grid(true)

# Note: this example is for transition from Xj to Xi

for jj = 1:length(partitions)
    colors = [:blue, :black, :magenta, :red, :cyan]
    
    for ii = 1:1:length(partitions)
        x_space = range(minimum(partitions[jj, :]), maximum(partitions[jj, :]), length = 1000)
        
        # True erf
        prob_true = zeros(length(x_space))
        sigma = 0.01
        epsilon = 1e-6
        m = 1      # sys dim
        scale = 1/(2^m)

        vl = minimum(partitions[ii, :])
        vu = maximum(partitions[ii, :])

        # Prod of erfs
        for pp = 1:length(x_space)
            y = 0.95*x_space[pp]
            erf_low = (y - vl)/(sigma*sqrt(2))
            erf_up = (y - vu)/(sigma*sqrt(2))
            prob_true[pp] = scale*(erf(erf_low) - erf(erf_up))
            # Notice this computes Pu --> Pu = 1 - Ps
        end

        min_prob_exact = prob_lower[jj, ii]
        max_prob_exact = prob_upper[jj, ii]

        plot!(x_space, prob_true, linewidth = 3, color = :red)

        plot!(x_space, fill(min_prob_exact, length(x_space)), linewidth = 3, color = :blue)
        plot!(x_space, fill(max_prob_exact, length(x_space)), linewidth = 3, color = :black)
    end
end

labels = ["true", "lower bound", "upper bound"]
legend!(labels, loc = :ne, fontsize = 8, textcolor = :black)
xlabel!("x")
ylabel!("P")
title!("Probability distribution")

prob_upper = hcat(prob_upper, prob_unsafe_upper')
prob_lower = hcat(prob_lower, prob_unsafe_lower')

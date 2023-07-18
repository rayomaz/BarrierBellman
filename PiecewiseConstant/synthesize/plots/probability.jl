using Revise
using Plots, DelimitedFiles, MAT
using SpecialFunctions: erf

# Plotting ibp-interval bounds on erf functions

# Probability distribution on hyperspace
partitions = readdlm(joinpath(@__DIR__, "../models/linear/state_partitions.txt"))
data_hyper = matopen(joinpath(@__DIR__, "../models/linear/probability_data_5_sigma_0.01.mat"))
prob_lower = read(data_hyper, "matrix_prob_lower")
prob_upper = read(data_hyper, "matrix_prob_upper")
prob_unsafe_lower = read(data_hyper, "matrix_prob_unsafe_lower")
prob_unsafe_upper = read(data_hyper, "matrix_prob_unsafe_upper")
close(data_hyper)


# Note: this example is for transition from Xⱼ to Xᵢ

f(x) = 1.05 * x
sigma = 0.01
m = 1      # sys dim

for ii in axes(partitions, 1)
    colors = [:blue, :black, :magenta, :red, :cyan]

    x = range(minimum(partitions[:, 1]), maximum(partitions[:, 2]), length = 1000)
        
    # True prob
    vl = partitions[ii, 1]
    vu = partitions[ii, 2]

    y = f.(x)

    erf_lower = @. (y - vl) / (sigma * sqrt(2))
    erf_upper = @. (y - vu) / (sigma * sqrt(2))
    prob_true = (1 / 2^m) .* (erf.(erf_lower) - erf.(erf_upper))

    p = plot(x, prob_true, linewidth = 3, color = :red, label="true")
    
    for jj in axes(partitions, 1)
        x_space = partitions[jj, :]
        
        plot!(p, x_space, fill(prob_lower[ii, jj], 2), linewidth = 3, color = :blue, label=jj == 1 ? "lower bound" : nothing)
        plot!(p, x_space, fill(prob_upper[ii, jj], 2), linewidth = 3, color = :black, label=jj == 1 ? "upper bound" : nothing)
    end

    savefig(p, "transition_prob$ii.png")
end

# xlabel!("x")
# ylabel!("P")
# title!("Probability distribution")

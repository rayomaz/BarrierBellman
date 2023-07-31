# Plotting functions


# Optimization function
function plot_environment(regions::Vector{<:RegionWithProbabilities}, initial_region::LazySet, obstacle_region::LazySet, B::Vector{Float64})

    # Create a new plot
    p = plot(xlabel = "θ (rad)",
             ylabel = "θ̇ (rad/s)",
             title  = "Pendulum Grid",
            #  xticks = 0:20:length(regions),
            #  yticks = 0:20:length(regions),
             legend = false)

    # Construct barriers
    @inbounds for (Xⱼ, Bⱼ) in zip(regions, B)

        hypercube = region(Xⱼ)

        # Initial set
        if !isdisjoint(initial_region, region(Xⱼ))
            plot!(hypercube, alpha=0.2, color=:green)

        # Obstacle
        elseif !isdisjoint(obstacle_region, region(Xⱼ))
            plot!(hypercube, alpha=0.8, color=:red)

        else
            plot!(hypercube, alpha=0.6, color=:blue)

        end
        

    end

    display(p)

end

        

        


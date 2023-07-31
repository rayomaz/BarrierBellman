using Plots, MAT
plotlyjs()

function plot_env(partitions, initial_state)

    # Create a new plot
    p = plot(xlabel = "θ (rad)",
             ylabel = "θ̇ (rad/s)",
             title  = "Pendulum Grid",
             xticks = 0:20:120,
             yticks = 0:20:120,
             legend = false)
    
    for jj in eachindex(partitions[:, 1, 1])
        parts = partitions[jj, :, :]
        x_1_low = parts[1]
        x_1_up = parts[2]
        x_2_low = parts[3]
        x_2_up = parts[4]
        
        P = [x_1_low x_2_low;
             x_1_low x_2_up;
             x_1_up  x_2_up;
             x_1_up  x_2_low]
        
        if jj in initial_state
            plot!(P[:, 1], P[:, 2], alpha=0.8, color=:red)
        elseif jj == obstacle_state
            plot!(P[:, 1], P[:, 2], alpha=0.2, color=:green)
        else
            plot!(P[:, 1], P[:, 2], alpha=0.6, color=:blue)
        end
    end

    display(p)
end

# Load data files
data_hyper = matopen("../../models/pendulum/partition_data_120.mat")
partitions = read(file, "partitions")
close(file)

initial_state = [55, 56, 65, 66]

# Plot the grid
plot_env(partitions, initial_state)

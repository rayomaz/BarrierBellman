using Plots, MAT
plotlyjs()

function plot_env(partitions, initial_state)
    # Define custom colors for different regions
    colors = ["#FF0000", "#00FF00", "#0000FF"]
    
    # Create a new plot
    p = plot(legend = false)
    
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
            plot!(P[:, 1], P[:, 2], alpha=0.8, color=colors[1])
        elseif jj == obstacle_state
            plot!(P[:, 1], P[:, 2], alpha=0.2, color=colors[2])
        else
            plot!(P[:, 1], P[:, 2], alpha=0.6, color=colors[3])
        end
    end
    
    # Plot properties
    xlabel!("θ (rad)")
    ylabel!("θ̇ (rad/s)")
    title!("Pendulum Grid")
    xticks!(0:20:120)
    yticks!(0:20:120)
    xgrid!
    ygrid!
    # legend = false

    display(p)
end

# Load data files
data_hyper = matread("../models/pendulum/partition_data_120.mat")

# Define state space
partitions = data_hyper["partitions"]
initial_state = [55, 56, 65, 66]

# Plot the grid
plot_env(partitions, initial_state)

%% Plotting pendulum grid
clc; clear; close all

% Load data files
data_hyper = load('../models/pendulum/partition_data_120.mat');

% Define state space
partitions = data_hyper.partitions;
initial_state = 65;
obstacle_state = false;

% Plot the grid
plot_env(partitions, initial_state, obstacle_state)

%% Functions
function plots = plot_env(partitions, initial_state, obstacle_state)

    figure
    
    for jj = 1:length(partitions)

        parts = partitions(jj, :, :);

        x_1_low = parts(1);
        x_1_up = parts(2);

        x_2_low = parts(3);
        x_2_up = parts(4);

        P = [x_1_low x_2_low; x_1_low x_2_up; x_1_up x_2_up; x_1_up x_2_low];

        if jj == initial_state
            plots = patch(P(:,1),P(:,2), 0.8, 'EdgeAlpha', 0.1);
        elseif jj == obstacle_state
            plots = patch(P(:,1),P(:,2), 0.2, 'EdgeAlpha', 0.1);
        else
            plots = patch(P(:,1),P(:,2), 0.6, 'EdgeAlpha', 0.1);
        end

        
    end
    
    % Plot properties
    c = colormap(jet(256));
    shading faceted
    caxis([0 1])
    colormap(flipud(c));

    % Label
    xlabel("$\theta$ (rad)", 'Interpreter','latex', "FontSize", 25)
    ylabel('$\dot{\theta}$ (rad/s)', 'Interpreter','latex', "FontSize", 25)
    set(gcf,'color','w');
    set(gca,'FontSize',20)
    
end

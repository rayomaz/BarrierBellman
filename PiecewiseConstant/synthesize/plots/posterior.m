%% Show NNDM linear bounds

clear; clc; close all;

% Load NNDM dynamics
load("../models/pendulum/partition_data_480.mat")
system_dimension = 2; 

% for parts = 1:number_hypercubes
for parts = 50 % Pick a hypercube number 
        
    % Explicit bounds for each hypercube
    M_h_ii = M_h(parts, :, :);
    M_l_ii = M_l(parts, :, :);
    B_h_ii = B_h(parts, :);
    B_l_ii = B_l(parts, :);

    % Figure initialize
    figure(parts)

    % Loop over vertices
    x_k_hcube_bound = partitions(parts, :, :);

    for vertices = 1:4

        if vertices == 1
            x_k_lower = min(x_k_hcube_bound(:,:,1));
            x_k_upper = min(x_k_hcube_bound(:,:,2));
        elseif vertices == 2
            x_k_lower = max(x_k_hcube_bound(:,:,1));
            x_k_upper = min(x_k_hcube_bound(:,:,2));
        elseif vertices == 3
            x_k_lower = min(x_k_hcube_bound(:,:,1));
            x_k_upper = max(x_k_hcube_bound(:,:,2));
        elseif vertices == 4
            x_k_lower = max(x_k_hcube_bound(:,:,1));
            x_k_upper = max(x_k_hcube_bound(:,:,2));
        end

        % Define current vertex
        x_vertex = [x_k_lower; x_k_upper];

        % Explicit dimension
        k_upper_explicit = zeros(system_dimension, 1);
        k_lower_explicit = zeros(system_dimension, 1);
        
        for kk = 1:system_dimension
    
            % Loop range
            range_length = system_dimension;
            loop_range = ((kk-1)*range_length+1):(kk*range_length);
            
            % Loop over hcube higher bound
            hyper_matrix_higher = M_h_ii(loop_range) * x_vertex + B_h_ii(kk);
            k_upper_explicit(kk, :) = hyper_matrix_higher;
    
            % Loop over hcube lower bound
            hyper_matrix_lower = M_l_ii(loop_range) * x_vertex + B_l_ii(kk);
            k_lower_explicit(kk, :) = hyper_matrix_lower;
         
        end
    
        % Visual confirmation rectangle
        dim_1 = 1;
        dim_2 = 2;
        x_explicit = k_lower_explicit(dim_1,1);
        y_explicit = k_lower_explicit(dim_2,1);
        w_explicit = k_upper_explicit(dim_1,1) - k_lower_explicit(dim_1,1);
        h_explicit = k_upper_explicit(dim_2,1) - k_lower_explicit(dim_2,1);
        
        rectangle_explicit = [x_explicit, y_explicit, w_explicit , h_explicit];
        
        % Plot
        plot(x_vertex(dim_1), x_vertex(dim_2), 'k.', 'MarkerSize', 50);
        hold on
        grid on
        
        if  w_explicit == 0
            plot(x_explicit, y_explicit, 'r.', 'MarkerSize', 50);
        else
            rectangle('Position', rectangle_explicit, 'EdgeColor', 'r', 'LineWidth', 3);
        end
        plot(x_vertex(dim_1), x_vertex(dim_2), 'r') 

    end

    % Visual confirmation rectangle
%     k_upper_global = zeros(system_dimension, 1);
%     k_lower_global = zeros(system_dimension, 1);

    % Global bounds each dimension
    k_upper_global = [state_space(1, 2), state_space(2, 2)];
    k_lower_global = [state_space(1, 1), state_space(2, 1)];

    % Create state space rectangle
    dim_1 = 1;
    dim_2 = 2;
    x_global = k_lower_global(1,dim_1);
    y_global = k_lower_global(1,dim_2);
    w_global = k_upper_global(1,dim_1) - k_lower_global(1,dim_1);
    h_global = k_upper_global(1,dim_2) - k_lower_global(1,dim_2);

    rectangle_global = [x_global, y_global, w_global , h_global];
    rectangle('Position', rectangle_global, 'EdgeColor', 'g', 'LineWidth', 3);


    legend('Vertex Points', 'Posterior', 'Location', 'Best')  
    
    
end

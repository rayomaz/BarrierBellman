%% Plotting beta distribution

clc; clear; close all

% Read data files
system = "linear"; % "pendulum", "cartpole", etc.

if system == "pendulum"
    data_hyper = load('../models/pendulum/partition_data_120.mat');
    partitions = data_hyper.partitions;
elseif system == "linear"
    partitions = load('../models/linear/state_partitions.txt');
end

stringData_certificate = read_file('../probabilities/beta.txt');
stringData_updated = read_file('../probabilities/beta_updated.txt');
stringData_dual = read_file('../probabilities/beta_dual.txt');

% Read probability values from data files
array_prob_certificate = extract_data(stringData_certificate);
array_prob_updated = extract_data(stringData_updated);
array_prob_dual = extract_data(stringData_dual);
max_prob_certificate = max(array_prob_certificate);

% Plot the grid and probability distribution
if system == "pendulum"
    plot_certifcate = plot_data(array_prob_certificate, partitions, max_certificate, "Upper bound");
    plot_dual = plot_data(array_prob_dual, partitions, max_certificate, "Dual");
elseif system == "linear"
    plot_certifcate = plot_data_linear_1D(array_prob_certificate, partitions, "Upper bound");
    plot_updated = plot_data_linear_1D(array_prob_updated, partitions, "Updated"); 
    plot_dual = plot_data_linear_1D(array_prob_dual, partitions, "Dual"); 

    % Adding text
    fontSize = 16;  % Adjust the font size as needed
    text(0, 0.12, 'Upper bound', 'Color', 'red', 'FontSize', fontSize);
    text(0, 0.08, "Updated", 'Color', 'black', 'FontSize', fontSize);
    text(0, 0.04, "Dual", 'Color', 'blue', 'FontSize', fontSize);
end


%% Functions
function stringData = read_file(file_name)
    FID = fopen(file_name);
    data = textscan(FID,'%s');
    fclose(FID);
    stringData = string(data{:});
end

function array_prob = extract_data(stringData)

    array_prob = zeros(1, length(stringData));

    for ii = 1:length(stringData)  
    
        string_ii = stringData{ii};
    
        if ii == 1
            string_ii = string_ii(2:end);
        end
    
        if ii == length(stringData)
            string_ii = string_ii(1:end-1);
        end

        array_prob(ii) = str2double(string_ii);
    
    end

end


function plots = plot_data_linear_1D(array_prob, partitions, title_type)

    if title_type == "Upper bound"
        figure
        hold on
    elseif title_type == "Dual" || title_type == "Updated"
        hold on
        grid on
    end
    
    for jj = 1:length(partitions)

        parts = partitions(jj, :, :);

        x_low = parts(1);
        x_up = parts(2);

        x_range = linspace(x_low, x_up, 100);
        p_range = array_prob(jj)*ones(1, length(x_range));

        if title_type == "Upper bound"
            plots = plot(x_range, p_range, 'r*', "LineWidth", 3);
        elseif title_type == "Dual"
            plots = plot(x_range, p_range, 'b-', "LineWidth", 3);
        elseif title_type == "Updated"
            plots = plot(x_range, p_range, 'k+', "LineWidth", 3);
        end
        
    end
   
    xlabel("$x$", 'Interpreter','latex', "FontSize", 25)
    ylabel('$\beta$', 'Interpreter','latex', "FontSize", 25)
    title("Beta")
    set(gcf,'color','w');
    set(gca,'FontSize',20)
    
end

function plots = plot_data(array_prob, partitions, max_prob, title_type)

    figure
    
    for jj = 1:length(partitions)

        parts = partitions(jj, :, :);

        x_1_low = parts(1);
        x_1_up = parts(2);

        x_2_low = parts(3);
        x_2_up = parts(4);

        P = [x_1_low x_2_low; x_1_low x_2_up; x_1_up x_2_up; x_1_up x_2_low];

        plots = patch(P(:,1),P(:,2), array_prob(jj), 'EdgeAlpha', 0.1);
        
    end
    
    % Plot properties
    c = gray;
    a = colorbar;
    shading faceted
    caxis([0 max(max_prob)])
    colormap(flipud(c));
    hold on
   
    % Plot box around
    delta = 0;
    x1_min = -deg2rad(12) - delta;
    x1_max =  deg2rad(12) + delta;
    
    x2_min = -deg2rad(57.27) - delta;
    x2_max = deg2rad(57.27) + delta;
    
    w =  x1_max - x1_min;
    h = x2_max - x2_min;
  
    recentangle_array = [x1_min x2_min w h]; 
    
    rectangle('Position', recentangle_array, 'LineWidth', 1.0)
    
    hold off

    xlabel("$\theta$ (rad)", 'Interpreter','latex', "FontSize", 25)
    ylabel('$\dot{\theta}$ (rad/s)', 'Interpreter','latex', "FontSize", 25)
    ylabel(a,'Pr','Rotation',0, 'Interpreter','latex', "FontSize", 25);
    title("Beta", title_type)
    set(gcf,'color','w');
    set(gca,'FontSize',20)
    xlim([x1_min x1_max])
    ylim([x2_min x2_max])
    
end



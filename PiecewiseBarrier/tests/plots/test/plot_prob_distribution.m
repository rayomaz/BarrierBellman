% Plot probability distribution

clc; clear; close all;

%% Read data files
system = "thermostat/";
path = "../../partitions/";
state_ranges = load(join([path,system,"state_partitions.txt"], ""));
control_ranges = load(join([path,system,"control_partitions.txt"], ""));
beta_ranges = load(join([path,system,"beta_magnitudes.txt"], ""));
eta_ranges = load(join([path,system,"eta_magnitudes.txt"], ""));
max_prob_certificate = 1;

%% Call the function
plot_data(state_ranges, control_ranges, beta_ranges, eta_ranges)

%% Plot the grid function
function plots = plot_data(state_ranges, control_ranges, beta_ranges, eta_ranges)

    figure
    
    for jj = 1:length(state_ranges)

        for zz = 1:length(control_ranges)
    
            x_1_low = state_ranges(jj, 1);
            x_1_up = state_ranges(jj, 2);
    
            x_2_low = control_ranges(zz, 1);
            x_2_up = control_ranges(zz, 2);
    
            P = [x_1_low x_2_low; x_1_low x_2_up; x_1_up x_2_up; x_1_up x_2_low];
            prob = beta_ranges(jj, zz) + eta_ranges(jj, zz);
            plots = patch(P(:,1),P(:,2), prob, 'EdgeAlpha', 0.1);
            
        end
    end
    
    % Plot properties
    c = gray;
    shading faceted
    colormap(flipud(c));
    a = colorbar;
    hold on
 
    xlabel("T ($\circ$ C)", 'Interpreter','latex', "FontSize", 15)
    ylabel('u', 'Interpreter','latex', "FontSize", 15)
    ylabel(a,'Pr','Rotation',0, 'Interpreter','latex', "FontSize", 15);

end






%% Plotting barrier pendulum 3D

clc; clear; close all

% Read data files
data_hyper = load('../models/pendulum/partition_data_120.mat');
partitions = data_hyper.partitions;

% Extract barrier txt files
stringData_certificate = read_file('../probabilities/barrier.txt');
stringData_dual = read_file('../probabilities/barrier_dual.txt');

% Read barrier values from data files
array_barrier = extract_data(stringData_certificate);
array_barrier_dual = extract_data(stringData_dual);
max_certificate = 1;

% Plot barriers
% plot_certifcate = plot_data(array_certificate, partitions, max_certificate, "Upper bound");
% plot_dual = plot_data(array_dual, partitions, max_certificate, "Dual");

figure
hold on

for jj = 1:length(partitions)

    parts = partitions(jj, :, :);

    x1 = linspace(parts(1), parts(2), 2);
    x2 = linspace(parts(3), parts(4), 2);

    % Create 2D grids for x1 and x2
    [X1, X2] = meshgrid(x1, x2);

%     B = array_barrier(jj) * ones(1, length(x1));
%     B = array_barrier_dual(jj) * ones(1, length(x1));
    B = array_barrier(jj) * ones(length(x2), length(x1));

%     plot3(x1, x2, B, "b", "LineWidth", 2);
    surf(X1, X2, B);

end

view(3)
grid on
xlabel("$\theta$ (rad)", 'Interpreter','latex', "FontSize", 12)
ylabel('$\dot{\theta}$ (rad/s)', 'Interpreter','latex', "FontSize", 12)
zlabel('B','Rotation',0, 'Interpreter','latex', "FontSize", 12);




% Functions
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

% function plots = plot_data(array_barrier, partitions, max_barrier, title_type)

    
    % Plot properties
%     c = gray;
%     a = colorbar;
%     shading faceted
%     caxis([0 max(max_prob)])
%     colormap(flipud(c));
%     hold on
%    
%     % Plot box around
%     delta = 0;
%     x1_min = -deg2rad(12) - delta;
%     x1_max =  deg2rad(12) + delta;
%     
%     x2_min = -deg2rad(57.27) - delta;
%     x2_max = deg2rad(57.27) + delta;
%     
%     w =  x1_max - x1_min;
%     h = x2_max - x2_min;
%   
%     recentangle_array = [x1_min x2_min w h]; 
%     
%     rectangle('Position', recentangle_array, 'LineWidth', 1.0)
%     
%     hold off
% 

%     title("Barrier", title_type)
%     set(gcf,'color','w');
%     set(gca,'FontSize',20)
%     xlim([x1_min x1_max])
%     ylim([x2_min x2_max])
% %     
% end
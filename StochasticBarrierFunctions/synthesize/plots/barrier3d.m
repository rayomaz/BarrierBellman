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
figure
hold on
plot_flag = "upper"; % "dual"; 

for jj = 1:length(partitions)

    parts = partitions(jj, :, :);

    % Extract the coordinates from the 'parts' array
    x1_min = parts(1);
    x1_max = parts(2);
    x2_min = parts(3);
    x2_max = parts(4);

    % Get barrier value (height)
    if plot_flag == "dual"
        B_val = array_barrier_dual(jj);
    elseif plot_flag == "upper"
        B_val = array_barrier(jj);
    end
    
    % Define the vertices of the rectangle
    vertices = [x1_min, x2_min, 0;
                x1_max, x2_min, 0;
                x1_max, x2_max, 0;
                x1_min, x2_max, 0;
                x1_min, x2_min, B_val;
                x1_max, x2_min, B_val;
                x1_max, x2_max, B_val;
                x1_min, x2_max, B_val;
                ];
    
    % Create the face index matrix for each face
    bottom_face = [1, 2, 3, 4];
    top_face = [5, 6, 7, 8];
    
    % Side faces are trapezoids (each face has 4 vertices)
    side_faces = [
        1, 2, 6, 5; % Face 1: Vertex 1, Vertex 2, Top Vertex 2, Vertex 1
        2, 3, 7, 6; % Face 2: Vertex 2, Vertex 3, Top Vertex 3, Vertex 2
        3, 4, 8, 7; % Face 3: Vertex 3, Vertex 4, Top Vertex 4, ertex 3
        4, 1, 5, 8  % Face 4: Vertex 4, Vertex 1, Top Vertex 1, Vertex 4
    ];
    
    % Combine all vertices and faces
    all_vertices = vertices;
    all_faces = [bottom_face; top_face; side_faces];

    % Set the color to transparent gray (RGB with alpha)
    gray_color = [0.5, 0.5, 0.5]; % RGB values for gray
    alpha_value = 0.5; % Set transparency to 0.5 (adjust as needed)

    
    % Plot the 3D filled rectangle using patch function
    patch('Vertices', all_vertices, 'Faces', all_faces, ...
          'FaceColor', gray_color, 'FaceAlpha', alpha_value);

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
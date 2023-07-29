using Plots, MAT
plotlyjs()

function read_file(file_name)
    data = read(file_name, String)
    stringData = split(data, '\n')
    return stringData
end

function extract_data(stringData)
    array_prob = Float64[]
    for string_ii in stringData
        # Remove unwanted characters "[" and ","
        string_ii = replace(string_ii, r"[\[\],]" => "")
        # Split the string by spaces and parse individual values
        values = split(string_ii)
        for value in values
            push!(array_prob, parse(Float64, value))
        end
    end
    return array_prob
end

using FileIO

function plot_barriers(partitions, array_barrier, array_barrier_dual, plot_flag)
    
    
    p = plot3d(legend=false)

    for jj in eachindex(partitions[:, 1, 1])
        parts = partitions[jj, :, :]

        # Extract the coordinates from the 'parts' array
        x1_min = parts[1, 1]
        x1_max = parts[1, 2]
        x2_min = parts[2, 1]
        x2_max = parts[2, 2]

        # Get barrier value (height)
        B_val = plot_flag == "dual" ? array_barrier_dual[jj] : array_barrier[jj]

        # Define the coordinates of the four vertices in 3D space
        vertices3D = [(x1_min, x2_min, 0),  # Vertex 1: (x1_min, x2_min, 0)
        (x1_max, x2_min, 0),  # Vertex 2: (x1_max, x2_min, 0)
        (x1_max, x2_max, 0),  # Vertex 3: (x1_max, x2_max, 0)
        (x1_min, x2_max, 0),  # Vertex 4: (x1_min, x2_max, 0)
        (x1_min, x2_min, B_val),  # Vertex 5: (x1_min, x2_min, B_val)
        (x1_max, x2_min, B_val),  # Vertex 6: (x1_max, x2_min, B_val)
        (x1_max, x2_max, B_val),  # Vertex 7: (x1_max, x2_max, B_val)
        (x1_min, x2_max, B_val)]  # Vertex 8: (x1_min, x2_max, B_val)

        # Define the faces (connectivity) of the rectangle
        faces = [(1, 2, 6, 5),  # Bottom face (vertices 1, 2, 6, 5)
        (2, 3, 7, 6),  # Side face (vertices 2, 3, 7, 6)
        (3, 4, 8, 7),  # Top face (vertices 3, 4, 8, 7)
        (4, 1, 5, 8),  # Side face (vertices 4, 1, 5, 8)
        (1, 2, 3, 4),  # Front face (vertices 1, 2, 3, 4)
        (5, 6, 7, 8)]  # Back face (vertices 5, 6, 7, 8)

        # Extract x, y, and z coordinates from the vertices
        x_coords = [v[1] for v in vertices3D]
        y_coords = [v[2] for v in vertices3D]
        z_coords = [v[3] for v in vertices3D]

        # Plot the 3D rectangle
        plotly()  # You can choose another backend if desired, like gr(), pyplot(), etc.
        plot!(x_coords[[1, 2, 6, 5, 1]], y_coords[[1, 2, 6, 5, 1]], z_coords[[1, 2, 6, 5, 1]], fill= true, fillalpha=0.6, label="Bottom")
        plot!(x_coords[[2, 3, 7, 6, 2]], y_coords[[2, 3, 7, 6, 2]], z_coords[[2, 3, 7, 6, 2]], fill= true, fillalpha=0.6, label="Side")
        plot!(x_coords[[3, 4, 8, 7, 3]], y_coords[[3, 4, 8, 7, 3]], z_coords[[3, 4, 8, 7, 3]], fill= true, fillalpha=0.6, label="Top")
        plot!(x_coords[[4, 1, 5, 8, 4]], y_coords[[4, 1, 5, 8, 4]], z_coords[[4, 1, 5, 8, 4]], fill= true, fillalpha=0.6, label="Side")
        plot!(x_coords[[1, 2, 3, 4, 1]], y_coords[[1, 2, 3, 4, 1]], z_coords[[1, 2, 3, 4, 1]], fill= true, fillalpha=0.6, label="Front")
        plot!(x_coords[[5, 6, 7, 8, 5]], y_coords[[5, 6, 7, 8, 5]], z_coords[[5, 6, 7, 8, 5]], fill= true, fillalpha=0.6, label="Back")

    end

    # Set the color to transparent gray (RGB with alpha)
    gray_color = RGB(0.5, 0.5, 0.5) # RGB values for gray
    alpha_value = 0.5 # Set transparency to 0.5 (adjust as needed)

    # Plot the 3D filled rectangles using plot function
    # plot3d!(all_vertices, all_faces, color=gray_color, alpha=alpha_value)
    xlabel!("θ (rad)")
    ylabel!("θ̇ (rad/s)")
    zlabel!("B")
    title!("Barrier Pendulum 3D")
    display(p)
end


# Read data files
data_hyper = matread("../models/pendulum/partition_data_120.mat")
partitions = data_hyper["partitions"]

# Extract barrier txt files
stringData_certificate = read_file("../probabilities/barrier.txt")
stringData_dual = read_file("../probabilities/barrier_dual.txt")

# Read barrier values from data files
array_barrier = extract_data(stringData_certificate)
array_barrier_dual = extract_data(stringData_dual)
max_certificate = 1

# Plot barriers
plot_flag = "upper" # "dual" or "upper"
plot_barriers(partitions, array_barrier, array_barrier_dual, plot_flag)




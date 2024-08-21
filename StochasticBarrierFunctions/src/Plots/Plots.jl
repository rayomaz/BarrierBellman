module Plots 

using StochasticBarrierFunctions, LazySets

using Plots

export plot_environment, plot_3d_barrier

# Optimization function
function plot_environment(regions::Vector{<:RegionWithProbabilities}, initial_region::LazySet, obstacle_region::LazySet, B::Vector{Float64})

    # Create a new plot
    p = plot(xlabel = "θ (rad)",
             ylabel = "θ̇ (rad/s)",
             title  = "Pendulum Grid",
            #  xticks = 0:20:length(regions),
            #  yticks = 0:20:length(regions),
             legend = false)

    # Construct environment
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


function plot_3d_barrier(regions::Vector{<:RegionWithProbabilities}, B::Vector{Float64})
    
    p = plot(xlabel = "θ (rad)",
             ylabel = "θ̇ (rad/s)",
             zlabel = "B",
             title  = "Barrier Pendulum 3D",
             legend = false)

    # Construct barriers
    @inbounds for (jj, (Xⱼ, Bⱼ)) in enumerate(zip(regions, B))

        hypercube = region(Xⱼ)
        vₗ = low(hypercube)
        vᵤ = high(hypercube)

        x1_min = vₗ[1]
        x1_max = vᵤ[1]

        x2_min = vₗ[2]
        x2_max = vᵤ[2]

        # Get barrier value (height)
        B_val = B[jj]

        # Define the coordinates of the four vertices in 3D space
        vertices3D = [(x1_min, x2_min, 0),      # Vertex 1: (x1_min, x2_min, 0)
                      (x1_max, x2_min, 0),      # Vertex 2: (x1_max, x2_min, 0)
                      (x1_max, x2_max, 0),      # Vertex 3: (x1_max, x2_max, 0)
                      (x1_min, x2_max, 0),      # Vertex 4: (x1_min, x2_max, 0)
                      (x1_min, x2_min, B_val),  # Vertex 5: (x1_min, x2_min, B_val)
                      (x1_max, x2_min, B_val),  # Vertex 6: (x1_max, x2_min, B_val)
                      (x1_max, x2_max, B_val),  # Vertex 7: (x1_max, x2_max, B_val)
                      (x1_min, x2_max, B_val)]  # Vertex 8: (x1_min, x2_max, B_val)

        # Define the faces (connectivity) of the rectangle
        faces = [[1, 2, 6, 5, 1],  # Bottom face (vertices 1, 2, 6, 5) --> the last element closes the polygon
                [2, 3, 7, 6, 2],   # Side face (vertices 2, 3, 7, 6)
                [3, 4, 8, 7, 3],   # Top face (vertices 3, 4, 8, 7)
                [4, 1, 5, 8, 4],   # Side face (vertices 4, 1, 5, 8)
                [1, 2, 3, 4, 1],   # Front face (vertices 1, 2, 3, 4)
                [5, 6, 7, 8, 5]]   # Back face (vertices 5, 6, 7, 8)

        # Extract x, y, and z coordinates from the vertices
        x_coords = [v[1] for v in vertices3D]
        y_coords = [v[2] for v in vertices3D]
        z_coords = [v[3] for v in vertices3D]

        # Plot the 3D rectangle
        plot!(x_coords[faces[1, :]], y_coords[faces[1, :]], z_coords[faces[1, :]], fill= true, fillalpha=0.6, label="Bottom")
        plot!(x_coords[faces[2, :]], y_coords[faces[2, :]], z_coords[faces[2, :]], fill= true, fillalpha=0.6, label="Side")
        plot!(x_coords[faces[3, :]], y_coords[faces[3, :]], z_coords[faces[3, :]], fill= true, fillalpha=0.6, label="Top")
        plot!(x_coords[faces[4, :]], y_coords[faces[4, :]], z_coords[faces[4, :]], fill= true, fillalpha=0.6, label="Side")
        plot!(x_coords[faces[5, :]], y_coords[faces[5, :]], z_coords[faces[5, :]], fill= true, fillalpha=0.6, label="Front")
        plot!(x_coords[faces[6, :]], y_coords[faces[6, :]], z_coords[faces[6, :]], fill= true, fillalpha=0.6, label="Back")

    end

    display(p)     

end

end # module
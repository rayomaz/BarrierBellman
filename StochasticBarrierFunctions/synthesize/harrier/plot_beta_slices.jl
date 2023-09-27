using Plots, LazySets
import ColorSchemes.magma

project(X, ids) = Hyperrectangle(LazySets.center(X)[ids], radius_hyperrectangle(X)[ids])
labels = ["x", "y", "theta", "xdot", "ydot", "thetadot"]
c = LazySets.center(region(probabilities[12385]))  # Insert the fixed region id (typical argmax_j beta_j).

for x in 1:5
    for y in (x + 1):6
        vary_ids = [x, y]
        fixed_ids = setdiff(1:6, vary_ids)

        plot_regions = filter(j -> LazySets.center(region(probabilities[j]))[fixed_ids] ≈ c[fixed_ids], 1:length(probabilities))

        p = plot(; xlabel=labels[x], ylabel=labels[y], legend=false, size=(600, 600))

        for j in plot_regions
            plot!(p, project(region(probabilities[j]), vary_ids), color=get(magma, beta_pgd[j]))
        end

        savefig(p, "figures/harrier_beta_$(x)_$(y).png")
    end
end

for x in 1:5
    for y in (x + 1):6
        vary_ids = [x, y]
        fixed_ids = setdiff(1:6, vary_ids)

        plot_regions = filter(j -> LazySets.center(region(probabilities[j]))[fixed_ids] ≈ c[fixed_ids], 1:length(probabilities))

        p = plot(; xlabel=labels[x], ylabel=labels[y], legend=false, size=(600, 600))

        for j in plot_regions
            plot!(p, project(region(probabilities[j]), vary_ids), color=get(magma, B_pgd.b[j]))
        end

        savefig(p, "figures/harrier_B_$(x)_$(y).png")
    end
end
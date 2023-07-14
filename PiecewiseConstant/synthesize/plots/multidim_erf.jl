using Revise
using Plots
# plotlyjs()
gr()

using SpecialFunctions: erf

# Plotting 2d erf functions
m = 2
x = -3:0.01:3
y = -3:0.01:3
    
# True prob
vlx = -0.5
vux = 0.5
vly = -1.5
vuy = 1.5

erf_term(μ, v, σ) = erf((μ - v) / (σ * sqrt(2)))
prob_true(x, y) = (1 / 2^m) * (erf_term(x, vlx, 1.0)  - erf_term(x, vux, 1.0)) * (erf_term(y, vly, 1.0) - erf_term(y, vuy, 1.0))

# p = plot(x, y, prob_true, st=:surface, camera=(30, 30), xlabel="x", ylabel="y", zlabel="P", title="Probability distribution")
# display(p)

p = plot(x, y, prob_true, st=:contour, aspect_ratio=:equal, xlim=(-3, 3), color=:thermal, xlabel="x", ylabel="y", zlabel="P", title="T(Hyperrectangle([$vlx, $vly], [$vux, $vuy]) | x)", size=(800, 800))
display(p)

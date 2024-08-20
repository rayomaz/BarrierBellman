using Revise
using Plots
plotlyjs()
# gr()

using SpecialFunctions: erf

# Plotting 2d erf functions
m = 2
    
# True prob
vlx = -0.5
vux = 0.5
vly = -0.5
vuy = 0.5

erf_term(μ, v, σ) = erf((μ - v) / (σ * sqrt(2)))
prob_true(x, y) = (1 / 2^m) * (erf_term(x, vlx, 1.0)  - erf_term(x, vux, 1.0)) * (erf_term(y, vly, 1.0) - erf_term(y, vuy, 1.0))

# p = plot(x, y, prob_true, st=:surface, camera=(30, 30), xlabel="x", ylabel="y", zlabel="P", title="Probability distribution")
# display(p)

x = -1.1:0.01:1.1
y = -1.1:0.01:1.1
p = plot(x, y, prob_true, st=:surface, color=:thermal, seriesalpha=0.4, xlabel="x", ylabel="y", zlabel="P", title="T(Hyperrectangle([$vlx, $vly], [$vux, $vuy]) | x)", size=(800, 800))


t = 0:0.001:2π
x = sin.(t)
y = cos.(t)
P = prob_true.(x, y)
plot!(p, x, y, P)

t = collect(-1:0.001:1)
n = length(t)
x = [t ones(n) -t -ones(n)]
y = [-ones(n) t ones(n) -t]
P = prob_true.(x, y)
plot!(p, x, y, P)

display(p)

p = plot(x, y, P, xlabel="x", ylabel="y", zlabel="P", title="T(Hyperrectangle([$vlx, $vly], [$vux, $vuy]) | x)", size=(800, 800))
display(p)

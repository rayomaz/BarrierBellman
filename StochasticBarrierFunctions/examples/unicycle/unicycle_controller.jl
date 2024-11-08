using Revise
using ControlSystems, LinearAlgebra, Plots
using ModelingToolkit, DifferentialEquations

@variables t x(t) y(t) θ(t) v(t) u₁(t) u₂(t)
D = Differential(t)

# Dynamics (https://doi.org/10.1016/S1474-6670(17)38011-4)
# ẋ = v cos(θ)
# ẏ = v sin(θ)
# θ̇ = ω

# Choose ODE formulation z = (x, y, θ, v)
# ẋ = z₄ cos z₃
# ẏ = z₄ sin z₃
# θ̇ = - [(sin z₃) / z₄ ] * u₁ + [(cos z₃) / z₄] * u₂
# v̇ = cos z₃ * u₁ + sin z₃ * u₂

eqs = [
    D(x) ~ v * cos(θ),
    D(y) ~ v * sin(θ),
    D(θ) ~ - (sin(θ) / v) * u₁ + (cos(θ) / v) * u₂,
    D(v) ~ cos(θ) * u₁ + sin(θ) * u₂,
    u₁ ~ u₁,
    u₂ ~ u₂
]

@named sys = ODESystem(eqs, t)

# Linearize
operating_point = Dict(x => 0, y => 0, θ => 1e-5, v => 1e-6, u₁ => 0, u₂ => 0)
matrices, simplified_sys = linearize(sys, [u₁, u₂], [x, y, θ, v]; op=operating_point)
A, B, C, D = matrices

lti_sys = ss(A, B, C, D)

# Discretize in time with zero-order hold
Ts = 0.1
dsys = c2d(lti_sys, Ts)

# Design LQR controller
Q = Diagonal([1e-2, 1e-1, 1e-1, 1e-4])
ρ = 1.0
R = ρ * I(2)

F = lqr(dsys, Q, R)

# Construct closed-loop system
u = [u₁, u₂]
z = [x, y, θ, v]
eqs = [
    eqs;
    u .~ -F * z
]
@named controlled_sys = ODESystem(eqs, t)
controlled_sys = structural_simplify(controlled_sys)

Te = 20
tspan = (0, Te)

x0 = Dict(x => -0.5, y => 0, θ => deg2rad(3), v => 1e-6)

prob = ODEProblem(controlled_sys, x0, tspan)
sol = solve(prob; saveat=Ts)

p1 = plot(sol, idxs=[(0, u₁), (0, u₂)], title="Control")
p2 = plot(sol, idxs=[(0, θ)], title="Angle")
p3 = plot(sol, idxs=[(0, x), (0, y)], title="Positions")
p4 = plot(sol, idxs=[(0, v)], title="Velocity")
p = plot(p1, p2, p3, p4, layout=(4, 1), size=(600, 1200))
display(p)
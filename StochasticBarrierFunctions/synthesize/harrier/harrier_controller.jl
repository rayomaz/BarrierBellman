using Revise
using ControlSystems, LinearAlgebra, Plots
using ModelingToolkit, DifferentialEquations

@parameters m=4 J=0.0475 r=0.25 g=9.81 c=0.05
@variables t x(t) y(t) θ(t) ẋ(t) ẏ(t) θ̇(t) F₁(t) F₂(t) u₁(t) u₂(t)
D = Differential(t)

# Dynamics:
# mẍ = F₁ cos θ - F₂ sin θ - cẋ
# mÿ = F₁ sin θ + F₂ cos θ - mg - cẏ
# Jθ̈ = rF₁

# Choose ODE formulation z = (x, y, θ, ẋ, ẏ, θ̇)
# ẋ = z₄
# ẏ = z₅
# θ̇ = z₆
# ẍ = (u₁ cos z₃ - u₂ sin z₃ - mg sin z₃ - cz₄) / m
# ÿ = (u₁ sin z₃ + u₂ cos z₃ + mg (cos z₃ - 1) - cz₅) / m
# θ̈ = r u₁ / J

# u₁ = F₁
# u₂ = F₂ - mg

eqs = [
    D(x) ~ ẋ,
    D(y) ~ ẏ,
    D(θ) ~ θ̇,
    m * D(ẋ) ~ F₁ * cos(θ) - F₂ * sin(θ) - c * ẋ,
    m * D(ẏ) ~ F₁ * sin(θ) + F₂ * cos(θ) - m * g - c * ẏ,
    J * D(θ̇) ~ r * F₁,
    u₁ ~ F₁,
    u₂ ~ F₂ - m * g
]

@named sys = ODESystem(eqs, t)

# Linearize
operating_point = Dict(x => 0, y => 0, θ => 0, ẋ => 0, ẏ => 0, θ̇ => 0, u₁ => 0, u₂ => 0)
matrices, simplified_sys = linearize(sys, [u₁, u₂], [x, y, θ, ẋ, ẏ, θ̇]; op=operating_point)
A, B, C, D = matrices

lti_sys = ss(A, B, C, D)

# Discretize in time with zero-order hold
Ts = 0.1
dsys = c2d(lti_sys, Ts)

# Design LQR controller with Q = diag(1e-2, 1e-2, 100, 1e-4, 1e-4, 1), R = 1I
Q = Diagonal([1e-2, 1e-2, 100, 1e-4, 1e-4, 1])
ρ = 1.0
R = ρ * I(2)

F = lqr(dsys, Q, R)
display(F)

# Construct closed-loop system
u = [u₁, u₂]
z = [x, y, θ, ẋ, ẏ, θ̇]
eqs = [
    eqs;
    u .~ -F * z
]
@named controlled_sys = ODESystem(eqs, t)
controlled_sys = structural_simplify(controlled_sys)

Te = 20
tspan = (0, Te)

# x0 = Dict(x => 0, y => 0, θ => 0, ẋ => 0, ẏ => 0, θ̇ => 0)
# x0 = Dict(x => 1, y => -1, θ => deg2rad(2), ẋ => 0, ẏ => 0, θ̇ => 0)
x0 = Dict(x => 10, y => 10, θ => deg2rad(12), ẋ => 3, ẏ => 3, θ̇ => 1)

prob = ODEProblem(controlled_sys, x0, tspan)
sol = solve(prob; saveat=Ts)

p1 = plot(sol, idxs=[(0, u₁), (0, u₂)], title="Control")
p2 = plot(sol, idxs=[(0, θ), (0, θ̇)], title="Angle and angular velocity")
p3 = plot(sol, idxs=[(0, x), (0, y)], title="Positions")
p4 = plot(sol, idxs=[(0, ẋ), (0, ẏ)], title="Velocities")
p = plot(p1, p2, p3, p4, layout=(4, 1), size=(600, 1200))
display(p)

using Revise
using ControlSystems, LinearAlgebra, Plots

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

# System description
m = 4
J = 0.0475
r = 0.25
g = 9.81
c = 0.05

A = [
    0 0 0 1 0 0;
    0 0 0 0 1 0;
    0 0 0 0 0 1;
    0 0 -g -c/m 0 0;
    0 0 0 0 -c/m 0;
    0 0 0 0 0 0
]
B = [0 0; 0 0; 0 0; 1/m 0; 0 1/m; r/J 0]
C = I
D = 0

sys = ss(A, B, C, D)

Ts = 0.1
dsys = c2d(sys, Ts)

function solve_lqr(Q, R)
    F = lqr(dsys, Q, R)
    u(x, t) = -F * x
    
    Te = 20
    x0 = [1; -1; 20 * 2 * π / 360; 0; 0; 0]
    x0 = [10.0; 10.0; deg2rad(12); 3.0; 3.0; 1.0]
    res = lsim(dsys, u, Te; x0=x0)
    
    p1 = plot(res.t, res.u', title="Control")
    p2 = plot(res.t, res.y[[3, 6], :]', title="States")
    p = plot(p1, p2, layout=(2, 1))
    display(p)

    return F
end

# Design LQR controller with Q = diag(1e-2, 1e-2, 100, 1e-4, 1e-4, 1), R = 1I
Q = Diagonal([1e-2, 1e-2, 100, 1e-4, 1e-4, 1])
ρ = 1.0
R = ρ * I(2)
display(solve_lqr(Q, R))
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

Ts = 0.01

A = I + Ts .* [0 0 0 1 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1; 0 0 -g -c/m 0 0; 0 0 0 0 -c/m 0; 0 0 0 0 0 0]
B = Ts .* [0 0; 0 0; 0 0; 1/m 0; 0 1/m; r/J 0]
C = [1 0 0 0 0 0; 0 1 0 0 0 0; 0 0 1 0 0 0; 0 0 0 0 0 0; 0 0 0 0 0 0]
D = [0 0; 0 0; 0 0; 1 0; 0 1]

sys = ss(A, B, C, D, Ts)

function solve_lqr(Q, R)
    F = lqr(Discrete, A, B, Q, R)
    u(x, t) = -F * x
    
    Te = 7
    t = 0:Ts:Te
    x0 = [1; -1; 20 * 2 * π / 360; 0; 0; 0]
    y = lsim(sys, u, t, x0)[1]
    
    p1 = plot(t, y[4:5, :]', title="Control")
    p2 = plot(t, y[1:3, :]', title="First 3 States")
    p = plot(p1, p2, layout=(2, 1))
    display(p)

    return F
end

# Design LQR controller with Q = diag(10, 100, 1, 1, 1, 1), R = 0.1I
Q = Diagonal([10, 100, 1, 1, 1, 1])
ρ = 0.1
R = ρ * I(2)
display(solve_lqr(Q, R))
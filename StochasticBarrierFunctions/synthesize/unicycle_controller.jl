using Revise
using ControlSystems, LinearAlgebra, Plots

# Dynamics(https://doi.org/10.1016/S1474-6670(17)38011-4)
# ẋ = v cos(θ)
# ẏ = v sin(θ)
# θ̇ = ω

# Choose ODE formulation z = (x, y, θ, v)
# ẋ = z₄ cos z₃
# ẏ = z₄ sin z₃
# θ̇ = - [(sin z₃) / z₄ ] * u₁ + [(cos z₃) / z₄] * u₂
# v̇ = cos z₃ * u₁ + sin z₃ * u₂

# System description
A = [
    0 1 0 0;
    0 0 0 0;
    0 0 0 1;
    0 0 0 0
]
B = [0 0; 1 0; 0 0; 0 1]
C = I
D = 0

sys = ss(A, B, C, D)

Ts = 0.1
dsys = c2d(sys, Ts)

function solve_lqr(Q, R)
    F = lqr(dsys, Q, R)
    u(x, t) = -F * x
    
    Te = 10
    x0 = [1; -1; 20 * 2 * π / 360; 0]
    res = lsim(dsys, u, Te; x0=x0)
    
    p1 = plot(res.t, res.u', title="Control")
    p2 = plot(res.t, res.y[[1, 2], :]', title="States")
    p = plot(p1, p2, layout=(2, 1))
    display(p)

    return F
end

# Design LQR controller with Q = diag(1e-6, 1e-4, 1e-6, 1e-4), R = 1I
Q = Diagonal([1e-6, 1e-4, 1e-6, 1e-4])
ρ = 1.0
R = ρ * I(2)
display(solve_lqr(Q, R))

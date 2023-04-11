import cvxpy as cp

x_low = 0.0
x_up = 1.0
x = cp.Variable()
f = cp.exp(-1/2*(x_low + 0.95*x)**2) #- cp.exp(-1/2*(x_up + 0.95*x)**2)

problem = cp.Problem(cp.Minimize(-f), [0 <= x, x <= 1])
print("Is problem a DQCP?", problem.is_dqcp())

problem = cp.Problem(cp.Maximize(f), [x_low <= x, x <= x_up])
problem.solve(solver=cp.SCS, qcp=True)  #, verbose=True
maximum = f.value

print(maximum)

import numpy as np

x_low = 0.2
x_up = 0.3
alpha = 1e-10 # learning rate
tolerance = 1e-6 # stopping criterion

def f(x):
    mu1 = 0.95 * x
    return np.exp(-1/2*(x_low + mu1)**2)

def df(x):
    mu1 = 0.95 * x
    return -1/2 * (x_low + mu1) * 0.95 * np.exp(-1/2*(x_low + mu1)**2)

x = np.random.uniform(x_low, x_up) # initial guess

while True:
    grad = df(x)
    x_new = x - alpha * grad
    if np.abs(x_new - x) < tolerance:
        break
    x = x_new

minimum = f(x)

print(minimum)

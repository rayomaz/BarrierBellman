# using NLopt

# function myfunc(x::Vector, grad::Vector)
#     if length(grad) > 0
#         grad[1] = 0
#         grad[2] = 0.5/sqrt(x[2])
#     end
#     return sqrt(x[2])
# end

# function myconstraint(x::Vector, grad::Vector, a, b)
#     if length(grad) > 0
#         grad[1] = 3a * (a*x[1] + b)^2
#         grad[2] = -1
#     end
#     (a*x[1] + b)^3 - x[2]
# end

# opt = Opt(:LD_MMA, 2)
# opt.lower_bounds = [-Inf, 0.]
# opt.xtol_rel = 1e-4

# opt.min_objective = myfunc
# inequality_constraint!(opt, (x,g) -> myconstraint(x,g,2,0), 1e-8)
# inequality_constraint!(opt, (x,g) -> myconstraint(x,g,-1,1), 1e-8)

# (minf,minx,ret) = optimize(opt, [1.234, 5.678])
# numevals = opt.numevals # the number of function evaluations
# println("got $minf at $minx after $numevals iterations (returned $ret)")


using JuMP
using NLopt

model = Model(NLopt.Optimizer)
set_optimizer_attribute(model, "algorithm", :LD_MMA)

a1 = 2
b1 = 0
a2 = -1
b2 = 1

@variable(model, x1)
@variable(model, x2 >= 0)

@NLobjective(model, Min, sqrt(x2))
@NLconstraint(model, x2 >= (a1*x1+b1)^3)
@NLconstraint(model, x2 >= (a2*x1+b2)^3)

set_start_value(x1, 1.234)
set_start_value(x2, 5.678)

JuMP.optimize!(model)

println("got ", objective_value(model), " at ", [value(x1), value(x2)])
using Test

@testset verbose = true "StochasticBarrierFunctions" begin
    test_files = ["abstraction.jl", "beta.jl", "dual.jl", "upper_bound.jl", "iterative_upper_bound.jl", "gradient_descent.jl", "sos.jl"]
    for f in test_files
        @testset verbose = true "$f" include(f)
    end
end
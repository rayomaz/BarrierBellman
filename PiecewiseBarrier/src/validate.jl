""" Barrier synthesis validation functions

    © Rayan Mazouz

"""

# Check non-negativity of barrier certificate in the state_space
function nonnegative_barrier(certificate, state_space, system_dimension)

    barrier_nonnegative = find_minimum(certificate, state_space, system_dimension)
    if barrier_nonnegative >= -ϵ
        print_green("Test passed: B(x) >= $(value(barrier_nonnegative)) ∈ Xs")
    else
        print_red("Barrier invalid: non-negative condition ∈ Xs invalid")
    end

end

# Evaluate barrier at boundary of safe - unsafe regions
function unsafe_barrier(certificate, state_space, system_dimension)

    barrier_unsafe = evaluate_function(certificate, state_space, system_dimension)

    if all(x -> x > 1, barrier_unsafe) >= 1+ϵ || all(x -> x > 1, barrier_unsafe) <= 1+ϵ
        print_green("Test passed: B(x) >= $(value(1.0)) ∈ Xu")
    else
        print_red("Barrier invalid: unsafe condition ∈ Xu invalid")
    end

end

# Minimum finder for n-dimensional state-space
function find_minimum(f, intervals, system_dimension)

    if system_dimension == 1
        result = optimize(f, intervals[1], intervals[2])
        return result.minimum
    else
        error("Write code for higher-dimensional system here ...")
    end
end

# Evaluate function at given x
function evaluate_function(f, intervals, system_dimension)

    boundary_values = []
    if system_dimension == 1
        for jj ∈ eachindex(intervals)
            result = f(intervals[jj])
            push!(boundary_values, result)
        end
        return boundary_values
    else
        error("Write code for higher-dimensional system here ...")
    end
end

# Validate text print
function print_green(text)
    print("\n", "\033[32m$(text)\033[0m")
end

# Validate text print
function print_red(text)
    print("\n", "\033[31m$(text)\033[0m")
end

function print_blue(text)
    print("\n", "\033[34m$(text)\033[0m")
end

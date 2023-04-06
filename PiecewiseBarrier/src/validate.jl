""" Barrier synthesis validation functions

    © Rayan Mazouz

"""

# Check non-negativity of barrier certificate on given interval
function nonnegative_barrier(certificate)


    if certificate < 0
        error("Invalid barrier: violiation of non-negativity term ")
        return Int(1)

    else
        print("Non-negativity constraint valid ")
        return 0
    end

end
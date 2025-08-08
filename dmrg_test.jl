using StatsBase

include("tt_cross.jl")

function dmrg_test()
    (order, n) = (7, 5)
    input_tensor = zeros(Tuple(fill(n, order)))
    for idx in CartesianIndices(input_tensor)
        i = j = 1
        for row in 1:floor(Int64,order/2)
            i += (idx[row] - 1) ^ row
        end
        for col in floor(Int64,order/2)+1:order
            j += (idx[col] - 1) ^ (order - col + 1)
        end
        input_tensor[idx] = 1 / (i + j - 1)
    end
    # psi = dmrg_cross(input_tensor, maxr, cutoff, tol, maxiter)
    psi = tt_cross(input_tensor, maxr, tol, maxiter)

    println(psi)
    psi_prod = deepcopy(psi[1])
    for i in 2:order
        psi_prod *= psi[i]
    end
    output_tensor = Array{Float64, order}(psi_prod, siteinds(psi))

    println(maximum(output_tensor - input_tensor))
    println(minimum(output_tensor - input_tensor))
    println(rmsd(output_tensor, input_tensor))
    println(norm(output_tensor - input_tensor) / norm(output_tensor))
end

# maxr = 100
maxr = 10
cutoff = 1.0e-6
tol = 0.01
maxiter = 10
dmrg_test()

using ITensors
using ITensorMPS
using LinearAlgebra
using HDF5

function tensor_train_cross(input_tensor, maxrank::Int64, cutoff::Float64, tol::Float64, n_iter_max::Int64, seedlist::Vector{Vector{Int64}}=Vector{Int64}[])
    tensor_shape = size(input_tensor)
    tensor_order = ndims(input_tensor)

    rank = fill(1, tensor_order - 1)

    sites = [siteind(tensor_shape[i], i) for i in 1:tensor_order]
    factor = randomMPS(sites)
    links = linkinds(factor)
    if !isempty(seedlist)
        rank = fill(length(seedlist), tensor_order - 1)

        for i in eachindex(sites)
            if i < tensor_order
                links[i] = Index(rank[i], "Link,l=$i")
            end
            if i == 1
                factor[i] = ITensor(links[i], sites[i])
            elseif i == tensor_order
                factor[i] = ITensor(links[i - 1], sites[i])
            else
                factor[i] = ITensor(links[i - 1], sites[i], links[i])
            end
        end

        for i in eachindex(seedlist)
            x = seedlist[i, :][]
            factor[1][sites[1]=>x[1], links[1]=>i] = 1.0
            for k in 2:tensor_order-1
                factor[k][links[k-1]=>i, sites[k]=>x[k], links[k]=>i] = 1.0
            end
            factor[tensor_order][links[tensor_order-1]=>i, sites[tensor_order]=>x[tensor_order]] = 1.0
        end
    end

    P = Vector{Matrix}(undef, tensor_order + 1)
    P[tensor_order + 1] = ones(1, 1)
    row_idx = Vector(undef, tensor_order - 1)
    col_idx = Vector(undef, tensor_order - 1)
    Q, R, links[tensor_order - 1] = qr(factor[tensor_order], sites[tensor_order]; tags=tags(links[tensor_order - 1]))
    factor[tensor_order] = deepcopy(Q)
    Qmat = Matrix(Q, links[tensor_order - 1], sites[tensor_order])
    J = maxvol(Matrix(transpose(Qmat)))
    col_idx[tensor_order - 1] = deepcopy(J)
    P[tensor_order] = Qmat[:, J]

    for k in tensor_order-1:-1:2
        factor[k] *= R
        Q, R, links[k - 1] = qr(factor[k], sites[k], links[k]; tags=tags(links[k - 1]))
        Q *= ITensor(P[k + 1], links[k], links[k]')
        noprime!(Q)
        factor[k] = deepcopy(Q)
        comb = combiner(links[k], sites[k])
        Qmat = Matrix(comb * Q, links[k - 1], combinedind(comb))
        J = maxvol(Matrix(transpose(Qmat)))
        new_idx = [
            [floor(Int64, (idx - 1) / rank[k]) + 1, mod(idx - 1, rank[k]) + 1] for idx in J
        ]
        next_col_idx = [
            [[jc[1]]; col_idx[k][jc[2]]] for jc in new_idx
        ]
        col_idx[k - 1] = next_col_idx
        P[k] = Qmat[:, J]
    end

    factor[1] *= R * ITensor(P[2], links[1], links[1]')
    noprime!(factor[1])
    comb = combiner(links[1], sites[1])
    Qmat = permutedims(Vector(comb * factor[1]))
    J = maxvol(Matrix(transpose(Qmat)))
    P[1] = Qmat[:, J]

    iter = 0
    not_converged = true

    for iter in 1:n_iter_max
        if !not_converged
            break
        end
        not_converged = false
        
        if left_right_ttcross(input_tensor, rank, row_idx, col_idx, factor, P, maxrank, cutoff, tol, iter)
            not_converged = true
        end
        if right_left_ttcross(input_tensor, rank, row_idx, col_idx, factor, P, maxrank, cutoff, tol, iter)
            not_converged = true
        end
        f = h5open("tt_cross.h5", "w")
        write(f, "factor$iter", factor)
        close(f)
    end

    if iter >= n_iter_max
        println("Maximum number of iterations reached.")
        flush(stdout)
    end
    if not_converged
        println("Low Rank Approximation algorithm did not converge.")
        flush(stdout)
    end

    return factor
end

function left_right_ttcross(input_tensor, rank::Vector{Int64}, row_idx, col_idx, factor::MPS, P::Vector{Matrix}, maxrank::Int64, cutoff::Float64, tol::Float64, iter::Int64)
    tensor_shape = size(input_tensor)
    tensor_order = ndims(input_tensor)

    sites = siteinds(factor)
    links = linkinds(factor)

    not_converged = false

    bond = ITensor(sites[1], sites[2], links[2])
    for i in 1:tensor_shape[1]
        for j in 1:tensor_shape[2]
            for ridx in 1:rank[2]
                idx = [[i, j]; col_idx[2][ridx]]
                bond[sites[1]=>i, sites[2]=>j, links[2]=>ridx] = input_tensor[idx...]
            end
        end
    end
    bond *= ITensor(inv(P[3]), links[2], links[2]')
    noprime!(bond)
    error = norm(bond - factor[1] * factor[2]) / norm(bond)
    if error > tol / sqrt(tensor_order - 1)
        not_converged = true
    end

    U, S, V = svd(bond, sites[1]; cutoff=cutoff, maxdim=maxrank, lefttags=tags(links[1]))
    links[1] = commonind(U, S)
    rank[1] = dim(links[1])
    println([S[i, i] for i in 1:rank[1]])
    factor[1] = deepcopy(U)
    factor[2] = S * V
    Umat = Matrix(U, sites[1], links[1])
    I = maxvol(Umat)
    row_idx[1] = deepcopy(I)
    P[2] = Umat[I, :]

    println("Right sweep $iter step 1 error: $error")
    flush(stdout)

    for k in 2:tensor_order-2
        bond = ITensor(links[k - 1], sites[k], sites[k + 1], links[k + 1])
        for lidx in 1:rank[k - 1]
            for i in 1:tensor_shape[k]
                for j in 1:tensor_shape[k + 1]
                    for ridx in 1:rank[k + 1]
                        idx = [row_idx[k - 1][lidx]; [i, j]; col_idx[k + 1][ridx]]
                        bond[links[k - 1]=>lidx, sites[k]=>i, sites[k+1]=>j, links[k+1]=>ridx] = input_tensor[idx...]
                    end
                end
            end
        end
        bond *= ITensor(inv(P[k]), links[k - 1]', links[k - 1]) * ITensor(inv(P[k + 2]), links[k + 1], links[k + 1]')
        noprime!(bond)
        error = norm(bond - factor[k] * factor[k + 1]) / norm(bond)
        if error > tol / sqrt(tensor_order - 1)
            not_converged = true
        end

        U, S, V = svd(bond, links[k - 1], sites[k]; cutoff=cutoff, maxdim=maxrank, lefttags=tags(links[k]))
        links[k] = commonind(U, S)
        rank[k] = dim(links[k])
        println([S[i, i] for i in 1:rank[k]])
        factor[k] = deepcopy(U)
        factor[k + 1] = S * V
        U *= ITensor(P[k], links[k - 1]', links[k - 1])
        noprime!(U)
        comb = combiner(sites[k], links[k - 1])
        Umat = Matrix(comb * U, combinedind(comb), links[k])
        I = maxvol(Umat)
        new_idx = [
            [floor(Int64, (idx - 1) / tensor_shape[k]) + 1, mod(idx - 1, tensor_shape[k]) + 1] for idx in I
        ]
        next_row_idx = [
            [row_idx[k - 1][ic[1]]; [ic[2]]] for ic in new_idx
        ]
        row_idx[k] = next_row_idx
        P[k + 1] = Umat[I, :]

        println("Right sweep $iter step $k error: $error")
        flush(stdout)
    end

    bond = ITensor(links[tensor_order - 2], sites[tensor_order - 1], sites[tensor_order])
    for lidx in 1:rank[tensor_order - 2]
        for i in 1:tensor_shape[tensor_order - 1]
            for j in 1:tensor_shape[tensor_order]
                idx = [row_idx[tensor_order - 2][lidx]; [i, j]]
                bond[links[tensor_order-2]=>lidx, sites[tensor_order-1]=>i, sites[tensor_order]=>j] = input_tensor[idx...]
            end
        end
    end
    bond *= ITensor(inv(P[tensor_order - 1]), links[tensor_order - 2]', links[tensor_order - 2]) * inv(P[tensor_order + 1])[]
    noprime!(bond)
    error = norm(bond - factor[tensor_order - 1] * factor[tensor_order]) / norm(bond)
    if error > tol / sqrt(tensor_order - 1)
        not_converged = true
    end

    U, S, V = svd(bond, links[tensor_order - 2], sites[tensor_order - 1]; cutoff=cutoff, maxdim=maxrank, lefttags=tags(links[tensor_order - 1]))
    links[tensor_order - 1] = commonind(U, S)
    rank[tensor_order - 1] = dim(links[tensor_order - 1])
    println([S[i, i] for i in 1:rank[tensor_order - 1]])
    factor[tensor_order - 1] = deepcopy(U)
    factor[tensor_order] = S * V
    U *= ITensor(P[tensor_order - 1], links[tensor_order - 2]', links[tensor_order - 2])
    noprime!(U)
    comb = combiner(sites[tensor_order - 1], links[tensor_order - 2])
    Umat = Matrix(comb * U, combinedind(comb), links[tensor_order - 1])
    I = maxvol(Umat)
    new_idx = [
        [floor(Int64, (idx - 1) / tensor_shape[tensor_order - 2]) + 1, mod(idx - 1, tensor_shape[tensor_order - 2]) + 1] for idx in I
    ]
    next_row_idx = [
        [row_idx[tensor_order - 2][ic[1]]; [ic[2]]] for ic in new_idx
    ]
    row_idx[tensor_order - 1] = next_row_idx
    P[tensor_order] = Umat[I, :]

    println("Right sweep $iter step $(tensor_order - 1) error: $error")
    flush(stdout)

    core = ITensor(links[tensor_order - 1], sites[tensor_order])
    for lidx in 1:rank[tensor_order - 1]
        for i in 1:tensor_shape[tensor_order]
            idx = [row_idx[tensor_order - 1][lidx]; [i]]
            core[links[tensor_order-1]=>lidx, sites[tensor_order]=>i] = input_tensor[idx...]
        end
    end
    # error = norm(core - factor[tensor_order]) / norm(core)
    # if error > tol / sqrt(tensor_order - 1)
    #     not_converged = true
    # end
    # factor[tensor_order] = deepcopy(core)
    comb = combiner(sites[tensor_order], links[tensor_order - 1])
    Umat = Matrix(transpose(permutedims(Vector(comb * core))))
    I = maxvol(Umat)
    P[tensor_order + 1] = Umat[I, :]

    # println("Right sweep $iter step $tensor_order error: $error")
    println("Link dims: $(linkdims(factor))")
    flush(stdout)

    return not_converged
end

function right_left_ttcross(input_tensor, rank::Vector{Int64}, row_idx, col_idx, factor::MPS, P::Vector{Matrix}, maxrank::Int64, cutoff::Float64, tol::Float64, iter::Int64)
    tensor_shape = size(input_tensor)
    tensor_order = ndims(input_tensor)

    sites = siteinds(factor)
    links = linkinds(factor)

    not_converged = false

    bond = ITensor(links[tensor_order - 2], sites[tensor_order - 1], sites[tensor_order])
    for lidx in 1:rank[tensor_order - 2]
        for i in 1:tensor_shape[tensor_order - 1]
            for j in 1:tensor_shape[tensor_order]
                idx = [row_idx[tensor_order - 2][lidx]; [i, j]]
                bond[links[tensor_order - 2]=>lidx, sites[tensor_order - 1]=>i, sites[tensor_order]=>j] = input_tensor[idx...]
            end
        end
    end
    bond *= ITensor(inv(P[tensor_order - 1]), links[tensor_order - 2]', links[tensor_order - 2])
    noprime!(bond)
    error = norm(bond - factor[tensor_order - 1] * factor[tensor_order]) / norm(bond)
    if error > tol / sqrt(tensor_order - 1)
        not_converged = true
    end

    U, S, V = svd(bond, sites[tensor_order]; cutoff=cutoff, maxdim=maxrank, lefttags=tags(links[tensor_order - 1]))
    links[tensor_order - 1] = commonind(U, S)
    rank[tensor_order - 1] = dim(links[tensor_order - 1])
    println([S[i, i] for i in 1:rank[tensor_order - 1]])
    factor[tensor_order - 1] = S * V
    factor[tensor_order] = deepcopy(U)
    Umat = Matrix(U, links[tensor_order - 1], sites[tensor_order])
    J = maxvol(Matrix(transpose(Umat)))
    col_idx[tensor_order - 1] = deepcopy(J)
    P[tensor_order] = Umat[:, J]

    println("Left sweep $iter step $tensor_order error: $error")
    flush(stdout)

    for k in tensor_order-2:-1:2
        bond = ITensor(links[k - 1], sites[k], sites[k + 1], links[k + 1])
        for lidx in 1:rank[k - 1]
            for i in 1:tensor_shape[k]
                for j in 1:tensor_shape[k + 1]
                    for ridx in 1:rank[k + 1]
                        idx = [row_idx[k - 1][lidx]; [i, j]; col_idx[k + 1][ridx]]
                        bond[links[k-1]=>lidx, sites[k]=>i, sites[k+1]=>j, links[k + 1]=>ridx] = input_tensor[idx...]
                    end
                end
            end
        end
        bond *= ITensor(inv(P[k]), links[k - 1]', links[k - 1]) * ITensor(inv(P[k + 2]), links[k + 1], links[k + 1]')
        noprime!(bond)
        error = norm(bond - factor[k] * factor[k + 1]) / norm(bond)
        if error > tol / sqrt(tensor_order - 1)
            not_converged = true
        end

        U, S, V = svd(bond, sites[k + 1], links[k + 1]; cutoff=cutoff, maxdim=maxrank, lefttags=tags(links[k]))
        links[k] = commonind(U, S)
        rank[k] = dim(links[k])
        println([S[i, i] for i in 1:rank[k]])
        factor[k] = S * V
        factor[k + 1] = deepcopy(U)
        U *= ITensor(P[k + 2], links[k + 1], links[k + 1]')
        noprime!(U)
        comb = combiner(links[k + 1], sites[k + 1])
        Umat = Matrix(comb * U, links[k], combinedind(comb))
        J = maxvol(Matrix(transpose(Umat)))
        new_idx = [
            [floor(Int64, (idx - 1) / rank[k + 1]) + 1, mod(idx - 1, rank[k + 1]) + 1] for idx in J
        ]
        next_col_idx = [
            [[jc[1]]; col_idx[k + 1][jc[2]]] for jc in new_idx
        ]
        col_idx[k] = next_col_idx
        P[k + 1] = Umat[:, J]

        println("Left sweep $iter step $(k + 1) error: $error")
        flush(stdout)
    end

    bond = ITensor(sites[1], sites[2], links[2])
    for i in 1:tensor_shape[1]
        for j in 1:tensor_shape[2]
            for ridx in 1:rank[2]
                idx = [[i, j]; col_idx[2][ridx]]
                bond[sites[1]=>i, sites[2]=>j, links[2]=>ridx] = input_tensor[idx...]
            end
        end
    end
    bond *= inv(P[1])[] * ITensor(inv(P[3]), links[2], links[2]')
    noprime!(bond)
    error = norm(bond - factor[1] * factor[2]) / norm(bond)
    if error > tol / sqrt(tensor_order - 1)
        not_converged = true
    end

    U, S, V = svd(bond, sites[2], links[2]; cutoff=cutoff, maxdim=maxrank, lefttags=tags(links[1]))
    links[1] = commonind(U, S)
    rank[1] = dim(links[1])
    println([S[i, i] for i in 1:rank[1]])
    factor[1] =  S * V
    factor[2] = deepcopy(U)
    U *= ITensor(P[3], links[2], links[2]')
    noprime!(U)
    comb = combiner(links[2], sites[2])
    Umat = Matrix(comb * U, links[1], combinedind(comb))
    J = maxvol(Matrix(transpose(Umat)))
    new_idx = [
        [floor(Int64, (idx - 1) / rank[2]) + 1, mod(idx - 1, rank[2]) + 1] for idx in J
    ]
    next_col_idx = [
        [[jc[1]]; col_idx[2][jc[2]]] for jc in new_idx
    ]
    col_idx[1] = next_col_idx
    P[2] = Umat[:, J]

    println("Left sweep $iter step 2 error: $error")
    flush(stdout)

    core = ITensor(sites[1], links[1])
    for i in 1:tensor_shape[1]
        for ridx in 1:rank[1]
            idx = [[i]; col_idx[1][ridx]]
            core[sites[1]=>i, links[1]=>ridx] = input_tensor[idx...]
        end
    end
    # error = norm(core - factor[1]) / norm(core)
    # if error > tol / sqrt(tensor_order - 1)
    #     not_converged = true
    # end
    # factor[1] = deepcopy(core)
    comb = combiner(links[1], sites[1])
    Umat = permutedims(Vector(comb * core))
    J = maxvol(Matrix(transpose(Umat)))
    P[1] = Umat[:, J]

    # println("Left sweep 1 step $tensor_order error: $error")
    println("Link dims: $(linkdims(factor))")
    flush(stdout)

    return not_converged
end

# function maxvol(A::Matrix{Float64})
#     (n, r) = size(A)

#     row_idx = zeros(Int64, r)

#     rest_of_rows = collect(1:n)

#     i = 1
#     A_new = A
#     while i <= r
#         mask = collect(1:size(A_new)[1])
#         rows_norms = [sum((A_new .^ 2)[i, :]) for i in 1:size(A_new)[1]]

#         if size(rows_norms) == ()
#             row_idx[i] = rest_of_rows
#             break
#         end

#         if any(rows_norms .== 0)
#             zero_idx = argmin(rows_norms)
#             splice!(mask, zero_idx)
#             rest_of_rows = rest_of_rows[mask]
#             A_new = A_new[mask, :]
#             continue
#         end

#         max_row_idx = argmax(rows_norms)
#         max_row = A[rest_of_rows[max_row_idx], :]

#         projection = A_new * max_row
#         normalization = sqrt.(rows_norms[max_row_idx] * rows_norms)
#         projection = projection ./ normalization

#         A_new -= A_new .* projection

#         splice!(mask, max_row_idx)
#         A_new = A_new[mask, :]

#         row_idx[i] = rest_of_rows[max_row_idx]
#         rest_of_rows = rest_of_rows[mask]
#         i += 1
#     end
    
#     return row_idx
# end

function maxvol(A::Matrix{Float64})
    _, _, p = qr(A', ColumnNorm())
    row_idx = p[1:size(A, 2)]    
    return row_idx
end

mutable struct ODEArray{T, N} <: AbstractArray{T, N}
    f
    grid::NTuple{N, LinRange{T, Int64}}

    function ODEArray(f, grid::NTuple{N, LinRange{T, Int64}}) where {T, N}
        new{T, N}(f, grid)
    end
end

function Base.size(A::ODEArray)
    return Tuple([length(elt) - 1 for elt in A.grid])
end

function Base.ndims(A::ODEArray)
    return length(A.grid)
end

function Base.getindex(A::ODEArray, elements::Int64...)
    return A.f([A.grid[i][elements[i]] for i in eachindex(elements)]...)
end

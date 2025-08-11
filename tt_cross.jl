using ITensors
using ITensorMPS
using LinearAlgebra
using HDF5

function dmrg_cross(input_tensor, maxrank::Int64, cutoff::Float64, tol::Float64, n_iter_max::Int64, seedlist::Vector{Vector{Int64}}=Vector{Int64}[])
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

    P = Vector{Matrix}(undef, tensor_order - 1)
    row_idx = Vector(undef, tensor_order - 1)
    col_idx = Vector(undef, tensor_order - 1)
    Q, R, links[tensor_order - 1] = qr(factor[tensor_order], sites[tensor_order]; tags=tags(links[tensor_order - 1]))
    factor[tensor_order] = Q
    Qmat = Matrix(Q, links[tensor_order - 1], sites[tensor_order])
    J = maxvol(Matrix(transpose(Qmat)))
    col_idx[tensor_order - 1] = J
    P[tensor_order - 1] = Qmat[:, J]

    for k in tensor_order-1:-1:2
        factor[k] *= R
        Q, R, links[k - 1] = qr(factor[k], sites[k], links[k]; tags=tags(links[k - 1]))
        Q *= ITensor(P[k], links[k], links[k]')
        noprime!(Q)
        factor[k] = Q
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
        P[k - 1] = Qmat[:, J]
    end

    factor[1] *= R * ITensor(P[1], links[1], links[1]')
    noprime!(factor[1])

    iter = 0
    not_converged = true

    for iter in 1:n_iter_max
        if !not_converged
            break
        end
        not_converged = false
        
        if left_right_dmrgcross(input_tensor, rank, row_idx, col_idx, factor, P, maxrank, cutoff, tol, iter)
            not_converged = true
        end
        if right_left_dmrgcross(input_tensor, rank, row_idx, col_idx, factor, P, maxrank, cutoff, tol, iter)
            not_converged = true
        end
        f = h5open("dmrg_cross_$iter.h5", "w")
        write(f, "factor", factor)
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

function left_right_dmrgcross(input_tensor, rank::Vector{Int64}, row_idx, col_idx, factor::MPS, P::Vector{Matrix}, maxrank::Int64, cutoff::Float64, tol::Float64, iter::Int64)
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
    bond *= ITensor(inv(P[2]), links[2], links[2]')
    noprime!(bond)
    error = norm(bond - factor[1] * factor[2]) / norm(bond)
    if error > tol / sqrt(tensor_order - 1)
        not_converged = true
    end

    U, S, V = maxrank == -1 ? svd(bond, sites[1]; cutoff=cutoff, lefttags=tags(links[1])) : svd(bond, sites[1]; cutoff=cutoff, maxdim=maxrank, lefttags=tags(links[1]))
    links[1] = commonind(U, S)
    rank[1] = ITensors.dim(links[1])
    println([S[i, i] for i in 1:rank[1]])
    factor[1] = U
    factor[2] = S * V
    Umat = Matrix(U, sites[1], links[1])
    I = maxvol(Umat)
    row_idx[1] = I
    P[1] = Umat[I, :]

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
        bond *= ITensor(inv(P[k - 1]), links[k - 1]', links[k - 1]) * ITensor(inv(P[k + 1]), links[k + 1], links[k + 1]')
        noprime!(bond)
        error = norm(bond - factor[k] * factor[k + 1]) / norm(bond)
        if error > tol / sqrt(tensor_order - 1)
            not_converged = true
        end

        U, S, V = maxrank == -1 ? svd(bond, links[k - 1], sites[k]; cutoff=cutoff, lefttags=tags(links[k])) : svd(bond, links[k - 1], sites[k]; cutoff=cutoff, maxdim=maxrank, lefttags=tags(links[k]))
        links[k] = commonind(U, S)
        rank[k] = ITensors.dim(links[k])
        println([S[i, i] for i in 1:rank[k]])
        factor[k] = U
        factor[k + 1] = S * V
        U *= ITensor(P[k - 1], links[k - 1]', links[k - 1])
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
        P[k] = Umat[I, :]

        println("Right sweep $iter step $k error: $error")
        flush(stdout)
    end

    println("Link dims: $(linkdims(factor))")
    flush(stdout)

    return not_converged
end

function right_left_dmrgcross(input_tensor, rank::Vector{Int64}, row_idx, col_idx, factor::MPS, P::Vector{Matrix}, maxrank::Int64, cutoff::Float64, tol::Float64, iter::Int64)
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
    bond *= ITensor(inv(P[tensor_order - 2]), links[tensor_order - 2]', links[tensor_order - 2])
    noprime!(bond)
    error = norm(bond - factor[tensor_order - 1] * factor[tensor_order]) / norm(bond)
    if error > tol / sqrt(tensor_order - 1)
        not_converged = true
    end

    U, S, V = maxrank == -1 ? svd(bond, sites[tensor_order]; cutoff=cutoff, lefttags=tags(links[tensor_order - 1])) : svd(bond, sites[tensor_order]; cutoff=cutoff, maxdim=maxrank, lefttags=tags(links[tensor_order - 1]))
    links[tensor_order - 1] = commonind(U, S)
    rank[tensor_order - 1] = ITensors.dim(links[tensor_order - 1])
    println([S[i, i] for i in 1:rank[tensor_order - 1]])
    factor[tensor_order - 1] = S * V
    factor[tensor_order] = U
    Umat = Matrix(U, links[tensor_order - 1], sites[tensor_order])
    J = maxvol(Matrix(transpose(Umat)))
    col_idx[tensor_order - 1] = J
    P[tensor_order - 1] = Umat[:, J]

    println("Left sweep $iter step $(tensor_order - 1) error: $error")
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
        bond *= ITensor(inv(P[k - 1]), links[k - 1]', links[k - 1]) * ITensor(inv(P[k + 1]), links[k + 1], links[k + 1]')
        noprime!(bond)
        error = norm(bond - factor[k] * factor[k + 1]) / norm(bond)
        if error > tol / sqrt(tensor_order - 1)
            not_converged = true
        end

        U, S, V = maxrank == -1 ? svd(bond, sites[k + 1], links[k + 1]; cutoff=cutoff, lefttags=tags(links[k])) : svd(bond, sites[k + 1], links[k + 1]; cutoff=cutoff, maxdim=maxrank, lefttags=tags(links[k]))
        links[k] = commonind(U, S)
        rank[k] = ITensors.dim(links[k])
        println([S[i, i] for i in 1:rank[k]])
        factor[k] = S * V
        factor[k + 1] = U
        U *= ITensor(P[k + 1], links[k + 1], links[k + 1]')
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
        P[k] = Umat[:, J]

        println("Left sweep $iter step $k error: $error")
        flush(stdout)
    end

    println("Link dims: $(linkdims(factor))")
    flush(stdout)

    return not_converged
end

function tt_cross(input_tensor, maxrank::Int64, tol::Float64, n_iter_max::Int64, seedlist::Vector{Vector{Int64}}=Vector{Int64}[])
    tensor_shape = size(input_tensor)
    tensor_order = ndims(input_tensor)

    rank = zeros(Int64, tensor_order - 1)
    for i in 1:tensor_order-1
        left = tensor_shape[1];
        for j in 2:i
            left *= tensor_shape[j]
        end
        right = tensor_shape[tensor_order];
        for j in tensor_order-1:-1:i+1
            right *= tensor_shape[j]
        end
        rank[i] = min(left, right, maxrank)
    end

    sites = [siteind(tensor_shape[i], i) for i in 1:tensor_order]
    factor_old = randomMPS(sites)
    factor_new = randomMPS(sites; linkdims=rank)

    row_idx = Vector(undef, tensor_order)
    col_idx = Vector(undef, tensor_order)
    row_idx[1] = [[]]
    col_idx[tensor_order] = [[]]

    if isempty(seedlist)
        push!(seedlist, [rand(1:tensor_shape[i]) for i in 1:tensor_order])
    end

    for k_col_idx in tensor_order-1:-1:1
        col_idx[k_col_idx] = []
        sortedlist = Vector(undef, length(seedlist))
        for i in eachindex(seedlist)
            sortedlist[i] = []
            if k_col_idx == tensor_order - 1
                for j in 1:tensor_shape[tensor_order]
                    push!(sortedlist[i], (abs(j - seedlist[i][tensor_order]), [j]))
                end
            else
                for j in 1:tensor_shape[k_col_idx + 1]
                    for k in 1:rank[k_col_idx + 1]
                        pivot = [j, col_idx[k_col_idx + 1][k]...]
                        push!(sortedlist[i], (norm(pivot - seedlist[i][k_col_idx+1:tensor_order]), pivot))
                    end
                end
            end
            sort!(sortedlist[i])
        end
        countlist = fill(1, length(seedlist))
        idx = 1
        while length(col_idx[k_col_idx]) < rank[k_col_idx]
            seed_idx = mod(idx - 1, length(seedlist)) + 1
            pivot = sortedlist[seed_idx][countlist[seed_idx]][2]
            if !(pivot in col_idx[k_col_idx])
                push!(col_idx[k_col_idx], pivot)
            end
            countlist[seed_idx] += 1
            idx += 1
        end
    end

    iter = 0

    error = ITensorMPS.dist(factor_new, factor_old) / norm(factor_new)
    for iter in 1:n_iter_max
        if error < tol
            break
        end

        factor_old = deepcopy(factor_new)

        for k in 1:tensor_order-1
            next_row_idx = left_right_ttcross_step(
                input_tensor, k, rank, row_idx, col_idx, siteind(factor_new, k), linkind(factor_new, k), k == 1 ? Index(1) : linkind(factor_new, k - 1)
            )
            row_idx[k + 1] = next_row_idx
            println("Step $k (right) done!")
            flush(stdout)
        end

        for k in tensor_order:-1:2
            next_col_idx, Q_skeleton = right_left_ttcross_step(
                input_tensor, k, rank, row_idx, col_idx, siteind(factor_new, k), k == tensor_order ? Index(1) : linkind(factor_new, k), linkind(factor_new, k - 1)
            )
            col_idx[k - 1] = next_col_idx
            factor_new[k] = Q_skeleton
            println("Step $k (left) done!")
            flush(stdout)
        end

        s = siteind(factor_new, 1)
        l = linkind(factor_new, 1)
        for i in 1:tensor_shape[1]
            for ridx in 1:rank[1]
                idx = [[i]; col_idx[1][ridx]]
                factor_new[1][s=>i, l=>ridx] = input_tensor[idx...]
            end
        end

        orthogonalize!(factor_new, 1)
        error = ITensorMPS.dist(factor_new, factor_old) / norm(factor_new)
        println("Sweep $iter error: $error")
        flush(stdout)

        f = h5open("tt_cross_$iter.h5", "w")
        write(f, "factor", factor_new)
        close(f)

        open("tt_cross_$iter.txt", "w") do file
            write(file, "$row_idx\n")
            write(file, "$col_idx\n")
        end
    end

    if iter >= n_iter_max
        println("Maximum number of iterations reached.")
    end
    if error > tol
        println("Low Rank Approximation algorithm did not converge.")
    end

    return factor_new
end

function left_right_ttcross_step(input_tensor, k::Int64, rank::Vector{Int64}, row_idx, col_idx, s::Index, l::Index, lp::Index)
    tensor_shape = size(input_tensor)

    core = k == 1 ? ITensor(s, l) : ITensor(s, lp, l)

    for i in 1:tensor_shape[k]
        for ridx in 1:rank[k]
            if k == 1
                idx = [[i]; col_idx[1][ridx]]
                core[s=>i, l=>ridx] = input_tensor[idx...]
            else
                for lidx in 1:rank[k - 1]
                    idx = [row_idx[k][lidx]; [i]; col_idx[k][ridx]]
                    core[s=>i, lp=>lidx, l=>ridx] = input_tensor[idx...]
                end
            end
        end
    end

    Q, _, q = k == 1 ? qr(core, s) : qr(core, (s, lp))
    C = k == 1 ? combiner(s) : combiner(s, lp)
    Qmat = Matrix(C * Q, combinedind(C), q)
    I = maxvol(Qmat)

    new_idx = [
        [floor(Int64, (idx - 1) / tensor_shape[k]) + 1, mod(idx - 1, tensor_shape[k]) + 1] for idx in I
    ]
    next_row_idx = [
        [row_idx[k][ic[1]]; [ic[2]]] for ic in new_idx
    ]

    return next_row_idx
end

function right_left_ttcross_step(input_tensor, k::Int64, rank::Vector{Int64}, row_idx, col_idx, s::Index, l::Index, lp::Index)
    tensor_shape = size(input_tensor)
    tensor_order = ndims(input_tensor)

    core = k == tensor_order ? ITensor(s, lp) : ITensor(s, lp, l)

    for i in 1:tensor_shape[k]
        for lidx in 1:rank[k - 1]
            if k == tensor_order
                idx = [row_idx[tensor_order][lidx]; [i]]
                core[s=>i, lp=>lidx] = input_tensor[idx...]
            else
                for ridx in 1:rank[k]
                    idx = [row_idx[k][lidx]; [i]; col_idx[k][ridx]]
                    core[s=>i, lp=>lidx, l=>ridx] = input_tensor[idx...]
                end
            end
        end
    end

    Q, _, q = k == tensor_order ? qr(core, s) : qr(core, (l, s))
    C = k == tensor_order ? combiner(s) : combiner(l, s)
    Qmat = Matrix(C * Q, q, combinedind(C))
    J = maxvol(Matrix(transpose(Qmat)))
    Q_inv = inv(Qmat[:, J])
    Q_skeleton = Q * delta(q, lp) * ITensor(Q_inv, lp', lp)
    noprime!(Q_skeleton)

    r = k == tensor_order ? 1 : rank[k]
    new_idx = [
        [floor(Int64, (idx - 1) / r) + 1, mod(idx - 1, r) + 1] for idx in J
    ]
    next_col_idx = [
        [[jc[1]]; col_idx[k][jc[2]]] for jc in new_idx
    ]

    return next_col_idx, Q_skeleton
end

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

function to_continuous(A::ODEArray, row_idx, col_idx)
    order = ndims(A)

    I = Vector{Vector{Vector{Float64}}}(undef, order)
    J = Vector{Vector{Vector{Float64}}}(undef, order)
    I[1] = []
    for i in 2:order
        I[i] = [[A.grid[j][row[j]] for j in 1:i-1] for row in row_idx[i]]
        J[i - 1] = [[A.grid[j][col[j - i + 1]] for j in i:order] for col in col_idx[i - 1]]
    end
    J[order] = []
    return I, J
end

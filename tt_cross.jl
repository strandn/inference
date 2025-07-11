using ITensors
using ITensorMPS
using Random
using Distributions

function tensor_train_cross(input_tensor, cutoff::Float64, maxrank::Int64, tol::Float64, n_iter_max::Int64, seedlist::Vector{Vector{Int64}}=Vector{Int64}[])
    tensor_shape = size(input_tensor)
    tensor_order = ndims(input_tensor)

    col_idx = fill([], tensor_order)
    rank = fill(1, tensor_order)
    if isempty(seedlist)
        for k_col_idx in 1:tensor_order-2
            newidx = [
                rand(1:tensor_shape[j])
                for j in k_col_idx+2:tensor_order
            ]
            push!(col_idx[k_col_idx], newidx)
        end
    else
        for k_col_idx in 1:tensor_order-2
            for seed in seedlist
                newidx = seed[k_col_idx+2:tensor_order]
                push!(col_idx[k_col_idx], newidx)
            end
            rank[k_col_idx] = length(seedlist)
        end
        rank[tensor_order - 1] = length(seedlist)
    end

    sites = [siteind(tensor_shape[i], i) for i in 1:tensor_order]
    factor_old = randomMPS(sites)
    factor_new = randomMPS(sites)

    iter = 0

    error = norm(factor_old - factor_new)
    threshold = tol * norm(factor_new)
    for iter in 1:n_iter_max
        if error < threshold
            break
        end

        factor_old = deepcopy(factor_new)

        row_idx = [[[]]]
        for k in 1:tensor_order-2
            sl = siteind(factor_new, k)
            sr = siteind(factor_new, k + 1)
            l = linkind(factor_new, k)
            lr = linkind(factor_new, k + 1)
            ll = k == 1 ? Index(1) : linkind(factor_new, k - 1)
            next_row_idx = left_right_ttcross_step(
                input_tensor, k, rank, row_idx, col_idx, sl, sr, l, lr, ll, cutoff, maxrank
            )
            push!(row_idx, next_row_idx)
        end

        col_idx = fill([[]], tensor_order)
        lr = Index(1)
        for k in tensor_order:-1:3
            sl = siteind(factor_new, k - 1)
            sr = siteind(factor_new, k)
            l = linkind(factor_new, k - 1)
            ll = linkind(factor_new, k - 2)
            next_col_idx, Q_skeleton, lr = right_left_ttcross_step(
                input_tensor, k, rank, row_idx, col_idx, sl, sr, l, lr, ll, cutoff, maxrank
            )
            col_idx[k - 2] = next_col_idx

            factor_new[k] = Q_skeleton
        end

        sl = siteind(factor_new, 1)
        sr = siteind(factor_new, 2)
        lefttags = tags(linkind(factor_new, 1))
        bond = ITensor(sl, sr, lr)
        for i in 1:tensor_shape[1]
            for j in 1:tensor_shape[2]
                for ridx in 1:rank[2]
                    idx = [[i, j]; col_idx[1][ridx]]
                    bond[sl=>i, sr=>j, lr=>ridx] = input_tensor[idx...]
                end
            end
        end
        factor_new[1], S, V = svd(bond, sl; cutoff=cutoff, maxdim=maxrank, lefttags=lefttags)
        factor_new[2] = S * V

        error = norm(factor_old - factor_new)
        println("Sweep $iter error: $error")
        println("Link dims: $(linkdims(factor_new))")
        flush(stdout)
        threshold = tol * norm(factor_new)
    end

    if iter >= n_iter_max
        println("Maximum number of iterations reached.")
        flush(stdout)
    end
    if norm(factor_old - factor_new) > tol * norm(factor_new)
        println("Low Rank Approximation algorithm did not converge.")
        flush(stdout)
    end

    return factor_new
end

function left_right_ttcross_step(input_tensor, k::Int64, rank::Vector{Int64}, row_idx, col_idx, sl::Index, sr::Index, l::Index, lr::Index, ll::Index, cutoff::Float64, maxrank::Int64)
    tensor_shape = size(input_tensor)

    bond = k == 1 ? ITensor(sl, sr, lr) : ITensor(sl, sr, ll, lr)

    for i in 1:tensor_shape[k]
        for j in 1:tensor_shape[k + 1]
            for ridx in 1:rank[k + 1]
                if k == 1
                    idx = [[i, j]; col_idx[1][ridx]]
                    bond[sl=>i, sr=>j, lr=>ridx] = input_tensor[idx...]
                else
                    for lidx in 1:rank[k - 1]
                        idx = [row_idx[k][lidx]; [i, j]; col_idx[k][ridx]]
                        bond[sl=>i, sr=>j, ll=>lidx, lr=>ridx] = input_tensor[idx...]
                    end
                end
            end
        end
    end

    U, S, V = k == 1 ? svd(bond, sl; cutoff=cutoff, maxdim=maxrank, righttags=tags(l)) : svd(bond, sl, ll; cutoff=cutoff, maxdim=maxrank, righttags=tags(l))
    q = commonind(S, V)
    rank[k] = dim(q)
    C = k == 1 ? combiner(sl) : combiner(sl, ll)
    I, _ = maxvol(Matrix(C * U * S, combinedind(C), q))

    new_idx = [
        [floor(Int64, (idx - 1) / tensor_shape[k]) + 1, mod(idx - 1, tensor_shape[k]) + 1] for idx in I
    ]
    next_row_idx = [
        [row_idx[k][ic[1]]; [ic[2]]] for ic in new_idx
    ]

    return next_row_idx
end

function right_left_ttcross_step(input_tensor, k::Int64, rank::Vector{Int64}, row_idx, col_idx, sl::Index, sr::Index, l::Index, lr::Index, ll::Index, cutoff::Float64, maxrank::Int64)
    tensor_shape = size(input_tensor)
    tensor_order = ndims(input_tensor)

    bond = k == tensor_order ? ITensor(sl, sr, ll) : ITensor(sl, sr, ll, lr)

    for i in 1:tensor_shape[k - 1]
        for j in 1:tensor_shape[k]
            for lidx in 1:rank[k - 2]
                if k == tensor_order
                    idx = [row_idx[tensor_order][lidx]; [i, j]]
                    bond[sl=>i, sr=>j, ll=>lidx] = input_tensor[idx...]
                else
                    for ridx in 1:rank[k]
                        idx = [row_idx[k][lidx]; [i, j]; col_idx[k][ridx]]
                        bond[sl=>i, sr=>j, ll=>lidx, lr=>ridx] = input_tensor[idx...]
                    end
                end
            end
        end
    end

    U, S, V = k == tensor_order ? svd(bond, sr; cutoff=cutoff, maxdim=maxrank, righttags=tags(l)) : svd(bond, lr, sr; cutoff=cutoff, maxdim=maxrank, righttags=tags(l))
    q = commonind(S, V)
    rank[k - 1] = dim(q)
    C = k == tensor_order ? combiner(sr) : combiner(lr, sr)
    J, Q_inv_mat = maxvol(Matrix(C * U * S, combinedind(C), q))
    Q_inv = ITensor(Q_inv_mat, q, q')
    Q_skeleton = U * S * Q_inv
    noprime!(Q_skeleton)

    r = k == tensor_order ? 1 : rank[k]
    new_idx = [
        [floor(Int64, (idx - 1) / r) + 1, mod(idx - 1, r) + 1] for idx in J
    ]
    next_col_idx = [
        [[jc[1]]; col_idx[k][jc[2]]] for jc in new_idx
    ]

    return next_col_idx, Q_skeleton, q
end

function maxvol(A::Matrix{Float64})
    (n, r) = size(A)

    row_idx = zeros(Int64, r)

    rest_of_rows = collect(1:n)

    i = 1
    A_new = A
    while i <= r
        mask = collect(1:size(A_new)[1])
        rows_norms = [sum((A_new .^ 2)[i, :]) for i in 1:size(A_new)[1]]

        if size(rows_norms) == ()
            row_idx[i] = rest_of_rows
            break
        end

        if any(rows_norms .== 0)
            zero_idx = argmin(rows_norms)
            splice!(mask, zero_idx)
            rest_of_rows = rest_of_rows[mask]
            A_new = A_new[mask, :]
            continue
        end

        max_row_idx = argmax(rows_norms)
        max_row = A[rest_of_rows[max_row_idx], :]

        projection = A_new * max_row
        normalization = sqrt.(rows_norms[max_row_idx] * rows_norms)
        projection = projection ./ normalization

        A_new -= A_new .* projection

        splice!(mask, max_row_idx)
        A_new = A_new[mask, :]

        row_idx[i] = rest_of_rows[max_row_idx]
        rest_of_rows = rest_of_rows[mask]
        i += 1
    end
    
    return row_idx, inv(A[row_idx, :])
end

mutable struct ODEArray{T, N} <: AbstractArray{T, N}
    f
    grid::NTuple{N, LinRange{T, Int64}}

    function ODEArray(f, grid::NTuple{N, LinRange{T, Int64}}) where {T, N}
        new{T, N}(f, grid)
    end
end

function Base.size(A::ODEArray)
    return Tuple([size(elt) for elt in A.grid])
end

function Base.ndims(A::ODEArray)
    return length(A.grid)
end

function Base.getindex(A::ODEArray, elements::CartesianIndex)
    return A.f([A.grid[i][elements[i]] for i in eachindex(elements)]...)
end

Base.getindex(A::ODEArray, elements::Int64...) = A[CartesianIndex(elements)]

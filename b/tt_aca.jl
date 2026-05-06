using Random
using Distributions
using QuadGK
using MPI
using LinearAlgebra
using ITensors
using ITensorMPS

# Struct holding metadata used for ACA calculations
mutable struct ResFunc{T, N}
    # Function to be approximated
    f
    # Number of dimensions
    ndims::Int64
    # Position of the current unfolding (ranges from 0 to ndims - 1)
    pos::Int64
    # Limits for each dimension
    domain::NTuple{N, Tuple{T, T}}
    # Row positions of pivots for each dimension
    I::Vector{Vector{Vector{T}}}
    # Column positions of pivots for each dimension
    J::Vector{Vector{Vector{T}}}
    # Convergence threshhold (algorithm stops if the largest residual in the current iteration falls below this number)
    cutoff::T
    offset::T
    periodicity::NTuple{N, Bool}

    # Constructor
    function ResFunc(f, domain::NTuple{N, Tuple{T, T}}, cutoff::T, periodicity::NTuple{N, Bool}) where {T, N}
        new{T, N}(f, N, 0, domain, [[[T[]]]; [Vector{T}[] for _ in 2:N]], [[[T[]]]; [Vector{T}[] for _ in 2:N]], cutoff, 0.0, periodicity)
    end
end

function expnegf(F::ResFunc{T, N}, elements::T...) where {T, N}
    return exp(F.offset - F.f(elements...))
end

# Evaluates the residual in the current iteration (determined by the number of pivots found so far)
# Uses dynamic programming/memoization to accelerate recursive evaluation
function (F::ResFunc{T, N})(elements::T...) where {T, N}
    (x, y) = ([elements[i] for i in 1:F.pos], [elements[i] for i in F.pos+1:F.ndims])
    k = length(F.I[F.pos + 1])
    old = undef
    new = undef
    old_sign = undef
    new_sign = undef

    for iter in 0:k
        new = zeros(k - iter + 1, k - iter + 1)
        new_sign = ones(Int64, k - iter + 1, k - iter + 1)
        for idx in CartesianIndices(new)
            if iter == 0
                row = idx[1] == k + 1 ? x : F.I[F.pos + 1][idx[1]]
                col = idx[2] == k + 1 ? y : F.J[F.pos + 1][idx[2]]
                new[idx] = F.f(row..., col...) - F.offset
            else
                arg1 = -old[idx[1] + 1, idx[2] + 1]
                arg2 = old[1, 1] - old[idx[1] + 1, 1] - old[1, idx[2] + 1]
                arglarge = max(arg1, arg2)

                sign1 = old_sign[idx[1] + 1, idx[2] + 1]
                sign2 = old_sign[1, 1] * old_sign[idx[1] + 1, 1] * old_sign[1, idx[2] + 1]

                delta = sign1 * exp(arg1 - arglarge) - sign2 * exp(arg2 - arglarge)
                new[idx] = -arglarge - log(abs(delta) + eps())  # add eps to guard against log(0)
                new_sign[idx] = ifelse(delta < 0.0, -1, 1)
            end
        end
        old = deepcopy(new)
        old_sign = deepcopy(new_sign)
    end
    return new[]
    # return new[], new_sign[]
end

# Updates I and J by inserting a new pivot at the current unfolding (pos)
function updateIJ(F::ResFunc{T, N}, ij::NTuple{N, T}) where {T, N}
    push!(F.I[F.pos + 1], [ij[j] for j in 1:F.pos])
    push!(F.J[F.pos + 1], [ij[j] for j in F.pos+1:F.ndims])
end

# Main ACA routine
# Takes multidimensional function F, and sequentially finds pivots which roughly maximize a dynamical residual function, moving from one unfolding to the next
# Pivots in each unfolding depend on pivots found for the previous unfolding
function continuous_aca(F::ResFunc{T, N}, rank::Vector{Int64}, n_chains::Int64, n_samples::Int64, n_samples_init::Int64, jump_width::Float64, mpi_comm::MPI.Comm) where {T, N}
    mpi_rank = MPI.Comm_rank(mpi_comm)
    mpi_size = MPI.Comm_size(mpi_comm)
    
    order = F.ndims
    if order == 1 && mpi_rank == 0
        error(
            "`continuous_aca` currently does not support system sizes of 1.",
        )
    end

    F.pos = 0
    for i in 1:order-1
        if mpi_rank == 0
            println("pos = $i")
            flush(stdout)
        end
        F.pos += 1
        
        # MCMC sampling routine to be parallelized across MPI processes
        n_chains_total = max(n_chains, mpi_size, length(F.I[i]))
        xylist = fill(Tuple(fill(0.0, order)), n_chains_total)
        reslist = fill(0.0, n_chains_total)
        res_new = 0.0
        r = length(F.I[i + 1]) + 1
        pivot_count = pivot_last = mod(r - 1, length(F.I[i])) + 1
        seedlist = fill(NaN, n_chains_total, F.ndims - i + 1)
        if i > 1
            for k in 1:n_chains_total
                idx = mod(k - 1, length(F.I[i])) + 1
                seedlist[k, :] = F.J[i][idx]
            end
        end
        while true
            if r > rank[i]
                break
            end
            # Determine number of tasks for the current process
            elements_per_task = div(n_chains_total, mpi_size)
            # Extra tasks to be carried out by the root process at the end
            remainder = rem(n_chains_total, mpi_size)
            local_xy = fill(Tuple(fill(0.0, order)), elements_per_task)
            local_res = fill(0.0, elements_per_task)
            n_samples_adj = (i == 1 && r == 1 ? n_samples_init : n_samples)
            for k in 1:elements_per_task
                # Run multiple Markov chains in parallel, approximate position of the largest current residual across all walkers
                idx = mod(mpi_rank * elements_per_task + k - 1, length(F.I[i])) + 1
                # idx = mod(pivot_count - 1, length(F.I[i])) + 1
                local_xy[k], local_res[k], seedlist[mpi_rank * elements_per_task + k, :] = max_metropolis(F, F.I[i][idx], n_samples_adj, jump_width, seedlist[mpi_rank * elements_per_task + k, :])
                # println(seedlist[mpi_rank * elements_per_task + k, :])
            end
            # if i == 1 && r == 1
            #     display(hcat(seedlist, [F.f(seedlist[idx, :]...) for idx in 1:n_chains_total]))
            # end
            # Collect results from all processes
            xydata = MPI.Gather(local_xy, 0, mpi_comm)
            resdata = MPI.Gather(local_res, 0, mpi_comm)
            if mpi_rank == 0
                xylist[1:mpi_size*elements_per_task] .= xydata
                reslist[1:mpi_size*elements_per_task] .= resdata
            end
            # Rank 0 process performs any remaining tasks
            if mpi_rank == 0 && remainder > 0
                for k in mpi_size*elements_per_task+1:n_chains_total
                    idx = mod(k - 1, length(F.I[i])) + 1
                    # idx = mod(pivot_count - 1, length(F.I[i])) + 1
                    xylist[k], reslist[k], seedlist[k, :] = max_metropolis(F, F.I[i][idx], n_samples_adj, jump_width, seedlist[k, :])
                end
            end
            
            # Find position of largest residuals
            idx = argmin(reslist)
            xy = [xylist[idx]]
            MPI.Bcast!(xy, 0, mpi_comm)
            if isempty(F.I[i + 1]) || F.f(xy[]...) < F.offset
                offset_new = [F.f(xy[]...)]
                MPI.Bcast!(offset_new, 0, mpi_comm)
                F.offset = offset_new[]
            end
            res_new = [exp(-F(xy[]...))]
            MPI.Bcast!(res_new, 0, mpi_comm)
            pivot_count = mod(pivot_count, length(F.I[i])) + 1
            if res_new[] < F.cutoff
                break
                # if pivot_count == pivot_last
                #     break
                # else
                #     continue
                # end
            end
            updateIJ(F, xy[])
            if mpi_rank == 0
                println("rank = $r res = $(res_new[]) xy = $(xy[]) offset = $(F.offset)")
                flush(stdout)
            end
            r += 1
            pivot_last = pivot_count
        end
    end

    return F.I, F.J
end

# MCMC sampler, searches function F and runs one Markov chain
# Outputs largest residual magnitude and corresponding position
function max_metropolis(F::ResFunc{T, N}, pivot::Vector{T}, n_samples::Int64, jump_width::Float64, seed::Vector{T}) where {T, N}
    order = F.ndims - F.pos + 1
    
    lb = [F.domain[i][1] for i in F.pos:F.ndims]
    ub = [F.domain[i][2] for i in F.pos:F.ndims]

    chain = zeros(n_samples, order)

    min_nlr = Inf
    max_xy = zeros(F.ndims)

    if isnan(seed[1])
        for k in 1:order
            chain[1, k] = rand() * (ub[k] - lb[k]) + lb[k]
        end
        # fx = F(pivot..., chain[1, :]...)  # Call once
    else
        chain[1, :] = seed
    end

    for i in 2:n_samples
        p_new = zeros(order)
        for k in 1:order
            p_new[k] = rand(Normal(chain[i - 1, k], jump_width * (ub[k] - lb[k])))
            if(F.periodicity[k])
                p_new[k] = mod(p_new[k] - lb[k], ub[k] - lb[k]) + lb[k]
            else
                if p_new[k] < lb[k]
                    p_new[k] = lb[k] + abs(p_new[k] - lb[k])
                elseif p_new[k] > ub[k]
                    p_new[k] = ub[k] - abs(p_new[k] - ub[k])
                end
            end
        end

        arg_old = [pivot; [chain[i - 1, k] for k in 1:order]]
        arg_new = [pivot; [p_new[k] for k in 1:order]]
        acceptance_prob = 0.0
        if !isfinite(F.f(arg_old...))
            f_new = Inf
            acceptance_prob = 1.0
        elseif isfinite(F.f(arg_new...))
            f_old = F(arg_old...)
            f_new = F(arg_new...)
            acceptance_prob = min(1, exp(f_old - f_new))
        end
        
        if rand() < acceptance_prob
            chain[i, :] = p_new
            if f_new < min_nlr
                min_nlr = f_new
                max_xy = arg_new
            end
        else
            chain[i, :] = chain[i - 1, :]
        end
    end

    return Tuple(max_xy), min_nlr, chain[n_samples, :]
end

# # Main ACA routine
# # Takes multidimensional function F, and sequentially finds pivots which roughly maximize a dynamical residual function, moving from one unfolding to the next
# # Pivots in each unfolding depend on pivots found for the previous unfolding
# function continuous_aca(F::ResFunc{T, N}, rank::Vector{Int64}, n_chains::Int64, n_samples::Int64, jump_width::Float64, mpi_comm::MPI.Comm) where {T, N}
#     mpi_rank = MPI.Comm_rank(mpi_comm)
#     mpi_size = MPI.Comm_size(mpi_comm)
    
#     order = F.ndims
#     if order == 1 && mpi_rank == 0
#         error(
#             "`continuous_aca` currently does not support system sizes of 1.",
#         )
#     end

#     F.pos = 0
#     for count in 1:order-1
#         if mpi_rank == 0
#             println("pos = $count")
#             flush(stdout)
#         end
#         F.pos += 1
        
#         # MCMC sampling routine to be parallelized across MPI processes
#         n_chains_total = max(n_chains, mpi_size, length(F.I[count]))
#         xylist = zeros(n_chains_total * n_samples, order)
#         reslist = zeros(n_chains_total * n_samples)
#         signlist = ones(Int64, n_chains_total * n_samples)
#         res_new = 0.0
#         r = length(F.I[count + 1]) + 1
#         # while r <= rank[count]
#         while r <= 1
#             # Determine number of tasks for the current process
#             elements_per_task = div(n_chains_total, mpi_size)
#             # Extra tasks to be carried out by the root process at the end
#             remainder = rem(n_chains_total, mpi_size)
#             local_xy = zeros(elements_per_task * n_samples, order)
#             local_res = zeros(elements_per_task * n_samples)
#             local_sign = ones(Int64, elements_per_task * n_samples)
#             for k in 1:elements_per_task
#                 # Run multiple Markov chains in parallel, approximate position of the largest current residual across all walkers
#                 idx = mod(mpi_rank * elements_per_task + k - 1, length(F.I[count])) + 1
#                 local_xy[(k-1)*n_samples+1:k*n_samples, :], local_res[(k-1)*n_samples+1:k*n_samples], local_sign[(k-1)*n_samples+1:k*n_samples] = max_metropolis(F, F.I[count][idx], n_samples, jump_width)
#             end
#             # Collect results from all processes
#             local_xy = reshape(local_xy, elements_per_task * n_samples * order)
#             xydata = MPI.Gather(local_xy, 0, mpi_comm)
#             resdata = MPI.Gather(local_res, 0, mpi_comm)
#             signdata = MPI.Gather(local_sign, 0, mpi_comm)
#             if mpi_rank == 0
#                 xydata = reshape(xydata, mpi_size * elements_per_task * n_samples, order)
#                 xylist[1:mpi_size*elements_per_task*n_samples, :] .= xydata
#                 reslist[1:mpi_size*elements_per_task*n_samples] .= resdata
#                 signlist[1:mpi_size*elements_per_task*n_samples] .= signdata
#             end
#             # Rank 0 process performs any remaining tasks
#             if mpi_rank == 0 && remainder > 0
#                 for k in mpi_size*elements_per_task+1:n_chains_total
#                     idx = mod(k - 1, length(F.I[count])) + 1
#                     xylist[(k-1)*n_samples+1:k*n_samples, :], reslist[(k-1)*n_samples+1:k*n_samples], signlist[(k-1)*n_samples+1:k*n_samples] = max_metropolis(F, F.I[count][idx], n_samples, jump_width)
#                 end
#             end

#             idx = argmin(reslist)
#             # println(reslist[idx])
#             xy = xylist[idx, :]
#             MPI.Bcast!(xy, 0, mpi_comm)
#             offset_delta = 0.0
#             if isempty(F.I[count + 1]) || F.f(xy...) < F.offset
#                 offset_new = [F.f(xy...)]
#                 MPI.Bcast!(offset_new, 0, mpi_comm)
#                 offset_delta = offset_new[] - F.offset
#                 F.offset = offset_new[]
#             end
#             reslist .-= offset_delta
#             # println(reslist[idx])

#             nlres_new, _ = F(xy...)
#             res_new = [exp(-nlres_new)]
#             MPI.Bcast!(res_new, 0, mpi_comm)
#             if res_new[] < F.cutoff
#                 break
#             end

#             new_pivots = []
#             npivots = [0]

#             if mpi_rank == 0
#                 Rk = deepcopy(reslist)
#                 Rk_sign = deepcopy(signlist)
#                 u = []
#                 v = []
#                 u_sign = []
#                 v_sign = []
#                 r_first = length(F.I[count + 1]) + 1
#                 nsamples = n_chains_total * n_samples
#                 while r <= rank[count]
#                     # println(length(Rk))
#                     # resxy = [(Rk[i], xylist[i, :]) for i in eachindex(Rk)]
#                     # sort!(resxy)
#                     # for i in 1:1000
#                     #     println(resxy[i])
#                     # end

#                     ik = 0
#                     dk = Inf
#                     dk_sign = 1
#                     for i in 1:nsamples
#                         if Rk[i] < dk
#                             ik = i
#                             dk = Rk[i]
#                             dk_sign = Rk_sign[i]
#                         end
#                     end
#                     Ri = zeros(nsamples)
#                     Rj = zeros(nsamples)
#                     Ri_sign = ones(Int64, nsamples)
#                     Rj_sign = ones(Int64, nsamples)
#                     for i in 1:nsamples
#                         arg = [xylist[i, 1:F.pos]; xylist[ik, F.pos+1:order]]
#                         # Ri[i] = F.f(arg...) - F.offset
#                         Ri[i], Ri_sign[i] = F(arg...)
#                         arg = [xylist[ik, 1:F.pos]; xylist[i, F.pos+1:order]]
#                         # Rj[i] = F.f(arg...) - F.offset
#                         Rj[i], Rj_sign[i] = F(arg...)
#                     end
#                     for l in 1:r-r_first
#                         for i in 1:nsamples
#                             arg1 = -Ri[i]
#                             arg2 = -u[l][i] - v[l][ik]
#                             arglarge = max(arg1, arg2)

#                             sign1 = Ri_sign[i]
#                             sign2 = u_sign[l][i] * v_sign[l][ik]

#                             delta = sign1 * exp(arg1 - arglarge) - sign2 * exp(arg2 - arglarge)
#                             Ri[i] = -arglarge - log(abs(delta) + eps())
#                             Ri_sign[i] = ifelse(delta < 0.0, -1, 1)

#                             arg1 = -Rj[i]
#                             arg2 = -v[l][i] - u[l][ik]
#                             arglarge = max(arg1, arg2)

#                             sign1 = Rj_sign[i]
#                             sign2 = v_sign[l][i] * u_sign[l][ik]

#                             delta = sign1 * exp(arg1 - arglarge) - sign2 * exp(arg2 - arglarge)
#                             Rj[i] = -arglarge - log(abs(delta) + eps())
#                             Rj_sign[i] = ifelse(delta < 0.0, -1, 1)
#                         end
#                     end
#                     Rj .-= dk
#                     Rj_sign .*= dk_sign
#                     push!(u, deepcopy(Ri))
#                     push!(v, deepcopy(Rj))
#                     push!(u_sign, deepcopy(Ri_sign))
#                     push!(v_sign, deepcopy(Rj_sign))

#                     xy = xylist[ik, :]
#                     # nlres_new, _ = F(xy...)
#                     # res_new = [exp(-nlres_new)]
#                     res_new = [exp(-dk)]
#                     if res_new[] < F.cutoff
#                         break
#                     end
#                     push!(new_pivots, xy)
#                     npivots[] += 1
#                     println("rank = $r res = $(res_new[]) xy = $xy offset = $(F.offset)")
#                     flush(stdout)

#                     for i in 1:nsamples
#                         arg1 = -Rk[i]
#                         arg2 = -u[r - r_first + 1][i] - v[r - r_first + 1][i]
#                         arglarge = max(arg1, arg2)

#                         sign1 = Rk_sign[i]
#                         sign2 = u_sign[r - r_first + 1][i] * v_sign[r - r_first + 1][i]

#                         delta = sign1 * exp(arg1 - arglarge) - sign2 * exp(arg2 - arglarge)
#                         Rk[i] = -arglarge - log(abs(delta) + eps())
#                         Rk_sign[i] = ifelse(delta < 0.0, -1, 1)
#                     end

#                     r += 1
#                 end
#             end

#             MPI.Bcast!(npivots, 0, mpi_comm)
#             for i in 1:npivots[]
#                 xy = zeros(order)
#                 if mpi_rank == 0
#                     xy = new_pivots[i]
#                 end
#                 MPI.Bcast!(xy, 0, mpi_comm)
#                 updateIJ(F, Tuple(xy))
#             end
#         end
#     end

#     return F.I, F.J
# end

# # MCMC sampler, searches function F and runs one Markov chain
# # Outputs largest residual magnitude and corresponding position
# function max_metropolis(F::ResFunc{T, N}, pivot::Vector{T}, n_samples::Int64, jump_width::Float64) where {T, N}
#     order = F.ndims - F.pos + 1
    
#     lb = [F.domain[i][1] for i in F.pos:F.ndims]
#     ub = [F.domain[i][2] for i in F.pos:F.ndims]

#     chain_xy = zeros(n_samples, order)
#     chain_res = zeros(n_samples)
#     chain_sign = ones(Int64, n_samples)

#     f_old = 0.0
#     f_new = 0.0
#     sign_old = 1
#     sign_new = 1

#     while true
#         for k in 1:order
#             chain_xy[1, k] = rand() * (ub[k] - lb[k]) + lb[k]
#         end
#         f_old, sign_old = F(pivot..., chain_xy[1, :]...)  # Call once
#         chain_res[1] = f_old
#         chain_sign[1] = sign_old
#         if isfinite(chain_res[1])
#             break
#         end
#     end

#     for i in 2:n_samples
#         p_new = zeros(order)
#         for k in 1:order
#             p_new[k] = rand(Normal(chain_xy[i - 1, k], jump_width * (ub[k] - lb[k])))
#             if(F.periodicity[k])
#                 p_new[k] = mod(p_new[k] - lb[k], ub[k] - lb[k]) + lb[k]
#             else
#                 if p_new[k] < lb[k]
#                     p_new[k] = lb[k] + abs(p_new[k] - lb[k])
#                 elseif p_new[k] > ub[k]
#                     p_new[k] = ub[k] - abs(p_new[k] - ub[k])
#                 end
#             end
#         end

#         arg_old = [pivot; [chain_xy[i - 1, k] for k in 1:order]]
#         arg_new = [pivot; [p_new[k] for k in 1:order]]
#         log_acceptance_prob = -Inf
#         if isfinite(F.f(arg_new...))
#             f_old, sign_old = F(arg_old...)
#             f_new, sign_new = F(arg_new...)
#             log_acceptance_prob = min(0.0, f_old - f_new)
#         end
        
#         if log(rand()) < log_acceptance_prob
#             chain_xy[i, :] = p_new
#             chain_res[i] = f_new
#             chain_sign[i] = sign_new
#         else
#             chain_xy[i, :] = chain_xy[i - 1, :]
#             chain_res[i] = f_old
#             chain_sign[i] = sign_old
#         end
#     end

#     return [repeat(pivot', size(chain_xy, 1)) chain_xy], chain_res, chain_sign
# end

function build_tt(F::ResFunc{T, N}, grid::NTuple{N, Vector{T}}) where {T, N}
    order = F.ndims
    npivots = [length(F.I[i]) for i in 2:order]
    psi = Vector{ITensor}(undef, order)
    sites = [siteind(length(grid[i]), i) for i in 1:order]
    links = Vector{Index}(undef, order - 1)
    links[1] = Index(npivots[1], "Link,l=1")
    psi[1] = ITensor(sites[1], links[1])
    for j in eachindex(grid[1])
        for k in 1:npivots[1]
            psi[1][sites[1]=>j, links[1]=>k] = expnegf(F, grid[1][j], F.J[2][k]...)
        end
    end
    # println("i = 1\n")
    # println(psi[1])
    # println()
    AIJ = zeros(npivots[1], npivots[1])
    for j in 1:npivots[1]
        for k in 1:npivots[1]
            AIJ[j, k] = expnegf(F, F.I[2][j]..., F.J[2][k]...)
        end
    end
    AIJinv = inv(AIJ)
    # println(norm(AIJ * AIJinv - I))
    # println()
    skeleton = ITensor(links[1], links[1]')
    for j in 1:npivots[1]
        for k in 1:npivots[1]
            skeleton[links[1]=>j, links[1]'=>k] = AIJinv[j, k]
        end
    end
    psi[1] *= skeleton
    noprime!(psi[1])
    # println(psi[1])
    # flush(stdout)
    for i in 2:order-1
        links[i] = Index(npivots[i], "Link,l=$i")
        psi[i] = ITensor(sites[i], links[i - 1], links[i])
        for j in eachindex(grid[i])
            for k in 1:npivots[i - 1]
                for l in 1:npivots[i]
                    psi[i][sites[i]=>j, links[i-1]=>k, links[i]=>l] = expnegf(F, F.I[i][k]..., grid[i][j], F.J[i + 1][l]...)
                end
            end
        end
        # println("\ni = $i\n")
        # println(psi[i])
        # println()
        AIJ = zeros(npivots[i], npivots[i])
        for j in 1:npivots[i]
            for k in 1:npivots[i]
                AIJ[j, k] = expnegf(F, F.I[i + 1][j]..., F.J[i + 1][k]...)
            end
        end
        AIJinv = inv(AIJ)
        # println(norm(AIJ * AIJinv - I))
        # println()
        skeleton = ITensor(links[i], links[i]')
        for j in 1:npivots[i]
            for k in 1:npivots[i]
                skeleton[links[i]=>j, links[i]'=>k] = AIJinv[j, k]
            end
        end
        psi[i] *= skeleton
        noprime!(psi[i])
        # println(psi[i])
        # flush(stdout)
    end
    psi[order] = ITensor(sites[order], links[order - 1])
    for j in eachindex(grid[order])
        for k in 1:npivots[order - 1]
            psi[order][sites[order]=>j, links[order-1]=>k] = expnegf(F, F.I[order][k]..., grid[order][j])
        end
    end
    # println("\ni = $order\n")
    # println(psi[order])
    # println()
    # flush(stdout)
    return MPS(psi)
end

# Approximates normalization constant, or partition function (if F is a Boltzmann distribution)
# Involes univariate integral calculations along all dimensions
function compute_norm(F::ResFunc{T, N}) where {T, N}
    order = F.ndims
    npivots = [length(F.I[i]) for i in 2:order]
    integrals = Vector{ITensor}(undef, order)
    skeleton = Vector{ITensor}(undef, order - 1)
    links = Vector{Index}(undef, order - 1)
    links[1] = Index(npivots[1], "Link,l=1")
    integrals[1] = ITensor(links[1])
    for j in 1:npivots[1]
        f(x) = expnegf(F, x, F.J[2][j]...)
        integrals[1][links[1]=>j] = quadgk(f, F.domain[1]...; maxevals=100)[1]
    end
    result = integrals[1]
    println("i = 1\n")
    println(integrals[1])
    println()
    AIJ = zeros(npivots[1], npivots[1])
    for j in 1:npivots[1]
        for k in 1:npivots[1]
            AIJ[j, k] = expnegf(F, F.I[2][j]..., F.J[2][k]...)
        end
    end
    AIJinv = inv(AIJ)
    println(norm(AIJ * AIJinv - I))
    println()
    skeleton[1] = ITensor(links[1], links[1]')
    for j in 1:npivots[1]
        for k in 1:npivots[1]
            skeleton[1][links[1]=>j, links[1]'=>k] = AIJinv[j, k]
        end
    end
    result *= skeleton[1]
    println(result)
    flush(stdout)
    for i in 2:order-1
        links[i] = Index(npivots[i], "Link,l=$i")
        integrals[i] = ITensor(links[i - 1]', links[i])
        for j in 1:npivots[i - 1]
            for k in 1:npivots[i]
                f(x) = expnegf(F, F.I[i][j]..., x, F.J[i + 1][k]...)
                integrals[i][links[i - 1]'=>j, links[i]=>k] = quadgk(f, F.domain[i]...; maxevals=100)[1]
            end
        end
        println("\ni = $i\n")
        println(integrals[i])
        println()
        AIJ = zeros(npivots[i], npivots[i])
        for j in 1:npivots[i]
            for k in 1:npivots[i]
                AIJ[j, k] = expnegf(F, F.I[i + 1][j]..., F.J[i + 1][k]...)
            end
        end
        AIJinv = inv(AIJ)
        println(norm(AIJ * AIJinv - I))
        println()
        skeleton[i] = ITensor(links[i], links[i]')
        for j in 1:npivots[i]
            for k in 1:npivots[i]
                skeleton[i][links[i]=>j, links[i]'=>k] = AIJinv[j, k]
            end
        end
        result *= integrals[i] * skeleton[i]
        println(result)
        flush(stdout)
    end
    integrals[order] = ITensor(links[order - 1]')
    for j in 1:npivots[order - 1]
        f(x) = expnegf(F, F.I[order][j]..., x)
        integrals[order][links[order - 1]'=>j] = quadgk(f, F.domain[order]...; maxevals=100)[1]
    end
    println("\ni = $order\n")
    println(integrals[order])
    println()
    flush(stdout)
    result *= integrals[order]
    return result[], integrals, skeleton, links
end

function compute_mu(F::ResFunc{T, N}, integrals::Vector{ITensor}, skeleton::Vector{ITensor}, links::Vector{Index}, pos::Int64) where {T, N}
    order = F.ndims
    npivots = [length(F.I[i]) for i in 2:order]
    
    Lenv = undef
    Renv = undef
    if pos != 1
        Lenv = integrals[1] * skeleton[1]
        for i in 2:pos-1
            Lenv *= integrals[i] * skeleton[i]
        end
    end
    if pos != order
        Renv = skeleton[order - 1] * integrals[order]
        for i in order-1:-1:pos+1
            Renv *= skeleton[i - 1] * integrals[i]
        end
    end
    
    result = undef
    if pos == 1
        result = ITensor(links[1])
        for j in 1:npivots[1]
            f(x) = x * expnegf(F, x, F.J[2][j]...)
            result[links[1]=>j] = quadgk(f, F.domain[1]...; maxevals=100)[1]
        end
        result *= Renv
    elseif pos == order
        result = ITensor(links[order - 1]')
        for j in 1:npivots[order - 1]
            f(x) = x * expnegf(F, F.I[order][j]..., x)
            result[links[order - 1]'=>j] = quadgk(f, F.domain[order]...; maxevals=100)[1]
        end
        result *= Lenv
    else
        result = ITensor(links[pos - 1]', links[pos])
        for j in 1:npivots[pos - 1]
            for k in 1:npivots[pos]
                f(x) = x * expnegf(F, F.I[pos][j]..., x, F.J[pos + 1][k]...)
                result[links[pos - 1]'=>j, links[pos]=>k] = quadgk(f, F.domain[pos]...; maxevals=100)[1]
            end
        end
        result *= Lenv * Renv
    end
    return result[]
end

function compute_cov(F::ResFunc{T, N}, integrals::Vector{ITensor}, skeleton::Vector{ITensor}, links::Vector{Index}, mu::Vector{Float64}, pos1::Int64, pos2::Int64) where {T, N}
    order = F.ndims
    npivots = [length(F.I[i]) for i in 2:order]
    
    result1 = undef
    if pos1 == 1
        result1 = ITensor(links[1])
        for j in 1:npivots[1]
            f(x) = (pos1 == pos2 ? (x - mu[1])^2 : (x - mu[1])) * expnegf(F, x, F.J[2][j]...)
            result1[links[1]=>j] = quadgk(f, F.domain[1]...; maxevals=100)[1]
        end
    elseif pos1 == order
        result1 = ITensor(links[order - 1]')
        for j in 1:npivots[order - 1]
            f(x) = (pos1 == pos2 ? (x - mu[order])^2 : (x - mu[order])) * expnegf(F, F.I[order][j]..., x)
            result1[links[order - 1]'=>j] = quadgk(f, F.domain[order]...; maxevals=100)[1]
        end
    else
        result1 = ITensor(links[pos1 - 1]', links[pos1])
        for j in 1:npivots[pos1 - 1]
            for k in 1:npivots[pos1]
                f(x) = (pos1 == pos2 ? (x - mu[pos1])^2 : (x - mu[pos1])) * expnegf(F, F.I[pos1][j]..., x, F.J[pos1 + 1][k]...)
                result1[links[pos1 - 1]'=>j, links[pos1]=>k] = quadgk(f, F.domain[pos1]...; maxevals=100)[1]
            end
        end
    end

    result2 = undef
    if pos1 != pos2
        if pos2 == 1
            result2 = ITensor(links[1])
            for j in 1:npivots[1]
                f(x) = (x - mu[1]) * expnegf(F, x, F.J[2][j]...)
                result2[links[1]=>j] = quadgk(f, F.domain[1]...; maxevals=100)[1]
            end
        elseif pos2 == order
            result2 = ITensor(links[order - 1]')
            for j in 1:npivots[order - 1]
                f(x) = (x - mu[order]) * expnegf(F, F.I[order][j]..., x)
                result2[links[order - 1]'=>j] = quadgk(f, F.domain[order]...; maxevals=100)[1]
            end
        else
            result2 = ITensor(links[pos2 - 1]', links[pos2])
            for j in 1:npivots[pos2 - 1]
                for k in 1:npivots[pos2]
                    f(x) = (x - mu[pos2]) * expnegf(F, F.I[pos2][j]..., x, F.J[pos2 + 1][k]...)
                    result2[links[pos2 - 1]'=>j, links[pos2]=>k] = quadgk(f, F.domain[pos2]...; maxevals=100)[1]
                end
            end
        end
    end

    result = (pos1 == 1 ? result1 : integrals[1]) * skeleton[1]
    for i in 2:order-1
        if pos1 == i
            result *= result1 * skeleton[i]
        elseif pos2 == i
            result *= result2 * skeleton[i]
        else
            result *= integrals[i] * skeleton[i]
        end
    end
    if pos1 == order
        result *= result1
    elseif pos2 == order
        result *= result2
    else
        result *= integrals[order]
    end
    return result[]
end

# Draws a single iid sample from the TT-represented distribution F
function sample_from_tt(F::ResFunc{T, N}, integrals::Vector{ITensor}, skeleton::Vector{ITensor}, links::Vector{Index}) where {T, N}
    order = F.ndims
    npivots = [length(F.I[i]) for i in 2:order]
    sample = Vector{T}(undef, order)

    rel_tol = 0.001

    for count in 1:order
        Renv = undef
        if count != order
            Renv = skeleton[order - 1] * integrals[order]
            for i in order-1:-1:count+1
                Renv *= skeleton[i - 1] * integrals[i]
            end
        end
        u = rand()
        println("u_$count = $u")
        flush(stdout)
        a, b = F.domain[count]
        abs_tol = rel_tol * abs(b - a)

        normi = undef
        if count == order
            f1(x) = expnegf(F, sample[1:order-1]..., x)
            normi = quadgk(f1, F.domain[count]...; maxevals=100)[1]
        else
            normi = ITensor(links[count])
            for j in 1:npivots[count]
                f1i(x) = count == 1 ? expnegf(F, x, F.J[2][j]...) : expnegf(F, sample[1:count-1]..., x, F.J[count + 1][j]...)
                normi[links[count]=>j] = quadgk(f1i, F.domain[count]...; maxevals=100)[1]
            end
            normi *= Renv
        end

        while b - a > abs_tol
            mid = (a + b) / 2
            cdfi = undef
            if count == order
                f2(x) = expnegf(F, sample[1:order-1]..., x)
                cdfi = quadgk(f2, F.domain[count][1], mid; maxevals=100)[1]
            else
                cdfi = ITensor(links[count])
                for j in 1:npivots[count]
                    f2i(x) = count == 1 ? expnegf(F, x, F.J[2][j]...) : expnegf(F, sample[1:count-1]..., x, F.J[count + 1][j]...)
                    cdfi[links[count]=>j] = quadgk(f2i, F.domain[count][1], mid; maxevals=100)[1]
                end
                cdfi *= Renv
            end
            if cdfi[] / normi[] < u
                a = mid
            else
                b = mid
            end
        end
        sample[count] = (a + b) / 2
    end
    return sample
end

function compute_marginal(F::ResFunc{T, N}, integrals::Vector{ITensor}, skeleton::Vector{ITensor}, links::Vector{Index}, pos::Int64) where {T, N}
    order = F.ndims
    npivots = [length(F.I[i]) for i in 2:order]

    nbins = 100
    result = zeros(nbins, nbins)
    
    Lenv = undef
    Renv = undef
    if pos != 1
        Lenv = integrals[1] * skeleton[1]
        for i in 2:pos-1
            Lenv *= integrals[i] * skeleton[i]
        end
    end
    if pos + 1 != order
        Renv = skeleton[order - 1] * integrals[order]
        for i in order-1:-1:pos+2
            Renv *= skeleton[i - 1] * integrals[i]
        end
    end
    xlist = LinRange(F.domain[pos]..., nbins + 1)
    ylist = LinRange(F.domain[pos + 1]..., nbins + 1)
    
    for j in 1:nbins
        for k in 1:nbins
            resulti = undef
            if pos == 1
                resulti = ITensor(links[2])
                for l in 1:npivots[2]
                    resulti[links[2]=>l] = expnegf(F, xlist[j], ylist[k], F.J[3][l]...)
                end
                resulti *= Renv
            elseif pos + 1 == order
                resulti = ITensor(links[order - 2]')
                for i in 1:npivots[order - 2]
                    resulti[links[order - 2]'=>i] = expnegf(F, F.I[order - 1][i]..., xlist[j], ylist[k])
                end
                resulti *= Lenv
            else
                resulti = ITensor(links[pos - 1]', links[pos + 1])
                for i in 1:npivots[pos - 1]
                    for l in 1:npivots[pos + 1]
                        resulti[links[pos - 1]'=>i, links[pos + 1]=>l] = expnegf(F, F.I[pos][i]..., xlist[j], ylist[k], F.J[pos + 2][l]...)
                    end
                end
                resulti *= Lenv * Renv
            end
            result[j, k] = resulti[]
        end
    end
    return result
end

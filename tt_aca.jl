using Random
using Distributions
using QuadGK
using MPI
using LinearAlgebra
using ITensors

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

    # Constructor
    function ResFunc(f, domain::NTuple{N, Tuple{T, T}}, cutoff::T) where {T, N}
        new{T, N}(f, N, 0, domain, [[[T[]]]; [Vector{T}[] for _ in 2:N]], [[[T[]]]; [Vector{T}[] for _ in 2:N]], cutoff, 0.0)
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
        new_sign = ones(k - iter + 1, k - iter + 1)
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
end

# Updates I and J by inserting a new pivot at the current unfolding (pos)
function updateIJ(F::ResFunc{T, N}, ij::NTuple{N, T}) where {T, N}
    push!(F.I[F.pos + 1], [ij[j] for j in 1:F.pos])
    push!(F.J[F.pos + 1], [ij[j] for j in F.pos+1:F.ndims])
end

# Main ACA routine
# Takes multidimensional function F, and sequentially finds pivots which roughly maximize a dynamical residual function, moving from one unfolding to the next
# Pivots in each unfolding depend on pivots found for the previous unfolding
function continuous_aca(F::ResFunc{T, N}, rank::Vector{Int64}, n_chains::Int64, n_samples::Int64, jump_width::Float64, mpi_comm::MPI.Comm) where {T, N}
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
        for r in length(F.I[i + 1])+1:rank[i]
            # Determine number of tasks for the current process
            elements_per_task = div(n_chains_total, mpi_size)
            # Extra tasks to be carried out by the root process at the end
            remainder = rem(n_chains_total, mpi_size)
            local_xy = fill(Tuple(fill(0.0, order)), elements_per_task)
            local_res = fill(0.0, elements_per_task)
            for k in 1:elements_per_task
                # Run multiple Markov chains in parallel, approximate position of the largest current residual across all walkers
                idx = mod(mpi_rank * elements_per_task + k - 1, length(F.I[i])) + 1
                local_xy[k], local_res[k] = max_metropolis(F, F.I[i][idx], n_samples, jump_width)
            end
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
                    xylist[k], reslist[k] = max_metropolis(F, F.I[i][idx], n_samples, jump_width)
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
            if res_new[] < F.cutoff
                break
            end
            updateIJ(F, xy[])
            if mpi_rank == 0
                println("rank = $r res = $(res_new[]) xy = $(xy[]) offset = $(F.offset)")
                flush(stdout)
            end
        end
    end

    return F.I, F.J
end

# MCMC sampler, searches function F and runs one Markov chain
# Outputs largest residual magnitude and corresponding position
function max_metropolis(F::ResFunc{T, N}, pivot::Vector{T}, n_samples::Int64, jump_width::Float64) where {T, N}
    order = F.ndims - F.pos + 1
    
    lb = [F.domain[i][1] for i in F.pos:F.ndims]
    ub = [F.domain[i][2] for i in F.pos:F.ndims]

    chain = zeros(n_samples, order)

    min_nlr = Inf
    max_xy = zeros(F.ndims)

    while true
        for k in 1:order
            chain[1, k] = rand() * (ub[k] - lb[k]) + lb[k]
        end
        fx = F(pivot..., chain[1, :]...)  # Call once
        if isfinite(fx)
            break
        end
    end

    for i in 2:n_samples
        p_new = zeros(order)
        for k in 1:order
            p_new[k] = rand(Normal(chain[i - 1, k], jump_width * (ub[k] - lb[k])))
            if p_new[k] < lb[k]
                p_new[k] = lb[k] + abs(p_new[k] - lb[k])
            elseif p_new[k] > ub[k]
                p_new[k] = ub[k] - abs(p_new[k] - ub[k])
            end
        end

        arg_old = [pivot; [chain[i - 1, k] for k in 1:order]]
        arg_new = [pivot; [p_new[k] for k in 1:order]]
        acceptance_prob = 0.0
        if isfinite(F.f(arg_new...))
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

    return Tuple(max_xy), min_nlr
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
        integrals[1][links[1]=>j] = quadgk(f, F.domain[1]...; maxevals=10^3)[1]
    end
    result = integrals[1]
    println("i = 1\n")
    println(integrals[1])
    AIJ = zeros(npivots[1], npivots[1])
    for j in 1:npivots[1]
        for k in 1:npivots[1]
            AIJ[j, k] = expnegf(F, F.I[2][j]..., F.J[2][k]...)
        end
    end
    AIJinv = inv(AIJ)
    println(norm(AIJ * AIJinv - I))
    flush(stdout)
    skeleton[1] = ITensor(links[1], links[1]')
    for j in 1:npivots[1]
        for k in 1:npivots[1]
            skeleton[1][links[1]=>j, links[1]'=>k] = AIJinv[j, k]
        end
    end
    result *= skeleton[1]
    println(result)
    for i in 2:order-1
        links[i] = Index(npivots[i], "Link,l=$i")
        integrals[i] = ITensor(links[i - 1]', links[i])
        for j in 1:npivots[i - 1]
            for k in 1:npivots[i]
                f(x) = expnegf(F, F.I[i][j]..., x, F.J[i + 1][k]...)
                integrals[i][links[i - 1]'=>j, links[i]=>k] = quadgk(f, F.domain[i]...; maxevals=10^3)[1]
            end
        end
        println("\ni = $i\n")
        println(integrals[i])
        AIJ = zeros(npivots[i], npivots[i])
        for j in 1:npivots[i]
            for k in 1:npivots[i]
                AIJ[j, k] = expnegf(F, F.I[i + 1][j]..., F.J[i + 1][k]...)
            end
        end
        AIJinv = inv(AIJ)
        println(norm(AIJ * AIJinv - I))
        flush(stdout)
        skeleton[i] = ITensor(links[i], links[i]')
        for j in 1:npivots[i]
            for k in 1:npivots[i]
                skeleton[i][links[i]=>j, links[i]'=>k] = AIJinv[j, k]
            end
        end
        result *= integrals[i] * skeleton[i]
        println(result)
    end
    integrals[order] = ITensor(links[order - 1]')
    for j in 1:npivots[order - 1]
        f(x) = expnegf(F, F.I[order][j]..., x)
        integrals[order][links[order - 1]'=>j] = quadgk(f, F.domain[order]...; maxevals=10^3)[1]
    end
    println("\ni = $order\n")
    println(integrals[order])
    println()
    flush(stdout)
    result *= integrals[order]
    return result[], integrals, skeleton, links
end

# Draws a single iid sample from the TT-represented distribution F
function sample_from_tt(F::ResFunc{T, N}, integrals::Vector{ITensor}, skeleton::Vector{ITensor}, links::Vector{Index}) where {T, N}
    order = F.ndims
    npivots = [length(F.I[i]) for i in 2:order]
    sample = Vector{T}(undef, order)

    rel_tol = 1.0e-3

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

        f(x) = if count == 1
            expnegf(F, x, F.J[2][j]...)
        elseif count == order
            expnegf(F, sample[1:order-1], x)
        else
            expnegf(F, sample[1:count-1], x, F.J[count + 1][j]...)
        end

        normi = ITensor(links[count])
        for j in 1:npivots[count]
            normi[links[count]=>j] = quadgk(f, F.domain[count]...; maxevals=10^3)[1]
        end
        normi *= Renv

        while b - a > abs_tol
            mid = (a + b) / 2
            cdfi = ITensor(links[count])
            for j in 1:npivots[count]
                cdfi[links[count]=>j] = quadgk(f, F.domain[count][1], mid; maxevals=10^3)[1]
            end
            cdfi *= Renv
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

using Random
using Distributions
using QuadGK
using MPI

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
    mu::Vector{T}
    sigma::Vector{T}

    # Constructor
    function ResFunc(f, domain::NTuple{N, Tuple{T, T}}, cutoff::T, mu::Vector{T}, sigma::Vector{T}) where {T, N}
        new{T, N}(f, N, 0, domain, [[[T[]]]; [Vector{T}[] for _ in 2:N]], [[[T[]]]; [Vector{T}[] for _ in 2:N]], cutoff, 0.0, mu, sigma)
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
    for iter in 0:k
        new = Complex.(zeros(k - iter + 1, k - iter + 1))
        for idx in CartesianIndices(new)
            if iter == 0
                row = idx[1] == k + 1 ? x : F.I[F.pos + 1][idx[1]]
                col = idx[2] == k + 1 ? y : F.J[F.pos + 1][idx[2]]
                new[idx] = F.f(row..., col...) - F.offset
            else
                f = old[idx[1] + 1, idx[2] + 1]
                # df = f - old[idx[1] + 1, 1] - old[1, idx[2] + 1] + old[1, 1]
                # new[idx] = f - log(Complex(1.0 - real(exp(df))))
                df = old[1, 1] - old[idx[1] + 1, 1] - old[1, idx[2] + 1]
                new[idx] = -log(Complex(real(exp(-f)) - real(exp(df))))
            end
        end
        old = deepcopy(new)
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
        n_chains_total = max(n_chains, mpi_size)
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
                local_xy[k], local_res[k] = max_metropolis(F, n_samples, jump_width)
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
                    xylist[k], reslist[k] = max_metropolis(F, n_samples, jump_width)
                end
            end
            
            # Find position of largest residuals
            idx = argmax(reslist)
            xy = [xylist[idx]]
            MPI.Bcast!(xy, 0, mpi_comm)
            res_new = [reslist[idx]]
            if isempty(F.I[i + 1]) || F.f(xy[]...) < F.offset
                offset_new = [F.f(xy[]...)]
                MPI.Bcast!(offset_new, 0, mpi_comm)
                F.offset = offset_new[]
                res_new = [exp(-real(F(xy[]...)))]
            end
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
function max_metropolis(F::ResFunc{T, N}, n_samples::Int64, jump_width::Float64) where {T, N}
    order = F.ndims
    
    lb = [F.domain[i][1] for i in 1:F.ndims]
    ub = [F.domain[i][2] for i in 1:F.ndims]

    chain = zeros(n_samples, order)

    max_res = 0.0
    max_xy = zeros(F.ndims)

    while true
        for k in 1:order
            chain[1, k] = Inf
            while chain[1, k] < F.domain[k][1] || chain[1, k] > F.domain[k][2]
                chain[1, k] = F.mu[k] + sqrt(F.sigma[k]) * randn()
            end
        end
        if expnegf(F, chain[1, 1:order]...) > 0.0 && exp(-real(F(chain[1, 1:order]...))) > 0.0
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

        arg_old = [chain[i - 1, k] for k in 1:order]
        arg_new = [p_new[k] for k in 1:order]
        f_old = real(F(arg_old...))
        f_new = real(F(arg_new...))
        acceptance_prob = min(1, exp(f_old - f_new))
        
        if rand() < acceptance_prob
            chain[i, :] = p_new
            if exp(-f_new) > max_res
                max_res = exp(-f_new)
                max_xy = arg_new
            end
        else
            chain[i, :] = chain[i - 1, :]
        end
    end

    return Tuple(max_xy), max_res
end

# Approximates normalization constant, or partition function (if F is a Boltzmann distribution)
# Involes univariate integral calculations along all dimensions
function compute_norm(F::ResFunc{T, N}) where {T, N}
    order = F.ndims
    npivots = [length(F.I[i]) for i in 2:order]
    println("i = 1")
    norm = zeros(1, npivots[1])
    for j in 1:npivots[1]
        println("j = $j")
        f(x) = expnegf(F, x, F.J[2][j]...)
        nbins = 500
        xlist = LinRange(F.domain[1]..., nbins + 1)
        for k in 1:nbins
            print("$(f(xlist[k])), ")
        end
        println()
        @time norm[j] = quadgk(f, F.domain[1]...; rtol=1.0e-4, atol=1.0e-6, maxevals=10^5)[1]
        println(norm[j])
    end
    AIJ = zeros(npivots[1], npivots[1])
    for j in 1:npivots[1]
        for k in 1:npivots[1]
            AIJ[j, k] = expnegf(F, F.I[2][j]..., F.J[2][k]...)
        end
    end
    norm *= inv(AIJ)
    for i in 2:order-1
        println("i = $i")
        normi = zeros((npivots[i - 1], npivots[i]))
        for j in 1:npivots[i - 1]
            for k in 1:npivots[i]
                println("j = $j k = $k")
                f(x) = expnegf(F, F.I[i][j]..., x, F.J[i + 1][k]...)
                @time normi[j, k] = quadgk(f, F.domain[i]...)[1]
            end
        end
        AIJ = zeros(npivots[i], npivots[i])
        for j in 1:npivots[i]
            for k in 1:npivots[i]
                AIJ[j, k] = expnegf(F, F.I[i + 1][j]..., F.J[i + 1][k]...)
            end
        end
        norm *= normi * inv(AIJ)
    end
    println("i = $order")
    R = zeros(npivots[order - 1])
    for j in 1:npivots[order - 1]
        println("j = $j")
        f(x) = expnegf(F, F.I[order][j]..., x)
        @time R[j] = quadgk(f, F.domain[order]...)[1]
    end
    norm *= R
    return norm[]
end

# Approximates the normalized 1,2-marginal at (x1, x2)
# Involves univariate integral calculations along x3,...,xd
function compute_marginal12(F::ResFunc{T, N}, x1::T, x2::T) where {T, N}
    order = F.ndims
    npivots = [length(F.I[i]) for i in 2:order]
    result = zeros(1, npivots[1])
    for j in 1:npivots[1]
        result[j] = expnegf(F, x1, F.J[2][j]...)
    end
    AIJ = zeros(npivots[1], npivots[1])
    for j in 1:npivots[1]
        for k in 1:npivots[1]
            AIJ[j, k] = expnegf(F, F.I[2][j]..., F.J[2][k]...)
        end
    end
    result *= inv(AIJ)
    for i in 2:order-1
        resulti = zeros((npivots[i - 1], npivots[i]))
        for j in 1:npivots[i - 1]
            for k in 1:npivots[i]
                if i == 2
                    resulti[j, k] = expnegf(F, F.I[i][j]..., x2, F.J[i + 1][k]...)
                else
                    f(x) = expnegf(F, F.I[i][j]..., x, F.J[i + 1][k]...)
                    resulti[j, k] = quadgk(f, F.domain[i]...)[1]
                end
            end
        end
        AIJ = zeros(npivots[i], npivots[i])
        for j in 1:npivots[i]
            for k in 1:npivots[i]
                AIJ[j, k] = expnegf(F, F.I[i + 1][j]..., F.J[i + 1][k]...)
            end
        end
        result *= resulti * inv(AIJ)
    end
    R = zeros(npivots[order - 1])
    for j in 1:npivots[order - 1]
        f(x) = expnegf(F, F.I[order][j]..., x)
        R[j] = quadgk(f, F.domain[order]...)[1]
    end
    result *= R
    return result[]
end

# Approximates mean values for each dimension
function compute_mu(F::ResFunc{T, N}, norm::T) where {T, N}
    order = F.ndims
    npivots = [length(F.I[i]) for i in 2:order]
    mu = [zeros(1, npivots[1]) for _ in 1:order]
    for j in 1:npivots[1]
        for pos in 1:order
            f(x) = if pos == 1
                x * expnegf(F, x, F.J[2][j]...)
            else
                expnegf(F, x, F.J[2][j]...)
            end
            mu[pos][j] = quadgk(f, F.domain[1]...)[1]
        end
    end
    AIJ = zeros(npivots[1], npivots[1])
    for j in 1:npivots[1]
        for k in 1:npivots[1]
            AIJ[j, k] = expnegf(F, F.I[2][j]..., F.J[2][k]...)
        end
    end
    for pos in 1:order
        mu[pos] *= inv(AIJ)
    end
    for i in 2:order-1
        normi = [zeros((npivots[i - 1], npivots[i])) for _ in 1:order]
        prev = deepcopy(mu)
        mu = [zeros(1, npivots[i]) for _ in 1:order]
        for j in 1:npivots[i - 1]
            for k in 1:npivots[i]
                for pos in 1:order
                    f(x) = if pos == i
                        x * expnegf(F, F.I[i][j]..., x, F.J[i + 1][k]...)
                    else
                        expnegf(F, F.I[i][j]..., x, F.J[i + 1][k]...)
                    end
                    normi[pos][j, k] = quadgk(f, F.domain[i]...)[1]
                end
            end
        end
        AIJ = zeros(npivots[i], npivots[i])
        for j in 1:npivots[i]
            for k in 1:npivots[i]
                AIJ[j, k] = expnegf(F, F.I[i + 1][j]..., F.J[i + 1][k]...)
            end
        end
        for pos in 1:order
            mu[pos] = prev[pos] * normi[pos] * inv(AIJ)
        end
    end
    R = [zeros(npivots[order - 1]) for _ in 1:order]
    prev = deepcopy(mu)
    mu = zeros(order)
    for j in 1:npivots[order - 1]
        for pos in 1:order
            f(x) = if pos == order
                x * expnegf(F, F.I[order][j]..., x)
            else
                expnegf(F, F.I[order][j]..., x)
            end
            R[pos][j] = quadgk(f, F.domain[order]...)[1]
        end
    end
    for pos in 1:order
        mu[pos] = (prev[pos] * R[pos])[]
    end
    return [mu[pos] / norm[] for pos in 1:order]
end

# Approximates variances for each dimension
function compute_var(F::ResFunc{T, N}, norm::T, mu::Vector{T}) where {T, N}
    order = F.ndims
    npivots = [length(F.I[i]) for i in 2:order]
    var = [zeros(1, npivots[1]) for _ in 1:order]
    for j in 1:npivots[1]
        for pos in 1:order
            f(x) = if pos == 1
                (x - mu[1]) ^ 2 * expnegf(F, x, F.J[2][j]...)
            else
                expnegf(F, x, F.J[2][j]...)
            end
            var[pos][j] = quadgk(f, F.domain[1]...)[1]
        end
    end
    AIJ = zeros(npivots[1], npivots[1])
    for j in 1:npivots[1]
        for k in 1:npivots[1]
            AIJ[j, k] = expnegf(F, F.I[2][j]..., F.J[2][k]...)
        end
    end
    for pos in 1:order
        var[pos] *= inv(AIJ)
    end
    for i in 2:order-1
        normi = [zeros((npivots[i - 1], npivots[i])) for _ in 1:order]
        prev = deepcopy(var)
        var = [zeros(1, npivots[i]) for _ in 1:order]
        for j in 1:npivots[i - 1]
            for k in 1:npivots[i]
                for pos in 1:order
                    f(x) = if pos == i
                        (x - mu[i]) ^ 2 * expnegf(F, F.I[i][j]..., x, F.J[i + 1][k]...)
                    else
                        expnegf(F, F.I[i][j]..., x, F.J[i + 1][k]...)
                    end
                    normi[pos][j, k] = quadgk(f, F.domain[i]...)[1]
                end
            end
        end
        AIJ = zeros(npivots[i], npivots[i])
        for j in 1:npivots[i]
            for k in 1:npivots[i]
                AIJ[j, k] = expnegf(F, F.I[i + 1][j]..., F.J[i + 1][k]...)
            end
        end
        for pos in 1:order
            var[pos] = prev[pos] * normi[pos] * inv(AIJ)
        end
    end
    R = [zeros(npivots[order - 1]) for _ in 1:order]
    prev = deepcopy(var)
    var = zeros(order)
    for j in 1:npivots[order - 1]
        for pos in 1:order
            f(x) = if pos == order
                (x - mu[order]) ^ 2 * expnegf(F, F.I[order][j]..., x)
            else
                expnegf(F, F.I[order][j]..., x)
            end
            R[pos][j] = quadgk(f, F.domain[order]...)[1]
        end
    end
    for pos in 1:order
        var[pos] = (prev[pos] * R[pos])[]
    end
    return [var[pos] / norm[] for pos in 1:order]
end

function compute_12(F::ResFunc{T, N}, x1::T, x2::T) where {T, N}
    npivots = [length(F.I[2])]
    result = zeros(1, npivots[1])
    for j in 1:npivots[1]
        result[j] = expnegf(F, x1, F.J[2][j]...)
    end
    AIJ = zeros(npivots[1], npivots[1])
    for j in 1:npivots[1]
        for k in 1:npivots[1]
            AIJ[j, k] = expnegf(F, F.I[2][j]..., F.J[2][k]...)
        end
    end
    result *= inv(AIJ)
    R = zeros(npivots[1])
    for j in 1:npivots[1]
        R[j] = expnegf(F, F.I[2][j]..., x2)
    end
    result *= R
    return result[]
end

# Draws a single iid sample from the normalized TT-represented distribution F
function sample_from_tt(F::ResFunc{T, N}) where {T, N}
    order = F.ndims
    npivots = [length(F.I[i]) for i in 2:order]
    sample = Vector{T}(undef, order)

    rel_tol = 1.0e-3

    for count in 1:order
        Lenv = undef
        Renv = undef
        if count != 1
            Lenv = zeros(1, npivots[1])
            for j in 1:npivots[1]
                Lenv[j] = expnegf(F, sample[1], F.J[2][j]...)
            end
            AIJ = zeros(npivots[1], npivots[1])
            for j in 1:npivots[1]
                for k in 1:npivots[1]
                    AIJ[j, k] = expnegf(F, F.I[2][j]..., F.J[2][k]...)
                end
            end
            Lenv *= inv(AIJ)
            for i in 2:count-1
                Lenvi = zeros(npivots[i - 1], npivots[i])
                for j in 1:npivots[i - 1]
                    for k in 1:npivots[i]
                        Lenvi[j, k] = expnegf(F, F.I[i][j]..., sample[i], F.J[i + 1][k]...)
                    end
                end
                AIJ = zeros(npivots[i], npivots[i])
                for j in 1:npivots[i]
                    for k in 1:npivots[i]
                        AIJ[j, k] = expnegf(F, F.I[i + 1][j]..., F.J[i + 1][k]...)
                    end
                end
                Lenv *= Lenvi * inv(AIJ)
            end
        end
        if count != order
            AIJ = zeros(npivots[order - 1], npivots[order - 1])
            for j in 1:npivots[order - 1]
                for k in 1:npivots[order - 1]
                    AIJ[j, k] = expnegf(F, F.I[order][j]..., F.J[order][k]...)
                end
            end
            Renv = inv(AIJ)
            R = zeros(npivots[order - 1])
            for j in 1:npivots[order - 1]
                f(x) = expnegf(F, F.I[order][j]..., x)
                R[j] = quadgk(f, F.domain[order]...)[1]
            end
            Renv *= R
            for i in order-1:-1:count+1
                Renvi = zeros(npivots[i - 1], npivots[i])
                for j in 1:npivots[i - 1]
                    for k in 1:npivots[i]
                        f(x) = expnegf(F, F.I[i][j]..., x, F.J[i + 1][k]...)
                        Renvi[j, k] = quadgk(f, F.domain[i]...)[1]
                    end
                end
                AIJ = zeros(npivots[i], npivots[i])
                for j in 1:npivots[i]
                    for k in 1:npivots[i]
                        AIJ[j, k] = expnegf(F, F.I[i + 1][j]..., F.J[i + 1][k]...)
                    end
                end
                Renv = Renvi * inv(AIJ) * Renv
            end
        end
        u = rand()
        println("u_$count = $u")
        a, b = F.domain[count]
        abs_tol = rel_tol * abs(b - a)

        normi = undef
        if count == 1
            normi = zeros(1, npivots[1])
            for j in 1:npivots[1]
                f(x) = expnegf(F, x, F.J[2][j]...)
                normi[j] = quadgk(f, F.domain[1]...)[1]
            end
            normi *= Renv
        elseif count == order
            normi = zeros(npivots[order - 1])
            for j in 1:npivots[order - 1]
                f(x) = expnegf(F, F.I[order][j]..., x)
                normi[j] = quadgk(f, F.domain[order]...)[1]
            end
            normi = Lenv * normi
        else
            normi = zeros((npivots[count - 1], npivots[count]))
            for j in 1:npivots[count - 1]
                for k in 1:npivots[count]
                    f(x) = expnegf(F, F.I[count][j]..., x, F.J[count + 1][k]...)
                    normi[j, k] = quadgk(f, F.domain[count]...)[1]
                end
            end
            normi = Lenv * normi * Renv
        end

        while b - a > abs_tol
            mid = (a + b) / 2
            cdfi = undef
            if count == 1
                cdfi = zeros(1, npivots[1])
                for j in 1:npivots[1]
                    f(x) = expnegf(F, x, F.J[2][j]...)
                    cdfi[j] = quadgk(f, F.domain[1][1], mid)[1]
                end
                cdfi *= Renv
            elseif count == order
                cdfi = zeros(npivots[order - 1])
                for j in 1:npivots[order - 1]
                    f(x) = expnegf(F, F.I[order][j]..., x)
                    cdfi[j] = quadgk(f, F.domain[order][1], mid)[1]
                end
                cdfi = Lenv * cdfi
            else
                cdfi = zeros((npivots[count - 1], npivots[count]))
                for j in 1:npivots[count - 1]
                    for k in 1:npivots[count]
                        f(x) = expnegf(F, F.I[count][j]..., x, F.J[count + 1][k]...)
                        cdfi[j, k] = quadgk(f, F.domain[count][1], mid)[1]
                    end
                end
                cdfi = Lenv * cdfi * Renv
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

function compute_marginal(F::ResFunc{T, N}, pos::Int64, norm::T) where {T, N}
    order = F.ndims
    npivots = [length(F.I[i]) for i in 2:order]

    nbins = 100
    result = zeros(nbins, nbins)
    
    Lenv = undef
    Renv = undef
    if pos != 1
        Lenv = zeros(1, npivots[1])
        for j in 1:npivots[1]
            f(x) = expnegf(F, x, F.J[2][j]...)
            Lenv[j] = quadgk(f, F.domain[1]...)[1]
        end
        AIJ = zeros(npivots[1], npivots[1])
        for j in 1:npivots[1]
            for k in 1:npivots[1]
                AIJ[j, k] = expnegf(F, F.I[2][j]..., F.J[2][k]...)
            end
        end
        Lenv *= inv(AIJ)
        for i in 2:pos-1
            Lenvi = zeros(npivots[i - 1], npivots[i])
            for j in 1:npivots[i - 1]
                for k in 1:npivots[i]
                    f(x) = expnegf(F, F.I[i][j]..., x, F.J[i + 1][k]...)
                    Lenvi[j, k] = quadgk(f, F.domain[i]...)[1]
                end
            end
            AIJ = zeros(npivots[i], npivots[i])
            for j in 1:npivots[i]
                for k in 1:npivots[i]
                    AIJ[j, k] = expnegf(F, F.I[i + 1][j]..., F.J[i + 1][k]...)
                end
            end
            Lenv *= Lenvi * inv(AIJ)
        end
    end
    if pos + 1 != order
        AIJ = zeros(npivots[order - 1], npivots[order - 1])
        for j in 1:npivots[order - 1]
            for k in 1:npivots[order - 1]
                AIJ[j, k] = expnegf(F, F.I[order][j]..., F.J[order][k]...)
            end
        end
        Renv = inv(AIJ)
        R = zeros(npivots[order - 1])
        for j in 1:npivots[order - 1]
            f(x) = expnegf(F, F.I[order][j]..., x)
            R[j] = quadgk(f, F.domain[order]...)[1]
        end
        Renv *= R
        for i in order-1:-1:pos+2
            Renvi = zeros(npivots[i - 1], npivots[i])
            for j in 1:npivots[i - 1]
                for k in 1:npivots[i]
                    f(x) = expnegf(F, F.I[i][j]..., x, F.J[i + 1][k]...)
                    Renvi[j, k] = quadgk(f, F.domain[i]...)[1]
                end
            end
            AIJ = zeros(npivots[i], npivots[i])
            for j in 1:npivots[i]
                for k in 1:npivots[i]
                    AIJ[j, k] = expnegf(F, F.I[i + 1][j]..., F.J[i + 1][k]...)
                end
            end
            Renv = Renvi * inv(AIJ) * Renv
        end
    end
    xlist = LinRange(F.domain[pos]..., nbins + 1)
    ylist = LinRange(F.domain[pos + 1]..., nbins + 1)
    
    for j in 1:nbins
        for k in 1:nbins
            resulti = undef
            if pos == 1
                resulti = zeros(1, npivots[2])
                for l in 1:npivots[2]
                    resulti[l] = expnegf(F, xlist[j], ylist[k], F.J[3][l]...)
                end
                resulti *= Renv
            elseif pos + 1 == order
                resulti = zeros(npivots[order - 2])
                for i in 1:npivots[order - 2]
                    resulti[i] = expnegf(F, F.I[order - 1][i]..., xlist[j], ylist[k])
                end
                resulti = Lenv * resulti
            else
                resulti = zeros((npivots[pos - 1], npivots[pos + 1]))
                for i in 1:npivots[pos - 1]
                    for l in 1:npivots[pos + 1]
                        resulti[i, l] = expnegf(F, F.I[pos][i]..., xlist[j], ylist[k], F.J[pos + 2][l]...)
                    end
                end
                resulti = Lenv * resulti * Renv
            end
            result[j, k] = resulti[] / norm
        end
    end
    return result
end

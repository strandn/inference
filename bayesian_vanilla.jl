using Random
using MPI
using LinearAlgebra
using Statistics

"""
    estimate_log_evidence_uniform(neglogposterior;
                                  domain::NTuple{N,Tuple{Float64,Float64}},
                                  comm::MPI.Comm,
                                  nsamples::Int=10_000,
                                  rng::AbstractRNG=Random.GLOBAL_RNG)

Estimates log normalization constant log(Z) via uniform sampling and Monte Carlo integration.

Arguments:
- `neglogposterior`: function accepting `x...`, returning scalar negative log-density (unnormalized).
- `domain`: tuple of tuples defining [a,b] for each dimension.
- `comm`: MPI communicator.
- `nsamples`: number of samples per process.
- `rng`: optional random number generator.

Returns:
- `log(Z)` on rank 0, `nothing` on other ranks.
"""
function estimate_log_evidence_uniform(neglogposterior;
        domain::NTuple{N,Tuple{Float64,Float64}},
        comm::MPI.Comm,
        nsamples::Int=10_000,
        rng::AbstractRNG=Random.GLOBAL_RNG) where {N}

    rank = MPI.Comm_rank(comm)

    # Precompute domain volume
    vol = prod(b - a for (a, b) in domain)

    # Generate one uniform sample from the domain
    function uniform_sample()
        return [rand(rng) * (b - a) + a for (a, b) in domain]
    end

    # Streaming log-sum-exp accumulation
    xmax_local = -Inf
    sumexp_shifted = 0.0

    for _ in 1:nsamples
        fx = neglogposterior(uniform_sample()...)
        val = -fx

        if val > xmax_local
            sumexp_shifted = sumexp_shifted * exp(xmax_local - val) + 1.0
            xmax_local = val
        else
            sumexp_shifted += exp(val - xmax_local)
        end
    end

    # Allreduce: global max
    xmax_global = MPI.Allreduce(xmax_local, MPI.MAX, comm)

    # Recalculate local sum with global xmax
    sumexp_adjusted = 0.0
    for _ in 1:nsamples
        fx = neglogposterior(uniform_sample()...)
        sumexp_adjusted += exp(-fx - xmax_global)
    end

    # Allreduce: global sum and sample count
    sumexp_global = MPI.Allreduce(sumexp_adjusted, +, comm)
    N_total = MPI.Allreduce(nsamples, +, comm)

    if rank == 0
        logZ = log(vol) - log(N_total) + xmax_global + log(sumexp_global)
        return -logZ
    end
end

"""
    mcmc_mean_cov_parallel(neglogposterior;
                           domain::NTuple{N, Tuple{Float64, Float64}},
                           comm::MPI.Comm,
                           nchains::Int,
                           nsamples::Int=10_000,
                           burnin::Int=1000,
                           proposal_std::Float64=1.0,
                           rng_seed::Int=42)

Run MCMC with potentially multiple chains per MPI process and estimate the
mean and covariance matrix of the posterior distribution.

Arguments:
- `neglogposterior`: function taking x... and returning negative log posterior
- `domain`: tuple of length-N tuples specifying domain bounds
- `comm`: MPI communicator
- `nchains`: total number of chains (≥ MPI size)
- `nsamples`: number of samples per chain after burn-in
- `burnin`: burn-in steps per chain
- `proposal_std`: stddev of isotropic Gaussian proposal
- `rng_seed`: base seed for RNGs (per-process seed = rng_seed + rank)

Returns on rank 0:
- `mean::Vector{Float64}`
- `covariance::Matrix{Float64}`
Other ranks return `nothing`.
"""
function mcmc_mean_cov_parallel(neglogposterior;
        domain::NTuple{N, Tuple{Float64, Float64}},
        comm::MPI.Comm,
        nchains::Int,
        nsamples::Int=10_000,
        burnin::Int=1000,
        proposal_std::Float64=0.01,
        thin::Int=100,
        rng_seed::Int=42) where {N}

    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    ndim = N

    # Determine how many chains to run on this rank
    chains_per_rank = fill(nchains ÷ nprocs, nprocs)
    for i in 1:(nchains % nprocs)
        chains_per_rank[i] += 1
    end
    local_nchains = chains_per_rank[rank+1]

    rng = MersenneTwister(rng_seed + rank)

    function uniform_sample(domain)
        return [rand(rng)*(b - a) + a for (a, b) in domain]
    end

    nrecord = nsamples  # total samples to record *after thinning*
    steps_needed = burnin + nrecord * thin

    all_local_samples = Matrix{Float64}(undef, 0, ndim)

    for _ in 1:local_nchains
        x = uniform_sample(domain)
        fx = neglogposterior(x...)
        chain_samples = Matrix{Float64}(undef, nrecord, ndim)

        scale = [(b - a) for (a, b) in domain]

        sample_idx = 0
        for step in 1:steps_needed
            x_prop = x .+ randn(rng, ndim) .* scale .* proposal_std
            fx_prop = neglogposterior(x_prop...)

            log_accept_ratio = fx - fx_prop
            if log(rand(rng)) < log_accept_ratio
                x, fx = x_prop, fx_prop
            end

            if step > burnin && (step - burnin) % thin == 0
                sample_idx += 1
                chain_samples[sample_idx, :] .= x
            end
        end

        all_local_samples = vcat(all_local_samples, chain_samples)
    end

    # Flatten local samples and gather
    local_vec = vec(all_local_samples)
    gathered = MPI.gather(local_vec, comm, root=0)

    if rank == 0
        total_len = sum(length.(gathered))
        total_rows = total_len ÷ ndim

        if total_len % ndim != 0
            error("Mismatch: total gathered length $total_len is not divisible by ndim = $ndim")
        end

        all_samples = reshape(vcat(gathered...), ndim, total_rows)'  # shape (total_rows, ndim)
        mean_vec = vec(mean(all_samples, dims=1))
        cov_mat = cov(all_samples)
        return mean_vec, cov_mat
    end
end

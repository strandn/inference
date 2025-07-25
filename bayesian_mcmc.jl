using Random
using Statistics
using LinearAlgebra
using MPI

"""
    estimate_log_evidence_parallel(neglogposterior; 
                                   domain::NTuple{N,Tuple{Float64,Float64}},
                                   comm::MPI.Comm,
                                   nsamples::Int=10_000,
                                   burnin::Int=1000,
                                   proposal_std::Float64=0.01,
                                   rng::AbstractRNG=Random.GLOBAL_RNG)

Estimates the negative log evidence -log(Z) using Metropolis-Hastings MCMC
with MPI-parallelized walkers.

Arguments:
- `neglogposterior`: Function accepting `x...`, returning scalar negative log posterior.
- `domain`: Tuple of tuples specifying uniform initialization range for each dimension.
- `comm`: MPI communicator.
- `nsamples`: Number of samples after burn-in per walker.
- `burnin`: Burn-in samples to discard.
- `proposal_std`: Standard deviation for isotropic Gaussian proposal.
- `rng`: Optional random number generator.

Returns:
- `-log(Z)` on rank 0, `nothing` on other ranks.
"""
function estimate_log_evidence_parallel(neglogposterior;
        domain::NTuple{N,Tuple{Float64,Float64}},
        comm::MPI.Comm,
        nsamples::Int=10_000,
        burnin::Int=1000,
        proposal_std::Float64=0.01,
        rng::AbstractRNG=Random.GLOBAL_RNG) where {N}

    rank = MPI.Comm_rank(comm)

    ndim = N

    # Sample uniformly from the domain
    function uniform_sample(domain)
        return [rand(rng) * (b - a) + a for (a, b) in domain]
    end

    # Initialize walker
    x = uniform_sample(domain)
    fx = neglogposterior(x...)

    samples = Float64[]

    for step in 1:(burnin + nsamples)
        x_prop = x .+ randn(rng, ndim) .* [(b - a) for (a, b) in domain] .* proposal_std
        fx_prop = neglogposterior(x_prop...)

        log_accept_ratio = fx - fx_prop
        if log(rand(rng)) < log_accept_ratio
            x, fx = x_prop, fx_prop
        end

        if step > burnin
            push!(samples, fx)
        end
    end

    # Gather all samples to rank 0
    all_samples = MPI.gather(samples, comm, root=0)

    if rank == 0
        all_fx = reduce(vcat, all_samples)
        N_total = length(all_fx)

        # Log-sum-exp trick
        xmax = maximum(-all_fx)
        log_Z = -log(N_total) + (xmax + log(sum(exp.(-all_fx .- xmax))))
        return -log_Z
    end
end

using Random
using Statistics
using LinearAlgebra
using MPI

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
- `nsamples`: number of samples **per process**.
- `rng`: optional random number generator.

Returns:
- `log(Z)` on rank 0, `nothing` on other ranks.
"""
function estimate_log_evidence_uniform(neglogposterior;
        domain::NTuple{N,Tuple{Float64,Float64}},
        comm::MPI.Comm,
        nsamples::Int=10^4,
        rng::AbstractRNG=Random.GLOBAL_RNG) where {N}

    rank = MPI.Comm_rank(comm)

    # Precompute domain volume
    vol = prod(b - a for (a, b) in domain)

    # Uniform samples from domain
    function uniform_sample()
        return [rand(rng) * (b - a) + a for (a, b) in domain]
    end

    # Local samples and evaluations
    local_fx = [neglogposterior(uniform_sample()...) for _ in 1:nsamples]

    # Perform numerically stable log-sum-exp
    xmax_local = maximum(-local_fx)

    # Reduce across all processes
    xmax_global = MPI.allreduce(xmax_local, MPI.MAX, comm)
    shifted_local = sum(exp.(-fx - xmax_global) for fx in local_fx)
    sumexp_global = MPI.allreduce(shifted_local, +, comm)
    N_total = MPI.allreduce(nsamples, +, comm)

    if rank == 0
        logZ = log(vol) - log(N_total) + xmax_global + log(sumexp_global)
        return -logZ
    end
end

function estimate_log_evidence_TI(neglogposterior;
        domain::NTuple{N,Tuple{Float64,Float64}},
        comm::MPI.Comm,
        nsamples::Int=10_000,
        burnin::Int=1000,
        proposal_std::Float64=0.01,
        nbetas::Int=10,
        rng::AbstractRNG=Random.GLOBAL_RNG) where {N}

    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    ndim = N
    vol = prod(b - a for (a, b) in domain)
    logZ0 = log(vol)

    # Split beta ladder among processes
    betas = range(0.0, 1.0; length=nbetas)
    local_betas = Iterators.partition(betas, cld(nbetas, size)) |> collect
    my_betas = local_betas[rank + 1]

    # Sample uniformly from domain
    function uniform_sample()
        return [rand(rng) * (b - a) + a for (a, b) in domain]
    end

    # Function to run Metropolis at fixed β
    function run_mcmc(beta, x_init)
    x = copy(x_init)
    fx = neglogposterior(x...)
    fxs = Float64[]
    accepted = 0

    scale = [(b - a) for (a, b) in domain]  # domain-wise scale
    prop_std = proposal_std

    for step in 1:(burnin + nsamples)
        x_prop = x .+ randn(rng, ndim) .* scale .* prop_std
        fx_prop = neglogposterior(x_prop...)
        log_accept_ratio = beta * (fx - fx_prop)

        if log(rand(rng)) < log_accept_ratio
            x, fx = x_prop, fx_prop
            accepted += 1
        end

        if step > burnin
            push!(fxs, fx)
        end
    end

    acc_rate = accepted / (burnin + nsamples)
    if acc_rate < 0.1
        @warn "Low acceptance rate at β=$beta: $(round(acc_rate, digits=3))"
    elseif acc_rate > 0.5
        @info "High acceptance rate at β=$beta: $(round(acc_rate, digits=3))"
    end

    return mean(fxs), x  # return final sample too (for warm start)
end

    # Compute local expectations for assigned βs
    local_expectations = []
    x = uniform_sample()  # shared warm start

    for β in my_betas
        avg_fx, x = run_mcmc(β, x)  # warm start from last β
        push!(local_expectations, (β, -avg_fx))
    end

    # Gather all (β, E) pairs to rank 0
    gathered = MPI.gather(local_expectations, comm, root=0)

    if rank == 0
        all_expectations = reduce(vcat, gathered)
        sort!(all_expectations, by=x->x[1])  # Sort by 
        
        for (β, E) in all_expectations
            println("β = $β, E[-f(x)] = $E")
        end

        # Trapezoidal integration
        logZ = logZ0
        for i in 1:length(all_expectations)-1
            β₁, E₁ = all_expectations[i]
            β₂, E₂ = all_expectations[i+1]
            logZ += (β₂ - β₁) * (E₁ + E₂) / 2
        end
        return -logZ
    end
end

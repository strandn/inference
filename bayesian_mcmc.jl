using Random
using Statistics
using LinearAlgebra
using MPI

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
    function run_mcmc(beta)
        x = uniform_sample()
        fx = neglogposterior(x...)
        fxs = Float64[]
        for step in 1:(burnin + nsamples)
            x_prop = x .+ randn(rng, ndim) .* [(b - a) for (a, b) in domain] .* proposal_std
            fx_prop = neglogposterior(x_prop...)
            log_accept_ratio = beta * (fx - fx_prop)
            if log(rand(rng)) < log_accept_ratio
                x, fx = x_prop, fx_prop
            end
            if step > burnin
                push!(fxs, fx)
            end
        end
        return mean(fxs)
    end

    # Compute local expectations for assigned βs
    local_expectations = [(β, -run_mcmc(β)) for β in my_betas]

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

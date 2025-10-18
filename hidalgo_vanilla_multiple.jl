using Distributions

include("bayesian_vanilla.jl")

function V(r, data)
    K = 3
    mu = r[1:3]
    lambda = exp.(r[4:6])
    q = [r[7:8]; 1.0 - r[7] - r[8]]
    beta = r[9]

    # ---- Hyperparameters ----
    M = mean(data)
    R = maximum(data) - minimum(data)
    κ = 4 / R^2
    α = 2.0
    g = 0.2
    h = 100 * g / (α * R^2)

    # ---- Priors ----
    # Means
    logprior_mu = sum(logpdf(Normal(M, sqrt(1/κ)), μk) for μk in mu)

    # Precisions (Gamma(α, β) with shape α, rate β)
    logprior_lambda = sum(logpdf(Gamma(α, beta), λk) for λk in lambda)

    # Hyperprior on β
    logprior_beta = logpdf(Gamma(g, h), beta)

    # Mixture weights (Dirichlet(1,...,1))
    # log Dirichlet(1,...,1) = 0 if q in simplex, -Inf otherwise
    logprior_q = (all(q .>= 0) && isapprox(sum(q), 1.0; atol=1e-8)) ? 0.0 : -Inf

    logprior = logprior_mu + logprior_lambda + logprior_beta + logprior_q

    # ---- Likelihood ----
    loglik = 0.0
    for yi in data
        # each mixture component density
        comps = [q[k] * pdf(Normal(mu[k], 1 / sqrt(lambda[k])), yi) for k in 1:K]
        pyi = sum(comps)
        if pyi <= 0
            return Inf  # guard against underflow/invalid parameters
        end
        loglik += log(pyi)
    end

    # ---- Posterior ----
    logpost = logprior + loglik
    return -logpost  # negative log posterior
end

function aca_hidalgo()
    data = parse.(Float64, filter(!isempty, readlines("hidalgo_stamp_thicknesses.csv")))
    data *= 100
    neglogposterior(mu1, mu2, mu3, ll1, ll2, ll3, q1, q2, beta) = V([mu1, mu2, mu3, ll1, ll2, ll3, q1, q2, beta], data)

    mu1_dom = (6.0, 12.0)
    mu2_dom = (6.0, 12.0)
    mu3_dom = (6.0, 12.0)
    ll1_dom = (-2.0, 5.0)
    ll2_dom = (-2.0, 5.0)
    ll3_dom = (-2.0, 5.0)
    q1_dom = (0.0, 0.6)
    q2_dom = (0.0, 0.6)
    beta_dom = (1.0, 4.0)

    dom = (mu1_dom, mu2_dom, mu3_dom, ll1_dom, ll2_dom, ll3_dom, q1_dom, q2_dom, beta_dom)

    if mpi_rank == 0
        println("Starting MC integration...")
        flush(stdout)
    end

    result = estimate_log_evidence_uniform(neglogposterior; domain=dom, comm=mpi_comm, nsamples=n_samples)

    if mpi_rank == 0
        println(result)
    end

    # # cov0 = undef
    # # open("hidalgo0cov.txt", "r") do file
    # #     cov0 = eval(Meta.parse(readline(file)))
    # # end

    # mu, cov = mcmc_mean_cov_parallel(neglogposterior; domain=dom, comm=mpi_comm, nchains=n_chains, nsamples=n_samples, proposal_std=jump_width, periodicity=Tuple(fill(false, 8)))
    # if mpi_rank == 0
    #     println(mu)
    #     display(cov)
    #     # println(LinearAlgebra.norm(cov - cov0) / LinearAlgebra.norm(cov0))
    #     flush(stdout)
    # end
end

MPI.Init()
mpi_comm = MPI.COMM_WORLD
mpi_rank = MPI.Comm_rank(mpi_comm)
mpi_size = MPI.Comm_size(mpi_comm)

n_chains = 20
n_samples = 10^8
jump_width = 0.01

for _ in 1:20
    start_time = time()
    aca_hidalgo()
    end_time = time()
    elapsed_time = end_time - start_time
    if mpi_rank == 0
        println("Elapsed time: $elapsed_time seconds")
        flush(stdout)
    end
end

MPI.Finalize()

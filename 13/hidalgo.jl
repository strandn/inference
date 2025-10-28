include("tt_aca.jl")

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

    F = ResFunc(neglogposterior, (mu1_dom, mu2_dom, mu3_dom, ll1_dom, ll2_dom, ll3_dom, q1_dom, q2_dom, beta_dom), cutoff, Tuple(fill(false, d)))

    if mpi_rank == 0
        println("Starting TT-cross ACA...")
    end

    IJ = continuous_aca(F, fill(maxr, d - 1), n_chains, n_samples, jump_width, mpi_comm)

    if mpi_rank == 0
        open("hidalgo_IJ.txt", "w") do file
            write(file, "$IJ\n")
            write(file, "$(F.offset)\n")
        end
    end
end

MPI.Init()
mpi_comm = MPI.COMM_WORLD
mpi_rank = MPI.Comm_rank(mpi_comm)
mpi_size = MPI.Comm_size(mpi_comm)

d = 9
maxr = 10
n_chains = 20
n_samples = 1000
jump_width = 0.01
cutoff = 0.001

start_time = time()
aca_hidalgo()
end_time = time()
elapsed_time = end_time - start_time
if mpi_rank == 0
    println("Elapsed time: $elapsed_time seconds")
end

MPI.Finalize()

using DifferentialEquations

include("bayesian_vanilla.jl")

function lv!(du, u, p, t)
    x, y = u
    a, b, c, d = p
    du[1] = a * x - b * x * y
    du[2] = -c * y + d * x * y
end

function V(r, tspan, nsteps, data_hare, data_lynx, mu, sigma)
    dt = (tspan[2] - tspan[1]) / nsteps
    x0 = r[1]
    y0 = r[2]
    a = r[3]
    b = r[4]
    c = r[5]
    d = r[6]
    prob = ODEProblem(lv!, [x0, y0], tspan, [a, b, c, d])
    obs_hare = undef
    obs_lynx = undef
    try
        sol = solve(prob, Tsit5(), saveat=dt)
        if sol.retcode == ReturnCode.Success
            obs_hare = sol[1, :]
            obs_lynx = sol[2, :]
        else
            throw(ErrorException("ODE solver failed"))
        end
    catch e
        obs_hare = fill(Inf, nsteps + 1)
        obs_lynx = fill(Inf, nsteps + 1)
    end

    s2 = 100.0
    diff = [x0, y0, a, b, c, d] - mu
    result = 1 / 2 * sum((diff .^ 2) ./ sigma)
    for i in 1:nsteps+1
        result += log(2 * pi * s2) + (data_hare[i] - obs_hare[i]) ^ 2 / (2 * s2) + (data_lynx[i] - obs_lynx[i]) ^ 2 / (2 * s2)
    end
    return result
end

function aca_lv()
    tspan = (1900.0, 1920.0)
    nsteps = 20

    data_hare = [30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4, 27.1, 40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7]
    data_lynx = [4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4, 8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6]

    mu = [35.0, 4.0, 0.5, 0.02, 1.0, 0.03]
    sigma = [1.0, 1.0, 0.01, 1.0e-4, 0.01, 1.0e-4]
    neglogposterior(x0, y0, a, b, c, d) = V([x0, y0, a, b, c, d], tspan, nsteps, data_hare, data_lynx, mu, sigma)

    x0_dom = (20.0, 50.0)
    y0_dom = (0.0, 9.0)
    a_dom = (0.2, 0.7)
    b_dom = (0.01, 0.05)
    c_dom = (0.4, 1.4)
    d_dom = (0.01, 0.05)

    dom = (x0_dom, y0_dom, a_dom, b_dom, c_dom, d_dom)

    if mpi_rank == 0
        println("Starting MC integration...")
        flush(stdout)
    end

    result = estimate_log_evidence_uniform(neglogposterior; domain=dom, comm=mpi_comm, nsamples=n_samples)

    if mpi_rank == 0
        println(result)
    end

    # # cov0 = undef
    # # open("lv0cov.txt", "r") do file
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

for _ in 1:19
    start_time = time()
    aca_lv()
    end_time = time()
    elapsed_time = end_time - start_time
    if mpi_rank == 0
        println("Elapsed time: $elapsed_time seconds")
        flush(stdout)
    end
end

MPI.Finalize()

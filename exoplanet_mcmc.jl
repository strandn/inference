include("bayesian_mcmc.jl")

function radialvelocity(v0, K, φ0, lnP, t)
    Ω = 2 * pi / exp(lnP)
    return v0 + K * cos(Ω * t + φ0)
end

function V(r, tspan, nsteps, data, mu, sigma)
    tlist = LinRange(tspan..., nsteps + 1)
    v0 = r[1]
    K = r[2]
    φ0 = r[3]
    lnP = r[4]
    obs = []
    for t in tlist
        push!(obs, radialvelocity(v0, K, φ0, lnP, t))
    end

    s2 = 3.24
    diff = [v0, K] - mu
    result = 1 / 2 * sum((diff .^ 2) ./ sigma)
    for i in 1:nsteps+1
        result += 1 / 2 * log(2 * pi * s2) + (data[i] - obs[i]) ^ 2 / (2 * s2)
    end
    return result
end

function aca_exoplanet()
    tspan = (0.0, 200.0)
    nsteps = 6

    data = []
    open("exoplanet_data.txt", "r") do file
        for line in eachline(file)
            cols = split(line)
            push!(data, parse(Float64, cols[2]))
        end
    end

    mu = [0.0, 5.0]
    sigma = [1.0, 9.0]
    neglogposterior(x0, K, φ0, lnP) = V([x0, K, φ0, lnP], tspan, nsteps, data, mu, sigma)

    v0_dom = (-5.0, 5.0)
    K_dom = (0.5, 20.0)
    φ0_dom = (0.0, 2 * pi)
    lnP_dom = (3.0, 5.0)

    dom = (v0_dom, K_dom, φ0_dom, lnP_dom)

    if mpi_rank == 0
        println("Starting vanilla MCMC...")
    end

    result = estimate_log_evidence_TI(neglogposterior; domain=dom, comm=mpi_comm, nsamples=n_samples, burnin=burnin, proposal_std=jump_width, nbetas=nbetas)

    if mpi_rank == 0
        println(result)
    end
end

MPI.Init()
mpi_comm = MPI.COMM_WORLD
mpi_rank = MPI.Comm_rank(mpi_comm)
mpi_size = MPI.Comm_size(mpi_comm)

n_samples = 10^4
burnin = 1000
jump_width = 0.01
nbetas = 100

aca_exoplanet()

MPI.Finalize()

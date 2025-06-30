using LinearAlgebra

include("tt_aca.jl")

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
    diff = [v0, K, φ0, lnP] - mu
    result = 1 / 2 * sum((diff .^ 2) ./ sigma)
    for i in 1:nsteps+1
        result += 1 / 2 * log(2 * pi * s2) + (data[i] - obs[i]) ^ 2 / (2 * s2)
    end
    return result
end

function aca_exoplanet()
    if mpi_rank == 0
        println("Generating data...")
    end

    tspan = (0.0, 200.0)
    nsteps = 6
    dt = (tspan[2] - tspan[1]) / nsteps
    tlist = LinRange(tspan..., nsteps + 1)
    v0_true = 1.0
    K_true = 10.0
    φ0_true = 5.0
    lnP_true = 4.2

    data = zeros(nsteps + 1)

    if mpi_rank == 0
        for i in 1:nsteps+1
            t = (i - 1) * dt
            data[i] = radialvelocity(v0_true, K_true, φ0_true, lnP_true, t)
        end
        data += sqrt(3.24) * randn(nsteps + 1)
    end

    MPI.Bcast!(data, 0, mpi_comm)

    mu = [0.0, 5.0, 3.0, 4.0]
    sigma = [1.0, 9.0, 2.25, 0.25]
    neglogposterior(x0, K, φ0, lnP) = V([x0, K, φ0, lnP], tspan, nsteps, data, mu, sigma)

    if mpi_rank == 0
        open("exoplanet_data.txt", "w") do file
            for i in 1:nsteps+1
                write(file, "$(tlist[i]) $(data[i])\n")
            end
        end
    end

    if mpi_rank == 0
        println("Computing true density...")
    end

    v0_dom = (-3.0, 3.0)
    K_dom = (0.5, 14.0)
    φ0_dom = (0.0, 2 * pi)
    lnP_dom = (3.0, 5.0)

    F = ResFunc(neglogposterior, (v0_dom, K_dom, φ0_dom, lnP_dom), cutoff, mu, sigma)

    if mpi_rank == 0
        println("Starting TT-cross ACA...")
    end

    IJ = continuous_aca(F, fill(maxr, d - 1), n_chains, n_samples, jump_width, mpi_comm)

    norm = 0.0
    normbuf = [0.0]

    if mpi_rank == 0
        open("exoplanet_IJ.txt", "w") do file
            write(file, "$IJ\n")
            write(file, "$(F.offset)\n")
        end
        norm = compute_norm(F)
        normbuf = [norm]
        println("norm = $norm")
    end

    MPI.Bcast!(normbuf, 0, mpi_comm)
    norm = normbuf[]
end

MPI.Init()
mpi_comm = MPI.COMM_WORLD
mpi_rank = MPI.Comm_rank(mpi_comm)
mpi_size = MPI.Comm_size(mpi_comm)

d = 4
maxr = 50
n_chains = 40
n_samples = 400
jump_width = 0.01
cutoff = 1.0e-3

start_time = time()
aca_exoplanet()
end_time = time()
elapsed_time = end_time - start_time
if mpi_rank == 0
    println("Elapsed time: $elapsed_time seconds")
end

MPI.Finalize()

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

    F = ResFunc(neglogposterior, (v0_dom, K_dom, φ0_dom, lnP_dom), cutoff)

    if mpi_rank == 0
        println("Starting TT-cross ACA...")
    end

    start_time = time()
    IJ = continuous_aca(F, fill(maxr, d - 1), n_chains, n_samples, jump_width, mpi_comm)
    end_time = time()
    elapsed_time = end_time - start_time
    if mpi_rank == 0
        println("Elapsed time: $elapsed_time seconds")
    end

    norm = 0.0
    if mpi_rank == 0
        open("exoplanet_IJ.txt", "w") do file
            write(file, "$IJ\n")
            write(file, "$(F.offset)\n")
        end
        norm = compute_norm(F)
        println("norm = $norm")
        println(F.offset - log(norm))
    end
end

MPI.Init()
mpi_comm = MPI.COMM_WORLD
mpi_rank = MPI.Comm_rank(mpi_comm)
mpi_size = MPI.Comm_size(mpi_comm)

d = 4
maxr = 50
n_chains = 10
n_samples = 500
jump_width = 0.01
cutoff = 0.01

aca_exoplanet()

MPI.Finalize()

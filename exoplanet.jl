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

    F = ResFunc(neglogposterior, (v0_dom, K_dom, φ0_dom, lnP_dom), cutoff, (false, false, true, false))

    if mpi_rank == 0
        println("Starting TT-cross ACA...")
    end

    IJ = continuous_aca(F, fill(maxr, d - 1), n_chains, n_samples, jump_width, mpi_comm)

    cov0 = undef
    open("exoplanet0cov.txt", "r") do file
        cov0 = eval(Meta.parse(readline(file)))
    end

    if mpi_rank == 0
        open("stamps_IJ.txt", "w") do file
            write(file, "$IJ\n")
            write(file, "$(F.offset)\n")
        end
        norm, integrals, skeleton, links = compute_norm(F)
        println("norm = $norm")
        println(F.offset - log(norm))
        
        mulist = zeros(d)
        for i in 1:d
            mulist[i] = compute_mu(F, integrals, skeleton, links, i) / norm
        end
        println(mulist)
        cov = zeros(d, d)
        for i in 1:d
            for j in i:d
                cov[i, j] = cov[j, i] = compute_cov(F, integrals, skeleton, links, mulist, i, j) / norm
            end
        end
        display(cov)
        println(LinearAlgebra.norm(cov - cov0) / LinearAlgebra.norm(cov0))
        flush(stdout)
    end
end

MPI.Init()
mpi_comm = MPI.COMM_WORLD
mpi_rank = MPI.Comm_rank(mpi_comm)
mpi_size = MPI.Comm_size(mpi_comm)

d = 4
maxr = 5
n_chains = 20
n_samples = 100
jump_width = 0.01
cutoff = 1.0e-4

start_time = time()
aca_exoplanet()
end_time = time()
elapsed_time = end_time - start_time
if mpi_rank == 0
    println("Elapsed time: $elapsed_time seconds")
end

MPI.Finalize()

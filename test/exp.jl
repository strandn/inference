using DifferentialEquations
using LinearAlgebra

include("tt_aca.jl")

function decay!(du, u, p, t)
    λ = p[1]
    du[1] = -λ * u[1]
end

function V(r, tspan, nsteps, data, mu, sigma)
    dt = (tspan[2] - tspan[1]) / nsteps
    x0 = r[1]
    λ = r[2]
    prob = ODEProblem(decay!, [x0], tspan, [λ])

    obs = undef
    try
        sol = solve(prob, Tsit5(), saveat=dt)
        if sol.retcode == ReturnCode.Success
            obs = sol[1, :]
        else
            throw(ErrorException("ODE solver failed"))
        end
    catch e
        obs = fill(Inf, nsteps + 1)
    end

    s2 = 0.1
    diff = [x0, λ] - mu
    result = 1 / 2 * sum((diff .^ 2) ./ sigma)
    for i in 1:nsteps+1
        result += 1 / 2 * log(2 * pi * s2) + (data[i] - obs[i]) ^ 2 / (2 * s2)
    end
    return result
end

function aca_exp()
    if mpi_rank == 0
        println("Generating data...")
    end

    tspan = (0.0, 10.0)
    nsteps = 50
    dt = (tspan[2] - tspan[1]) / nsteps
    tlist = LinRange(tspan..., nsteps + 1)
    x0_true = 7.5
    λ_true = 0.5

    truedata = zeros(nsteps + 1)
    data = zeros(nsteps + 1)

    if mpi_rank == 0
        prob = ODEProblem(decay!, [x0_true], tspan, [λ_true])
        sol = solve(prob, Tsit5(), saveat=dt)

        truedata = sol[1, :]
        data = deepcopy(truedata)
        data += sqrt(0.1) * randn(length(data))
    end

    MPI.Bcast!(truedata, 0, mpi_comm)
    MPI.Bcast!(data, 0, mpi_comm)

    mu = [7.5, 0.5]
    sigma = [1.0, 0.04]
    neglogposterior(x0, λ) = V([x0, λ], tspan, nsteps, data, mu, sigma)

    if mpi_rank == 0
        open("exp_data.txt", "w") do file
            for i in 1:nsteps+1
                write(file, "$(tlist[i]) $(truedata[i]) $(data[i])\n")
            end
        end
    end

    x0_dom = (5.0, 10.0)
    λ_dom = (0.1, 1.0)

    if mpi_rank == 0
        println("Computing true density...")
    end
    
    nbins = 100
    x0_vals = LinRange(x0_dom..., nbins + 1)
    λ_vals = LinRange(λ_dom..., nbins + 1)

    local_n = div(nbins ^ 2, mpi_size)
    local_start = mpi_rank * local_n + 1

    local_dens = zeros(local_n)

    for local_ij in 1:local_n
        ij = local_start + local_ij - 1
        x = x0_vals[div(ij - 1, nbins) + 1]
        y = λ_vals[rem(ij - 1, nbins) + 1]
        local_dens[local_ij] = exp(-neglogposterior(x, y))
    end

    global_dens = MPI.Gather(local_dens, 0, mpi_comm)

    if mpi_rank == 0
        dens = vcat([global_dens; zeros(rem(nbins ^ 2, mpi_size))]...)
        for ij in mpi_size*local_n+1:nbins^2
            x = x0_vals[div(ij - 1, nbins) + 1]
            y = λ_vals[rem(ij - 1, nbins) + 1]
            dens[ij] = exp(-neglogposterior(x, y))
        end

        open("exp_density_true.txt", "w") do file
            for ij in 1:nbins^2
                i = div(ij - 1, nbins) + 1
                j = rem(ij - 1, nbins) + 1
                write(file, "$(x0_vals[i]) $(λ_vals[j]) $(dens[ij])\n")
            end
        end
    end

    F = ResFunc(neglogposterior, (x0_dom, λ_dom), cutoff, mu, sigma)

    if mpi_rank == 0
        println("Starting TT-cross ACA...")
    end

    IJ = continuous_aca(F, fill(maxr, d - 1), n_chains, n_samples, jump_width, mpi_comm)

    norm = 0.0
    normbuf = [0.0]

    if mpi_rank == 0
        open("exp_IJ.txt", "w") do file
            write(file, "$IJ\n")
            write(file, "$(F.offset)\n")
        end
        norm = compute_norm(F)
        normbuf = [norm]
        println("norm = $norm")
    end

    MPI.Bcast!(normbuf, 0, mpi_comm)
    norm = normbuf[]

    local_dens = zeros(local_n)

    for local_ij in 1:local_n
        ij = local_start + local_ij - 1
        x = x0_vals[div(ij - 1, nbins) + 1]
        y = λ_vals[rem(ij - 1, nbins) + 1]
        local_dens[local_ij] = compute_12(F, x, y) / norm
    end

    global_dens = MPI.Gather(local_dens, 0, mpi_comm)

    if mpi_rank == 0
        dens = vcat([global_dens; zeros(rem(nbins ^ 2, mpi_size))]...)
        for ij in mpi_size*local_n+1:nbins^2
            x = x0_vals[div(ij - 1, nbins) + 1]
            y = λ_vals[rem(ij - 1, nbins) + 1]
            dens[ij] = compute_12(F, x, y) / norm
        end

        open("exp_density_aca.txt", "w") do file
            for ij in 1:nbins^2
                i = div(ij - 1, nbins) + 1
                j = rem(ij - 1, nbins) + 1
                write(file, "$(x0_vals[i]) $(λ_vals[j]) $(dens[ij])\n")
            end
        end
    end
end

MPI.Init()
mpi_comm = MPI.COMM_WORLD
mpi_rank = MPI.Comm_rank(mpi_comm)
mpi_size = MPI.Comm_size(mpi_comm)

d = 2
maxr = 50
n_chains = 1
n_samples = 1000
jump_width = 0.01
cutoff = 1.0e-3

start_time = time()
aca_exp()
end_time = time()
elapsed_time = end_time - start_time
if mpi_rank == 0
    println("Elapsed time: $elapsed_time seconds")
end

MPI.Finalize()

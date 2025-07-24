using DifferentialEquations
using LinearAlgebra

include("tt_aca.jl")

function gts!(du, u, p, t)
    x, y = u
    α1, α2, β, γ = p
    du[1] = α1 / (1 + y ^ β) - x
    du[2] = α2 / (1 + x ^ γ) - y
end

function V(r, tspan, nsteps, data_x, data_y, mu, sigma)
    dt = (tspan[2] - tspan[1]) / nsteps
    x0 = r[1]
    y0 = r[2]
    α1 = r[3]
    α2 = r[4]
    β = r[5]
    γ = r[6]
    prob = ODEProblem(gts!, [x0, y0], tspan, [α1, α2, β, γ])

    obs_x = undef
    obs_y = undef
    try
        sol = solve(prob, Tsit5(), saveat=dt)
        if sol.retcode == ReturnCode.Success
            obs_x = sol[1, :]
            obs_y = sol[2, :]
        else
            throw(ErrorException("ODE solver failed"))
        end
    catch e
        obs_x = fill(Inf, nsteps + 1)
        obs_y = fill(Inf, nsteps + 1)
    end

    s2 = 0.25
    diff = [x0, y0, α1, α2, β, γ] - mu
    result = 1 / 2 * sum((diff .^ 2) ./ sigma)
    for i in eachindex(data_x)
        result += log(2 * pi * s2) + (data_x[i] - obs_x[i]) ^ 2 / (2 * s2) + (data_y[i] - obs_y[i]) ^ 2 / (2 * s2)
    end
    return result
end

function aca_gts()
    if mpi_rank == 0
        println("Generating data...")
    end

    tspan = (0.0, 30.0)
    nsteps = 25
    dt = (tspan[2] - tspan[1]) / nsteps
    tlist = LinRange(tspan..., nsteps + 1)
    x0_true = 0.1
    y0_true = 5.0
    α1_true = 36.0
    α2_true = 34.0
    β_true = 2.0
    γ_true = 2.0

    truedata_x = zeros(nsteps + 1)
    truedata_y = zeros(nsteps + 1)
    data_x = zeros(nsteps + 1)
    data_y = zeros(nsteps + 1)

    if mpi_rank == 0
        prob = ODEProblem(gts!, [x0_true, y0_true], tspan, [α1_true, α2_true, β_true, γ_true])
        sol = solve(prob, Tsit5(), saveat=dt)

        truedata_x = sol[1, :]
        truedata_y = sol[2, :]
        data_x = deepcopy(truedata_x)
        data_y = deepcopy(truedata_y)
        data_x += sqrt(0.25) * randn(length(data_x))
        data_y += sqrt(0.25) * randn(length(data_y))
    end

    MPI.Bcast!(truedata_x, 0, mpi_comm)
    MPI.Bcast!(truedata_y, 0, mpi_comm)
    MPI.Bcast!(data_x, 0, mpi_comm)
    MPI.Bcast!(data_y, 0, mpi_comm)

    mu = [5.0, 5.0, 35.0, 35.0, 2.0, 2.0]
    sigma = [25.0, 25.0, 400.0, 400.0, 4.0, 4.0]
    neglogposterior(x0, y0, α1, α2, β, γ) = V([x0, y0, α1, α2, β, γ], tspan, nsteps, data_x, data_y, mu, sigma)

    if mpi_rank == 0
        open("gts_data.txt", "w") do file
            for i in 1:nsteps+1
                write(file, "$(tlist[i]) $(truedata_x[i]) $(truedata_y[i]) $(data_x[i]) $(data_y[i])\n")
            end
        end
    end

    if mpi_rank == 0
        println("Computing true density...")
    end

    x0_dom = (0.0, 10.0)
    y0_dom = (0.0, 10.0)
    α1_dom = (5.0, 100.0)
    α2_dom = (5.0, 100.0)
    β_dom = (1.0, 5.0)
    γ_dom = (1.0, 5.0)

    F = ResFunc(neglogposterior, (x0_dom, y0_dom, α1_dom, α2_dom, β_dom, γ_dom), cutoff, mu, sigma)

    if mpi_rank == 0
        println("Starting TT-cross ACA...")
    end

    IJ = continuous_aca(F, fill(maxr, d - 1), n_chains, n_samples, jump_width, mpi_comm)

    norm = 0.0
    normbuf = [0.0]

    if mpi_rank == 0
        open("gts_IJ.txt", "w") do file
            write(file, "$IJ\n")
            write(file, "$(F.offset)\n")
        end
        norm = compute_norm(F)
        println("norm = $norm")
    end
end

MPI.Init()
mpi_comm = MPI.COMM_WORLD
mpi_rank = MPI.Comm_rank(mpi_comm)
mpi_size = MPI.Comm_size(mpi_comm)

d = 6
maxr = 50
n_chains = 40
n_samples = 10 ^ 4
jump_width = 0.001
cutoff = 1.0e-3

start_time = time()
aca_gts()
end_time = time()
elapsed_time = end_time - start_time
if mpi_rank == 0
    println("Elapsed time: $elapsed_time seconds")
end

MPI.Finalize()

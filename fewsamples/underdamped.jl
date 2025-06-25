using DifferentialEquations
using LinearAlgebra

include("tt_aca.jl")

function damped_oscillator!(du, u, p, t)
    x, v = u
    ω, γ = p
    du[1] = v
    du[2] = -ω^2 * x - γ * v
end

function V(r, tspan, dt, data_x, data_v)
    x0 = r[1]
    v0 = r[2]
    ω = r[3]
    γ = r[4]
    prob = ODEProblem(damped_oscillator!, [x0, v0], tspan, [ω, γ])
    sol = solve(prob, Tsit5(), saveat=dt)
    obs_x = sol[1, :]
    obs_v = sol[2, :]

    s2 = 0.15
    mu = [5.0, 5.0, 1.0, 2.0]
    sigma = zeros(4, 4)
    sigma[1, 1] = sigma[2, 2] = 1.0
    sigma[3, 3] = 0.1
    sigma[4, 4] = 2.0
    diff = [x0, v0, ω, γ] - mu
    result = 1 / 2 * dot(diff, inv(sigma) * diff)
    for i in eachindex(data_x)
        result += log(2 * pi * s2) + (data_x[i] - obs_x[i]) ^ 2 / (2 * s2) + (data_v[i] - obs_v[i]) ^ 2 / (2 * s2)
    end
    return result
end

function aca_damped()
    if mpi_rank == 0
        println("Generating data...")
    end

    tspan = (0.0, 20.0)
    nsteps = 50
    dt = (tspan[2] - tspan[1]) / nsteps
    tlist = LinRange(tspan..., nsteps + 1)
    x0_true = 7.5
    v0_true = 2.5
    ω_true = 1.0
    γ_true = 0.4

    truedata_x = zeros(nsteps + 1)
    truedata_v = zeros(nsteps + 1)
    data_x = zeros(nsteps + 1)
    data_v = zeros(nsteps + 1)

    if mpi_rank == 0
        prob = ODEProblem(damped_oscillator!, [x0_true, v0_true], tspan, [ω_true, γ_true])
        sol = solve(prob, Tsit5(), saveat=dt)

        truedata_x = sol[1, :]
        truedata_v = sol[2, :]
        data_x = deepcopy(truedata_x)
        data_v = deepcopy(truedata_v)
        data_x += sqrt(0.15) * randn(length(data_x))
        data_v += sqrt(0.15) * randn(length(data_v))
    end

    MPI.Bcast!(truedata_x, 0, mpi_comm)
    MPI.Bcast!(truedata_v, 0, mpi_comm)
    MPI.Bcast!(data_x, 0, mpi_comm)
    MPI.Bcast!(data_v, 0, mpi_comm)

    neglogposterior(x0, v0, ω, γ) = V([x0, v0, ω, γ], tspan, dt, data_x, data_v)

    if mpi_rank == 0
        open("underdamped_data.txt", "w") do file
            for i in 1:nsteps+1
                write(file, "$(tlist[i]) $(truedata_x[i]) $(truedata_v[i]) $(data_x[i]) $(data_v[i])\n")
            end
        end
    end

    if mpi_rank == 0
        println("Computing true density...")
    end

    x0_dom = (2.5, 12.5)
    v0_dom = (0.5, 5.0)
    ω_dom = (0.5, 2.0)
    γ_dom = (0.1, 7.5)

    F = ResFunc(neglogposterior, (x0_dom, v0_dom, ω_dom, γ_dom), cutoff)

    if mpi_rank == 0
        println("Starting TT-cross ACA...")
    end

    IJ = continuous_aca(F, fill(maxr, d - 1), n_chains, n_samples, jump_width, mpi_comm)

    norm = 0.0
    normbuf = [0.0]

    if mpi_rank == 0
        open("underdamped_IJ.txt", "w") do file
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
n_chains = 48
n_samples = 50
jump_width = 0.1
cutoff = 1.0e-3

start_time = time()
aca_damped()
end_time = time()
elapsed_time = end_time - start_time
if mpi_rank == 0
    println("Elapsed time: $elapsed_time seconds")
end

MPI.Finalize()

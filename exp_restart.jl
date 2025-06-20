using DifferentialEquations
using LinearAlgebra

include("tt_aca.jl")

function decay!(du, u, p, t)
    λ = p[1]
    du[1] = -λ * u[1]
end

function V(r, tspan, dt, data)
    x0 = r[1]
    λ = r[2]
    prob = ODEProblem(decay!, [x0], tspan, [λ])
    sol = solve(prob, Tsit5(), saveat=dt)
    obs = sol[1, :]

    s2 = 0.1
    mu = [7.5, 0.5]
    sigma = zeros(2, 2)
    sigma[1, 1] = 1.0
    sigma[2, 2] = 0.04
    diff = [x0, λ] - mu
    result = 1 / 2 * dot(diff, inv(sigma) * diff)
    for i in eachindex(data)
        result += 1 / 2 * log(2 * pi * s2) + (data[i] - obs[i]) ^ 2 / (2 * s2)
    end
    return result
end

function aca_exp()
    F = ResFunc(posterior, (x0_dom, λ_dom), cutoff)

    F.I, F.J = ([[Float64[]], [[7.624630710003532], [7.452575277253249], [7.809898861921559], [7.3198778503006565], [7.951530384483066], [7.6972670020390535], [7.179410261654521], [8.10889391020572]]], [[Float64[]], [[0.5122457164564077], [0.4940020831714453], [0.5315903582618265], [0.4802146772761345], [0.5455731301572813], [0.5042776126156895], [0.46663661695798025], [0.5596315436060152]]])

    norm = 0.0
    normbuf = [0.0]

    if mpi_rank == 0
        println(IJ)
        norm = compute_norm(F)
        normbuf = [norm]
        println("norm = $norm")
    end

    MPI.Bcast!(normbuf, 0, mpi_comm)
    norm = normbuf[]

    if mpi_rank == 0
        println("Collecting sample 1...")
        sample = sample_from_tt(F, norm)
        println(sample)
        println("Collecting sample 2...")
        sample = sample_from_tt(F, norm)
        println(sample)
        println("Collecting sample 3...")
        sample = sample_from_tt(F, norm)
        println(sample)
    end
end

MPI.Init()
mpi_comm = MPI.COMM_WORLD
mpi_rank = MPI.Comm_rank(mpi_comm)
mpi_size = MPI.Comm_size(mpi_comm)

d = 2
maxr = 50
n_chains = 50
n_samples = 100
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

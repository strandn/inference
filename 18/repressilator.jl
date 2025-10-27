using DifferentialEquations

include("tt_aca.jl")

function repressilator!(du, u, p, t)
    X1, X2, X3 = u
    α1, α2, α3, m, η = p
    du[1] = α1 / (1 + X2 ^ m) - η * X1
    du[2] = α2 / (1 + X3 ^ m) - η * X2
    du[3] = α3 / (1 + X1 ^ m) - η * X3
end

function V(r, tspan, nsteps, data, mu, sigma)
    dt = (tspan[2] - tspan[1]) / nsteps
    α1 = r[1]
    α2 = r[2]
    α3 = r[3]
    m = r[4]
    η = r[5]
    X10 = r[6]
    X20 = r[7]
    X30 = r[8]
    prob = ODEProblem(repressilator!, [X10, X20, X30], tspan, [α1, α2, α3, m, η])
    obs = undef
    try
        sol = solve(prob, Tsit5(), saveat=dt)
        if sol.retcode == ReturnCode.Success
            obs = sol[1, :] + sol[2, :] + sol[3, :]
        else
            throw(ErrorException("ODE solver failed"))
        end
    catch e
        obs = fill(Inf, nsteps + 1)
    end

    s2 = 0.25
    diff = [α1, α2, α3, m, η, X10, X20, X30] - mu
    result = 1 / 2 * sum((diff .^ 2) ./ sigma)
    for i in 1:nsteps+1
        result += 1 / 2 * log(2 * pi * s2) + (data[i] - obs[i]) ^ 2 / (2 * s2)
    end
    return result
end

function aca_repressilator()
    tspan = (0.0, 30.0)
    nsteps = 50

    data = []
    open("repressilator_data.txt", "r") do file
        for line in eachline(file)
            cols = split(line)
            push!(data, parse(Float64, cols[6]))
        end
    end

    mu = [15.0, 15.0, 15.0, 5.0, 5.0, 2.0, 2.0, 2.0]
    sigma = [25.0, 25.0, 25.0, 25.0, 25.0, 4.0, 4.0, 4.0]
    neglogposterior(α1, α2, α3, m, η, X10, X20, X30) = V([α1, α2, α3, m, η, X10, X20, X30], tspan, nsteps, data, mu, sigma)

    α1_dom = (0.5, 25.0)
    α2_dom = (0.5, 25.0)
    α3_dom = (0.5, 25.0)
    m_dom = (3.0, 5.0)
    η_dom = (0.95, 1.05)
    X10_dom = (0.5, 3.5)
    X20_dom = (0.5, 3.5)
    X30_dom = (0.5, 3.5)

    F = ResFunc(neglogposterior, (α1_dom, α2_dom, α3_dom, m_dom, η_dom, X10_dom, X20_dom, X30_dom), cutoff, Tuple(fill(false, d)))

    if mpi_rank == 0
        println("Starting TT-cross ACA...")
    end

    IJ = continuous_aca(F, fill(maxr, d - 1), n_chains, n_samples, jump_width, mpi_comm)

    if mpi_rank == 0
        open("repressilator_IJ.txt", "w") do file
            write(file, "$IJ\n")
            write(file, "$(F.offset)\n")
        end
    end
end

MPI.Init()
mpi_comm = MPI.COMM_WORLD
mpi_rank = MPI.Comm_rank(mpi_comm)
mpi_size = MPI.Comm_size(mpi_comm)

d = 8
maxr = 10
n_chains = 50
n_samples = 200
jump_width = 0.01
cutoff = 0.001

start_time = time()
aca_repressilator()
end_time = time()
elapsed_time = end_time - start_time
if mpi_rank == 0
    println("Elapsed time: $elapsed_time seconds")
end

MPI.Finalize()

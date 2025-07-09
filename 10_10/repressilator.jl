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
    X10 = r[1]
    X20 = r[2]
    X30 = r[3]
    α1 = r[4]
    α2 = r[5]
    α3 = r[6]
    m = r[7]
    η = r[8]
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
        obs = fill(200.0, nsteps + 1)
    end

    s2 = 0.25
    diff = [X10, X20, X30, α1, α2, α3, m, η] - mu
    result = 1 / 2 * sum((diff .^ 2) ./ sigma)
    for i in 1:nsteps+1
        result += 1 / 2 * log(2 * pi * s2) + (data[i] - obs[i]) ^ 2 / (2 * s2)
    end
    return result
end

function aca_repressilator()
    if mpi_rank == 0
        println("Generating data...")
    end

    tspan = (0.0, 30.0)
    nsteps = 50
    dt = (tspan[2] - tspan[1]) / nsteps
    tlist = LinRange(tspan..., nsteps + 1)
    X10_true = X20_true = X30_true = 2.0
    α1_true = 10.0
    α2_true = 15.0
    α3_true = 20.0
    m_true = 4.0
    η_true = 1.0

    truedata1 = zeros(nsteps + 1)
    truedata2 = zeros(nsteps + 1)
    truedata3 = zeros(nsteps + 1)
    truedata = zeros(nsteps + 1)
    data = zeros(nsteps + 1)

    if mpi_rank == 0
        prob = ODEProblem(repressilator!, [X10_true, X20_true, X30_true], tspan, [α1_true, α2_true, α3_true, m_true, η_true])
        sol = solve(prob, Tsit5(), saveat=dt)

        truedata1 = sol[1, :]
        truedata2 = sol[2, :]
        truedata3 = sol[3, :]
        truedata = truedata1 + truedata2 + truedata3
        data = deepcopy(truedata)
        data += sqrt(0.25) * randn(nsteps + 1)
    end

    MPI.Bcast!(truedata1, 0, mpi_comm)
    MPI.Bcast!(truedata2, 0, mpi_comm)
    MPI.Bcast!(truedata3, 0, mpi_comm)
    MPI.Bcast!(truedata, 0, mpi_comm)
    MPI.Bcast!(data, 0, mpi_comm)

    mu = [2.0, 2.0, 2.0, 15.0, 15.0, 15.0, 5.0, 5.0]
    sigma = [4.0, 4.0, 4.0, 25.0, 25.0, 25.0, 25.0, 25.0]
    neglogposterior(X10, X20, X30, α1, α2, α3, m, η) = V([X10, X20, X30, α1, α2, α3, m, η], tspan, nsteps, data, mu, sigma)

    if mpi_rank == 0
        open("repressilator_data.txt", "w") do file
            for i in 1:nsteps+1
                write(file, "$(tlist[i]) $(truedata1[i]) $(truedata2[i]) $(truedata3[i]) $(truedata[i]) $(data[i])\n")
            end
        end
    end

    X10_dom = (0.5, 3.5)
    X20_dom = (0.5, 3.5)
    X30_dom = (0.5, 3.5)
    α1_dom = (0.5, 25.0)
    α2_dom = (0.5, 25.0)
    α3_dom = (0.5, 25.0)
    m_dom = (3.0, 5.0)
    η_dom = (0.95, 1.05)

    F = ResFunc(neglogposterior, (X10_dom, X20_dom, X30_dom, α1_dom, α2_dom, α3_dom, m_dom, η_dom), cutoff, mu, sigma)

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
    normbuf = [0.0]

    if mpi_rank == 0
        open("repressilator_IJ.txt", "w") do file
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

d = 8
maxr = 50
n_chains = 40
n_samples = 10 ^ 4
jump_width = 0.001
cutoff = 0.001

aca_repressilator()

MPI.Finalize()

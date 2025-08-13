using DifferentialEquations

include("tt_aca.jl")

function lv!(du, u, p, t)
    x, y = u
    a, b, c, d = p
    du[1] = a * x - b * x * y
    du[2] = -c * y + d * x * y
end

function V(r, tspan, nsteps, data, mu, sigma)
    dt = (tspan[2] - tspan[1]) / nsteps
    x0 = r[1]
    y0 = r[2]
    a = r[3]
    b = r[4]
    c = r[5]
    d = r[6]
    prob = ODEProblem(lv!, [x0, y0], tspan, [a, b, c, d])
    obs = undef
    try
        sol = solve(prob, Tsit5(), saveat=dt)
        if sol.retcode == ReturnCode.Success
            obs = sol[1, :] + sol[2, :]
        else
            throw(ErrorException("ODE solver failed"))
        end
    catch e
        obs = fill(Inf, nsteps + 1)
    end

    s2 = 12.25
    diff = [X10, X20, X30, α1, α2, α3, m, η] - mu
    result = 1 / 2 * sum((diff .^ 2) ./ sigma)
    for i in 1:nsteps+1
        result += 1 / 2 * log(2 * pi * s2) + (data[i] - obs[i]) ^ 2 / (2 * s2)
    end
    return result
end

function aca_lv()
    tspan = (0.0, 50.0)
    nsteps = 30

    data = []
    open("lv_data.txt", "r") do file
        for line in eachline(file)
            cols = split(line)
            push!(data, parse(Float64, cols[5]))
        end
    end

    mu = [15.0, 15.0, 0.7, 0.05, 0.7, 0.05]
    sigma = [225.0, 225.0, 0.04, 0.0025, 0.04, 0.0025]
    neglogposterior(x0, y0, a, b, c, d) = V([x0, y0, a, b, c, d], tspan, nsteps, data, mu, sigma)

    x0_dom = (5.0, 50.0)
    y0_dom = (5.0, 50.0)
    a_dom = (0.1, 1.0)
    b_dom = (0.01, 0.15)
    c_dom = (0.1, 1.1)
    d_dom = (0.01, 0.15)

    F = ResFunc(neglogposterior, (x0_dom, y0_dom, a_dom, b_dom, c_dom, d_dom), cutoff, Tuple(fill(false, d)))

    if mpi_rank == 0
        println("Starting TT-cross ACA...")
    end

    IJ = continuous_aca(F, fill(maxr, d - 1), n_chains, n_samples, jump_width, mpi_comm)

    norm = 0.0
    if mpi_rank == 0
        open("lv_IJ.txt", "w") do file
            write(file, "$IJ\n")
            write(file, "$(F.offset)\n")
        end
        norm, _, _ = compute_norm(F)
        println("norm = $norm")
        println(F.offset - log(norm))
        flush(stdout)
    end
end

MPI.Init()
mpi_comm = MPI.COMM_WORLD
mpi_rank = MPI.Comm_rank(mpi_comm)
mpi_size = MPI.Comm_size(mpi_comm)

d = 6
maxr = 50
n_chains = 20
n_samples = 1000
jump_width = 0.01
cutoff = 1.0e-4

start_time = time()
aca_lv()
end_time = time()
elapsed_time = end_time - start_time
if mpi_rank == 0
    println("Elapsed time: $elapsed_time seconds")
end

MPI.Finalize()

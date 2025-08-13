using DifferentialEquations

include("tt_aca.jl")

function planetary!(du, u, p, t)
    # u = [q1, q2, p1, p2]
    q12 = u[1:2]          # position vector (q1, q2)
    p12 = u[3:4]          # momentum vector (p1, p2)
    
    m, k = p[1], p[2]   # We'll pass mass m and force constant k as parameters
    
    r3 = (q12[1]^2 + q12[2]^2)^(1.5)  # |q|^3
    
    du[1] = p12[1] / m    # dq1/dt = p1 / m
    du[2] = p12[2] / m    # dq2/dt = p2 / m
    
    du[3] = -k * q12[1] / r3  # dp1/dt = -k q1 / |q|^3
    du[4] = -k * q12[2] / r3  # dp2/dt = -k q2 / |q|^3
end

function V(r, tspan, nsteps, data_x, data_y, mu, sigma)
    dt = (tspan[2] - tspan[1]) / nsteps
    qx0 = r[1]
    qy0 = r[2]
    px0 = r[3]
    py0 = r[4]
    m = r[5]
    k = r[6]
    prob = ODEProblem(planetary!, [qx0, qy0, px0, py0], tspan, [m, k])
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

    s2 = 0.01
    diff = [qx0, qy0, px0, py0, m, k] - mu
    result = 1 / 2 * sum((diff .^ 2) ./ sigma)
    for i in 1:nsteps+1
        result += log(2 * pi * s2) + (data_x[i] - obs_x[i]) ^ 2 / (2 * s2) + (data_y[i] - obs_y[i]) ^ 2 / (2 * s2)
    end
    return result
end

function aca_planetary()
    tspan = (0.0, 15.0)
    nsteps = 20

    data1 = []
    data2 = []
    open("planetary_data.txt", "r") do file
        for line in eachline(file)
            cols = split(line)
            push!(data1, parse(Float64, cols[4]))
            push!(data2, parse(Float64, cols[5]))
        end
    end

    mu = [1.0, 1.0, 1.0, 1.0, 5.0, 5.0]
    sigma = [1.0, 1.0, 1.0, 1.0, 25.0, 25.0]
    neglogposterior(qx0, qy0, px0, py0, m, k) = V([qx0, qy0, px0, py0, m, k], tspan, nsteps, data1, data2, mu, sigma)

    qx0_dom = (-2.0, 4.0)
    qy0_dom = (-2.0, 4.0)
    px0_dom = (-2.0, 4.0)
    py0_dom = (-2.0, 4.0)
    m_dom = (0.1, 10.0)
    k_dom = (0.1, 10.0)

    F = ResFunc(neglogposterior, (qx0_dom, qy0_dom, px0_dom, py0_dom, m_dom, k_dom), cutoff, Tuple(fill(false, d)))

    if mpi_rank == 0
        println("Starting TT-cross ACA...")
    end

    IJ = continuous_aca(F, fill(maxr, d - 1), n_chains, n_samples, jump_width, mpi_comm)

    norm = 0.0
    if mpi_rank == 0
        open("planetary_IJ.txt", "w") do file
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
maxr = 5
n_chains = 100
n_samples = 5000
jump_width = 0.01
cutoff = 1.0e-4

start_time = time()
aca_planetary()
end_time = time()
elapsed_time = end_time - start_time
if mpi_rank == 0
    println("Elapsed time: $elapsed_time seconds")
end

MPI.Finalize()

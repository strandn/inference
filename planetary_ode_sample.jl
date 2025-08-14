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

    qx0_dom = (0.95, 1.1)
    qy0_dom = (0.1, 0.4)
    px0_dom = (-0.05, 0.7)
    py0_dom = (-0.4, 4.0)
    m_dom = (0.1, 4.5)
    k_dom = (0.1, 4.5)

    F = ResFunc(neglogposterior, (qx0_dom, qy0_dom, px0_dom, py0_dom, m_dom, k_dom), 0.0, Tuple(fill(false, d)))

    open("planetary_IJ.txt", "r") do file
        F.I, F.J = eval(Meta.parse(readline(file)))
        F.offset = parse(Float64, readline(file))
    end

    norm, integrals, skeleton, links = compute_norm(F)
    println("norm = $norm\n")
    println(F.offset - log(norm))
    println()
    flush(stdout)

    nbins = 100
    grid = (
        LinRange(qx0_dom..., nbins + 1),
        LinRange(qy0_dom..., nbins + 1),
        LinRange(px0_dom..., nbins + 1),
        LinRange(py0_dom..., nbins + 1),
        LinRange(m_dom..., nbins + 1),
        LinRange(k_dom..., nbins + 1)
    )

    for count in 1:d-1
        dens = compute_marginal(F, integrals, skeleton, links, count)
        open("planetary_marginal_$count.txt", "w") do file
            for i in 1:nbins
                for j in 1:nbins
                    write(file, "$(grid[count][i]) $(grid[count + 1][j]) $(dens[i, j] / norm)\n")
                end
            end
        end
    end

    open("planetary_samples.txt", "w") do file
        for i in 1:30
            println("Collecting sample $i...")
            sample = sample_from_tt(F, integrals, skeleton, links)
            write(file, "$(sample[1]) $(sample[2]) $(sample[3]) $(sample[4]) $(sample[5]) $(sample[6])\n")
        end
    end
end

d = 6

start_time = time()
aca_planetary()
end_time = time()
elapsed_time = end_time - start_time
println("Elapsed time: $elapsed_time seconds")

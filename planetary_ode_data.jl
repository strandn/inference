using DifferentialEquations

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
    println("Generating data...")

    tspan = (0.0, 15.0)
    nsteps = 20
    dt = (tspan[2] - tspan[1]) / nsteps
    tlist = LinRange(tspan..., nsteps + 1)
    qx0_true = 1.0
    qy0_true = 0.2
    px0_true = 0.2
    py0_true = 1.0
    m_true = 1.0
    k_true = 1.0

    truedata1 = zeros(nsteps + 1)
    truedata2 = zeros(nsteps + 1)
    data1 = zeros(nsteps + 1)
    data2 = zeros(nsteps + 1)

    prob = ODEProblem(planetary!, [qx0_true, qy0_true, px0_true, py0_true], tspan, [m_true, k_true])
    sol = solve(prob, Tsit5(), saveat=dt)

    truedata1 = sol[1, :]
    truedata2 = sol[2, :]
    data1 = deepcopy(truedata1)
    data2 = deepcopy(truedata2)
    data1 += sqrt(0.01) * randn(nsteps + 1)
    data2 += sqrt(0.01) * randn(nsteps + 1)

    mu = [1.0, 1.0, 1.0, 1.0, 5.0, 5.0]
    sigma = [1.0, 1.0, 1.0, 1.0, 25.0, 25.0]
    neglogposterior(qx0, qy0, px0, py0, m, k) = V([qx0, qy0, px0, py0, m, k], tspan, nsteps, data1, data2, mu, sigma)

    open("planetary_data.txt", "w") do file
        for i in 1:nsteps+1
            write(file, "$(tlist[i]) $(truedata1[i]) $(truedata2[i]) $(data1[i]) $(data2[i])\n")
        end
    end
end

aca_planetary()

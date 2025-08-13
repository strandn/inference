using DifferentialEquations

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
    println("Generating data...")

    tspan = (0.0, 50.0)
    nsteps = 30
    dt = (tspan[2] - tspan[1]) / nsteps
    tlist = LinRange(tspan..., nsteps + 1)
    x0_true = 5.0
    y0_true = 30.0
    a_true = 0.55
    b_true = 0.03
    c_true = 0.85
    d_true = 0.025

    truedata1 = zeros(nsteps + 1)
    truedata2 = zeros(nsteps + 1)
    truedata = zeros(nsteps + 1)
    data = zeros(nsteps + 1)

    prob = ODEProblem(lv!, [x0_true, y0_true], tspan, [a_true, b_true, c_true, d_true])
    sol = solve(prob, Tsit5(), saveat=dt)

    truedata1 = sol[1, :]
    truedata2 = sol[2, :]
    truedata = truedata1 + truedata2
    data = deepcopy(truedata)
    data += sqrt(12.25) * randn(nsteps + 1)

    mu = [15.0, 15.0, 0.7, 0.05, 0.7, 0.05]
    sigma = [225.0, 225.0, 0.04, 0.0025, 0.04, 0.0025]
    neglogposterior(x0, y0, a, b, c, d) = V([x0, y0, a, b, c, d], tspan, nsteps, data, mu, sigma)

    open("lv_data.txt", "w") do file
        for i in 1:nsteps+1
            write(file, "$(tlist[i]) $(truedata1[i]) $(truedata2[i]) $(truedata[i]) $(data[i])\n")
        end
    end
end

aca_lv()

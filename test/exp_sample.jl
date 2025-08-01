using DifferentialEquations
using LinearAlgebra

include("tt_aca.jl")

function decay!(du, u, p, t)
    λ = p[1]
    du[1] = -λ * u[1]
end

function V(r, tspan, nsteps, data, mu, sigma)
    dt = (tspan[2] - tspan[1]) / nsteps
    x0 = r[1]
    λ = r[2]
    prob = ODEProblem(decay!, [x0], tspan, [λ])

    obs = undef
    try
        sol = solve(prob, Tsit5(), saveat=dt)
        if sol.retcode == ReturnCode.Success
            obs = sol[1, :]
        else
            throw(ErrorException("ODE solver failed"))
        end
    catch e
        obs = fill(Inf, nsteps + 1)
    end

    s2 = 0.1
    diff = [x0, λ] - mu
    result = 1 / 2 * sum((diff .^ 2) ./ sigma)
    for i in 1:nsteps+1
        result += 1 / 2 * log(2 * pi * s2) + (data[i] - obs[i]) ^ 2 / (2 * s2)
    end
    return result
end

function aca_exp()
    tspan = (0.0, 10.0)
    nsteps = 50
    dt = (tspan[2] - tspan[1]) / nsteps

    data = []
    open("exp_data.txt", "r") do file
        for line in eachline(file)
            cols = split(line)
            push!(data, parse(Float64, cols[3]))
        end
    end

    mu = [7.5, 0.5]
    sigma = [1.0, 0.04]
    neglogposterior(x0, λ) = V([x0, λ], tspan, nsteps, data, mu, sigma)

    x0_dom = (5.0, 10.0)
    λ_dom = (0.1, 1.0)

    F = ResFunc(neglogposterior, (x0_dom, λ_dom), 0.0, mu, sigma)

    open("exp_IJ.txt", "r") do file
        F.I, F.J = eval(Meta.parse(readline(file)))
        F.offset = parse(Float64, readline(file))
    end

    open("exp_samples.txt", "w") do file
        for i in 1:10
            println("Collecting sample $i...")
            sample = sample_from_tt(F)
            write(file, "$(sample[1]) $(sample[2])\n")
        end
    end
end

start_time = time()
aca_exp()
end_time = time()
elapsed_time = end_time - start_time
println("Elapsed time: $elapsed_time seconds")

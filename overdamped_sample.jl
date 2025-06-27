using DifferentialEquations
using LinearAlgebra

include("tt_aca.jl")

function damped_oscillator!(du, u, p, t)
    x, v = u
    ω, γ = p
    du[1] = v
    du[2] = -ω^2 * x - γ * v
end

function V(r, tspan, nsteps, data_x, data_v, mu, sigma)
    dt = (tspan[2] - tspan[1]) / nsteps
    x0 = r[1]
    v0 = r[2]
    ω = r[3]
    γ = r[4]
    prob = ODEProblem(damped_oscillator!, [x0, v0], tspan, [ω, γ])
    sol = solve(prob, Tsit5(), saveat=dt)
    obs_x = undef
    obs_v = undef
    if sol.retcode == ReturnCode.Success
        obs_x = sol[1, :]
        obs_v = sol[2, :]
    else
        obs_x = fill(200.0, nsteps + 1)
        obs_v = fill(200.0, nsteps + 1)
    end

    s2 = 0.15
    diff = [x0, v0, ω, γ] - mu
    result = 1 / 2 * sum((diff .^ 2) ./ sigma)
    for i in eachindex(data_x)
        result += log(2 * pi * s2) + (data_x[i] - obs_x[i]) ^ 2 / (2 * s2) + (data_v[i] - obs_v[i]) ^ 2 / (2 * s2)
    end
    return result
end

function aca_damped()
    tspan = (0.0, 20.0)
    nsteps = 50
    dt = (tspan[2] - tspan[1]) / nsteps

    data_x = []
    data_v = []
    open("overdamped_data.txt", "r") do file
        for line in eachline(file)
            cols = split(line)
            push!(data_x, parse(Float64, cols[4]))
            push!(data_v, parse(Float64, cols[5]))
        end
    end

    mu = [7.0, 3.0, 1.5, 2.5]
    sigma = [4.0, 9.0, 1.0, 16.0]
    neglogposterior(x0, v0, ω, γ) = V([x0, v0, ω, γ], tspan, nsteps, data_x, data_v, mu, sigma)

    x0_dom = (3.0, 12.0)
    v0_dom = (-1.0, 7.0)
    ω_dom = (0.1, 2.0)
    γ_dom = (0.1, 8.5)

    F = ResFunc(neglogposterior, (x0_dom, v0_dom, ω_dom, γ_dom), 0.0, mu, sigma)

    open("overdamped_IJ.txt", "r") do file
        F.I, F.J = eval(Meta.parse(readline(file)))
        F.offset = parse(Float64, readline(file))
    end

    norm = compute_norm(F)
    println("norm = $norm")

    nbins = 100
    grid = (LinRange(x0_dom..., nbins + 1), LinRange(v0_dom..., nbins + 1), LinRange(ω_dom..., nbins + 1), LinRange(γ_dom..., nbins + 1))

    for count in 1:3
        dens = compute_marginal(F, count, norm)
        open("overdamped_marginal_$count.txt", "w") do file
            for i in 1:nbins
                for j in 1:nbins
                    write(file, "$(grid[count][i]) $(grid[count + 1][j]) $(dens[i, j])\n")
                end
            end
        end
    end

    open("overdamped_samples.txt", "w") do file
        for i in 1:10
            println("Collecting sample $i...")
            sample = sample_from_tt(F)
            write(file, "$(sample[1]) $(sample[2]) $(sample[3]) $(sample[4])\n")
        end
    end
end

start_time = time()
aca_damped()
end_time = time()
elapsed_time = end_time - start_time
println("Elapsed time: $elapsed_time seconds")

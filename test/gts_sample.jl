using DifferentialEquations
using LinearAlgebra

include("tt_aca.jl")

function gts!(du, u, p, t)
    x, y = u
    α1, α2, β, γ = p
    du[1] = α1 / (1 + y ^ β) - x
    du[2] = α2 / (1 + x ^ γ) - y
end

function V(r, tspan, nsteps, data_x, data_y, mu, sigma)
    dt = (tspan[2] - tspan[1]) / nsteps
    x0 = r[1]
    y0 = r[2]
    α1 = r[3]
    α2 = r[4]
    β = r[5]
    γ = r[6]
    prob = ODEProblem(gts!, [x0, y0], tspan, [α1, α2, β, γ])

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

    s2 = 0.2
    diff = [x0, y0, α1, α2, β, γ] - mu
    result = 1 / 2 * sum((diff .^ 2) ./ sigma)
    for i in eachindex(data_x)
        result += log(2 * pi * s2) + (data_x[i] - obs_x[i]) ^ 2 / (2 * s2) + (data_y[i] - obs_y[i]) ^ 2 / (2 * s2)
    end
    return result
end

function aca_gts()
    tspan = (0.0, 30.0)
    nsteps = 40

    data_x = []
    data_y = []
    open("gts_data.txt", "r") do file
        for line in eachline(file)
            cols = split(line)
            push!(data_x, parse(Float64, cols[4]))
            push!(data_y, parse(Float64, cols[5]))
        end
    end

    mu = [2.5, 2.5, 35.0, 35.0, 2.0, 2.0]
    sigma = [6.25, 6.25, 400.0, 400.0, 4.0, 4.0]
    neglogposterior(x0, y0, α1, α2, β, γ) = V([x0, y0, α1, α2, β, γ], tspan, nsteps, data_x, data_y, mu, sigma)

    x0_dom = (0.0, 10.0)
    y0_dom = (0.0, 10.0)
    α1_dom = (5.0, 100.0)
    α2_dom = (5.0, 100.0)
    β_dom = (1.0, 5.0)
    γ_dom = (1.0, 5.0)

    F = ResFunc(neglogposterior, (x0_dom, y0_dom, α1_dom, α2_dom, β_dom, γ_dom), 0.0, mu, sigma)

    open("gts_IJ.txt", "r") do file
        F.I, F.J = eval(Meta.parse(readline(file)))
        F.offset = parse(Float64, readline(file))
    end

    norm = compute_norm(F)
    println("norm = $norm")

    nbins = 100
    grid = (
        LinRange(x0_dom..., nbins + 1),
        LinRange(y0_dom..., nbins + 1),
        LinRange(α1_dom..., nbins + 1),
        LinRange(α2_dom..., nbins + 1),
        LinRange(β_dom..., nbins + 1),
        LinRange(γ_dom..., nbins + 1)
    )

    for count in 1:5
        dens = compute_marginal(F, count, norm)
        open("gts_marginal_$count.txt", "w") do file
            for i in 1:nbins
                for j in 1:nbins
                    write(file, "$(grid[count][i]) $(grid[count + 1][j]) $(dens[i, j])\n")
                end
            end
        end
    end

    open("gts_samples.txt", "w") do file
        for i in 1:10
            println("Collecting sample $i...")
            sample = sample_from_tt(F)
            write(file, "$(sample[1]) $(sample[2]) $(sample[3]) $(sample[4]) $(sample[5]) $(sample[6])\n")
        end
    end
end

start_time = time()
aca_gts()
end_time = time()
elapsed_time = end_time - start_time
println("Elapsed time: $elapsed_time seconds")

using DifferentialEquations

include("tt_cross.jl")

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

    s2 = 0.25
    diff = [x0, y0, α1, α2, β, γ] - mu
    result = 1 / 2 * sum((diff .^ 2) ./ sigma)
    for i in eachindex(data_x)
        result += log(2 * pi * s2) + (data_x[i] - obs_x[i]) ^ 2 / (2 * s2) + (data_y[i] - obs_y[i]) ^ 2 / (2 * s2)
    end
    return result
end

function dmrg_gts()
    tspan = (0.0, 30.0)
    nsteps = 25

    x0_true = 2.0
    y0_true = 3.0
    α1_true = 50.0
    α2_true = 16.0
    β_true = 2.5
    γ_true = 1.5
    
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

    x0_dom = (0.0, 5.0)
    y0_dom = (0.0, 5.5)
    α1_dom = (5.0, 100.0)
    α2_dom = (13.5, 18.5)
    β_dom = (1.0, 3.5)
    γ_dom = (1.0, 2.75)

    nbins = 100
     grid = (
        LinRange(x0_dom..., nbins + 1),
        LinRange(y0_dom..., nbins + 1),
        LinRange(α1_dom..., nbins + 1),
        LinRange(α2_dom..., nbins + 1),
        LinRange(β_dom..., nbins + 1),
        LinRange(γ_dom..., nbins + 1)
    )
    x0_idx = searchsortedfirst(grid[1], x0_true)
    y0_idx = searchsortedfirst(grid[2], y0_true)
    α1_idx = searchsortedfirst(grid[3], α1_true)
    α2_idx = searchsortedfirst(grid[4], α2_true)
    β_idx = searchsortedfirst(grid[5], β_true)
    γ_idx = searchsortedfirst(grid[6], γ_true)

    offset = neglogposterior(x0_true, y0_true, α1_true, α2_true, β_true, γ_true)

    println("Starting DMRG cross...")
    flush(stdout)

    posterior(x...) = exp(offset - neglogposterior(x...))
    A = ODEArray(posterior, grid)
    seedlist = [
        [x0_idx, y0_idx, α1_idx, α2_idx, β_idx, γ_idx]
    ]
    dmrg_cross(A, maxr, cutoff, tol, maxiter)
    # dmrg_cross(A, maxr, cutoff, tol, maxiter, seedlist)
end

d = 8
maxr = 100
cutoff = 1.0e-12
tol = 1.0e-4
maxiter = 10

dmrg_gts()

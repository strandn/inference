using DifferentialEquations
using LinearAlgebra
using ITensors
using ITensorMPS

function damped_oscillator!(du, u, p, t)
    x, v = u
    ω, γ = p
    du[1] = v
    du[2] = -ω ^ 2 * x - γ * v
end

function nlp(r, tspan, nsteps, data_x, data_v, mu, sigma)
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

function ttsvd_damped()
    tspan = (0.0, 20.0)
    nsteps = 50
    dt = (tspan[2] - tspan[1]) / nsteps
    x0_true = 7.5
    v0_true = 2.5
    ω_true = 1.0
    γ_true = 0.4

    truedata_x = zeros(nsteps + 1)
    truedata_v = zeros(nsteps + 1)
    data_x = zeros(nsteps + 1)
    data_v = zeros(nsteps + 1)

    prob = ODEProblem(damped_oscillator!, [x0_true, v0_true], tspan, [ω_true, γ_true])
    sol = solve(prob, Tsit5(), saveat=dt)

    truedata_x = sol[1, :]
    truedata_v = sol[2, :]
    data_x = deepcopy(truedata_x)
    data_v = deepcopy(truedata_v)
    data_x += sqrt(0.15) * randn(length(data_x))
    data_v += sqrt(0.15) * randn(length(data_v))

    mu = [7.0, 3.0, 1.5, 2.5]
    sigma = [4.0, 9.0, 1.0, 16.0]
    neglogposterior(x0, v0, ω, γ) = nlp([x0, v0, ω, γ], tspan, nsteps, data_x, data_v, mu, sigma)

    x0_dom = (3.0, 12.0)
    v0_dom = (-1.0, 7.0)
    ω_dom = (0.1, 2.0)
    γ_dom = (0.1, 7.5)

    nbins = 100
    grid = (LinRange(x0_dom..., nbins + 1), LinRange(v0_dom..., nbins + 1), LinRange(ω_dom..., nbins + 1), LinRange(γ_dom..., nbins + 1))

    println("Populating tensor...\n")
    
    A = zeros(Float64, nbins, nbins, nbins, nbins)
    nlA = zeros(Float64, nbins, nbins, nbins, nbins)
    Threads.@threads for i in 1:nbins
        for j in 1:nbins
            for k in 1:nbins
                for l in 1:nbins
                    nlA[i, j, k, l] = neglogposterior(grid[1][i], grid[2][j], grid[3][k], grid[4][l])
                end
            end
        end
    end

    peak = argmin(nlA)
    println([grid[i][peak[i]] for i in 1:d])
    println()
    ranges = [c-2:c+2 for c in Tuple(peak)]
    display(nlA[ranges...])
    println()
    
    nlA .-= minimum(nlA)
    A .= exp.(-nlA)

    display(A[ranges...])
    println()

    psi = Vector{ITensor}(undef, d)
    nlpsi = Vector{ITensor}(undef, d)
    sites = siteinds(nbins, d)

    println("Computing nlposterior TT...\n")

    nlpsi[1], S, V = svd(ITensor(nlA, sites...), sites[1]; cutoff=cutoff)
    for i in 2:d-1
        link = commonindex(nlpsi[i - 1], S)
        nlpsi[i], S, V = svd(S * V, link, sites[i]; cutoff=cutoff)
    end
    nlpsi[d] = S * V

    @show MPS(nlpsi)

    println("Computing posterior TT...\n")

    psi[1], S, V = svd(ITensor(A, sites...), sites[1]; cutoff=cutoff)
    for i in 2:d-1
        link = commonindex(psi[i - 1], S)
        psi[i], S, V = svd(S * V, link, sites[i]; cutoff=cutoff)
    end
    psi[d] = S * V

    @show MPS(psi)
end

d = 4
cutoff = 1.0e-8
ttsvd_damped()

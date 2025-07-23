using DifferentialEquations
using ITensors
using ITensorMPS
using HDF5

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

    offset = neglogposterior(x0_true, y0_true, α1_true, α2_true, β_true, γ_true)

    f = h5open("dmrg_cross_$iter.h5", "r")
    psi = read(f, "factor", MPS)
    close(f)

    sites = siteinds(psi)
    oneslist = [ITensor(ones(nbins), sites[i]) for i in 1:d]
    norm = psi[1] * oneslist[1]
    for i in 2:d
        norm *= psi[i] * oneslist[i]
    end

    println(offset - log(norm[]))

    for pos in 1:d-1
        Lenv = undef
        Renv = undef
        if pos != 1
            Lenv = psi[1] * oneslist[1]
            for i in 2:pos-1
                Lenv *= psi[i] * oneslist[i]
            end
        end
        if pos != d - 1
            Renv = psi[d] * oneslist[d]
            for i in d-1:-1:pos+2
                Renv *= psi[i] * oneslist[i]
            end
        end
        result = undef
        if pos == 1
            result = psi[1] * psi[2] * Renv
        elseif pos + 1 == d
            result = Lenv * psi[d - 1] * psi[d]
        else
            result = Lenv * psi[pos] * psi[pos + 1] * Renv
        end
        result /= norm
        open("dmrg_gts_marginal_$pos.txt", "w") do file
            for i in 1:nbins
                for j in 1:nbins
                    write(file, "$(grid[pos][i]) $(grid[pos + 1][j]) $(result[sites[pos] => i, sites[pos + 1] => j])\n")
                end
            end
        end
    end
end

d = 6
iter = 1

dmrg_gts()

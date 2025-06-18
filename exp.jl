using DifferentialEquations
using LinearAlgebra

tspan = (0.0, 10.0)
nsteps = 50
tlist = LinRange(tspan..., nsteps + 1)

function decay!(du, u, p, t)
    λ = p[1]
    du[1] = -λ * u[1]
end

function posterior(x0, λ)
    prob = ODEProblem(decay!, [x0], tspan, [λ])
    sol = solve(prob, Tsit5(), saveat=(tspan[2]-tspan[1])/nsteps)
    obs = sol[1, :]

    s2 = 0.1
    mu = [7.5, 0.5]
    sigma = zeros(2, 2)
    sigma[1, 1] = 1.0
    sigma[2, 2] = 0.04
    diff = [u[1], p[1]] - mu
    result = -1 / 2 * dot(diff, inv(sigma) * diff)
    for i in eachindex(data)
        result -= 1 / 2 * log(2 * pi * s2) + (data[i] - obs[i]) ^ 2 / (2 * s2)
    end
    return exp(result)
end

function ttcross_exp()
    x0_true = 7.5
    λ_true = 0.5

    prob = ODEProblem(decay!, [x0_true], tspan, [λ_true])
    sol = solve(prob, Tsit5(), saveat=(tspan[2]-tspan[1])/nsteps)

    data = sol[1, :]
    data += sqrt(0.1) * randn(length(data))

    open("exp_data.txt", "w") do file
        for i in 1:nsteps+1
            write(file, "$(tlist[i]) $(sol[1, i]) $(data[i])\n")
        end
    end

    bins = 50
    x0_dom = [5.0, 10.0]
    λ_dom = [0.1, 1.0]
    x0_vals = [x0_dom[1] + (x0_dom[2] - x0_dom[1]) * (i - 1) / bins for i in 1:bins+1]
    λ_vals = [λ_dom[1] + (λ_dom[2] - λ_dom[1]) * (i - 1) / bins for i in 1:bins+1]
    density = zeros(bins + 1, bins + 1)
    for i in 1:bins+1
        println(i)
        for j in 1:bins+1
            density[j, i] = posterior([x0_vals[i]], [λ_vals[j]])
        end
    end

    open("exp_density_true.txt", "w") do file
        for i in 1:bins+1
            for j in 1:bins+1
                write(file, "$(x0_vals[i]) $(λ_vals[j]) $(density[j, i])\n")
            end
        end
    end

    
end

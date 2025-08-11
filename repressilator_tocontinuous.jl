using DifferentialEquations

include("tt_aca.jl")
include("tt_cross.jl")

function repressilator!(du, u, p, t)
    X1, X2, X3 = u
    α1, α2, α3, m, η = p
    du[1] = α1 / (1 + X2 ^ m) - η * X1
    du[2] = α2 / (1 + X3 ^ m) - η * X2
    du[3] = α3 / (1 + X1 ^ m) - η * X3
end

function V(r, tspan, nsteps, data, mu, sigma)
    dt = (tspan[2] - tspan[1]) / nsteps
    X10 = r[1]
    X20 = r[2]
    X30 = r[3]
    α1 = r[4]
    α2 = r[5]
    α3 = r[6]
    m = r[7]
    η = r[8]
    prob = ODEProblem(repressilator!, [X10, X20, X30], tspan, [α1, α2, α3, m, η])
    obs = undef
    try
        sol = solve(prob, Tsit5(), saveat=dt)
        if sol.retcode == ReturnCode.Success
            obs = sol[1, :] + sol[2, :] + sol[3, :]
        else
            throw(ErrorException("ODE solver failed"))
        end
    catch e
        obs = fill(Inf, nsteps + 1)
    end

    s2 = 0.25
    diff = [X10, X20, X30, α1, α2, α3, m, η] - mu
    result = 1 / 2 * sum((diff .^ 2) ./ sigma)
    for i in 1:nsteps+1
        result += 1 / 2 * log(2 * pi * s2) + (data[i] - obs[i]) ^ 2 / (2 * s2)
    end
    return result
end

function tt_repressilator()
    tspan = (0.0, 30.0)
    nsteps = 50

    data = []
    open("repressilator_data.txt", "r") do file
        for line in eachline(file)
            cols = split(line)
            push!(data, parse(Float64, cols[6]))
        end
    end

    mu = [2.0, 2.0, 2.0, 15.0, 15.0, 15.0, 5.0, 5.0]
    sigma = [4.0, 4.0, 4.0, 25.0, 25.0, 25.0, 25.0, 25.0]
    neglogposterior(X10, X20, X30, α1, α2, α3, m, η) = V([X10, X20, X30, α1, α2, α3, m, η], tspan, nsteps, data, mu, sigma)

    X10_true = X20_true = X30_true = 2.0
    α1_true = 10.0
    α2_true = 15.0
    α3_true = 20.0
    m_true = 4.0
    η_true = 1.0

    X10_dom = (0.5, 3.5)
    X20_dom = (0.5, 3.5)
    X30_dom = (0.5, 3.5)
    α1_dom = (0.5, 25.0)
    α2_dom = (0.5, 25.0)
    α3_dom = (0.5, 25.0)
    m_dom = (3.0, 5.0)
    η_dom = (0.95, 1.05)

    grid = (
        LinRange(X10_dom..., nbins + 1),
        LinRange(X20_dom..., nbins + 1),
        LinRange(X30_dom..., nbins + 1),
        LinRange(α1_dom..., nbins + 1),
        LinRange(α2_dom..., nbins + 1),
        LinRange(α3_dom..., nbins + 1),
        LinRange(m_dom..., nbins + 1),
        LinRange(η_dom..., nbins + 1)
    )

    offset = neglogposterior(X10_true, X20_true, X30_true, α1_true, α2_true, α3_true, m_true, η_true)

    posterior(x...) = exp(offset - neglogposterior(x...))
    A = ODEArray(posterior, grid)

    row_idx = undef
    col_idx = undef
    open("tt_cross_iter_$iter.txt", "r") do file
        row_idx = eval(Meta.parse(readline(file)))
        col_idx = eval(Meta.parse(readline(file)))
    end
    IJ = to_continuous(A, row_idx, col_idx)

    open("repressilator_IJ.txt", "w") do file
        write(file, "$IJ\n")
        write(file, "$offset\n")
    end
    norm, _, _ = compute_norm(F)
    println("norm = $norm")
    println(F.offset - log(norm))
    flush(stdout)
end

nbins = 50
iter = 19

start_time = time()
tt_repressilator()
end_time = time()
elapsed_time = end_time - start_time
println("Elapsed time: $elapsed_time seconds")

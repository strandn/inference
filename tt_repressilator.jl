using DifferentialEquations
using StatsBase
using Clustering

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

    X10_dom = (0.5, 3.5)
    X20_dom = (0.5, 3.5)
    X30_dom = (0.5, 3.5)
    α1_dom = (0.5, 25.0)
    α2_dom = (0.5, 25.0)
    α3_dom = (0.5, 25.0)
    m_dom = (3.0, 5.0)
    η_dom = (0.95, 1.05)

    grid_full = (
        collect(LinRange(X10_dom..., nbins + 1)),
        collect(LinRange(X20_dom..., nbins + 1)),
        collect(LinRange(X30_dom..., nbins + 1)),
        collect(LinRange(α1_dom..., nbins + 1)),
        collect(LinRange(α2_dom..., nbins + 1)),
        collect(LinRange(α3_dom..., nbins + 1)),
        collect(LinRange(m_dom..., nbins + 1)),
        collect(LinRange(η_dom..., nbins + 1))
    )

    samples = zeros(nsamples, d)
    open("vanilla_samples.txt", "r") do file
        for i in 1:nsamples
            sample = eval(Meta.parse(readline(file)))
            samples[i, :] = sample
        end
    end
    R = kmeans(samples', 3)
    borders = []
    for i in 1:d
        if i == 4 || i == 5 || i == 6
            clusterborders = []
            for j in 1:3
                idx = findall(x -> x == j, assignments(R))
                avg = mean(samples[idx, i])
                sd = std(samples[idx, i])
                push!(clusterborders, (avg - 5 * sd, avg + 5 * sd))
            end
            push!(borders, clusterborders)
        else
            avg = mean(samples[:, i])
            sd = std(samples[:, i])
            push!(borders, [(avg - 5 * sd, avg + 5 * sd)])
        end
    end

    grid = Tuple([Float64[] for _ in 1:d])
    for i in 1:d
        for border in borders[i]
            first = searchsortedlast(grid_full[i], border[1])
            if first == 0
                first = 1
            end
            last = searchsortedfirst(grid_full[i], border[2])
            if last == nbins + 1
                last = nbins
            end
            append!(grid[i], grid_full[i][first:last])
        end
        unique!(grid[i])
        sort!(grid[i])
    end
    println([length(g) for g in grid])

    offset = neglogposterior(samples[1, :]...)

    println("Starting TT cross...")
    flush(stdout)

    posterior(x...) = exp(offset - neglogposterior(x...))
    # A = ODEArray(posterior, grid_full)
    A = ODEArray(posterior, grid)

    seedlist = zeros(Int64, nsamples, d)
    for i in 1:nsamples
        for j in 1:d
            hi = searchsortedfirst(grid[j], samples[i, j])
            if hi == 1
                seedlist[i, j] = 1
            elseif hi == length(grid[j]) + 1
                seedlist[i, j] = length(grid[j])
            else
                lo = hi - 1
                if abs(grid[j][lo] - samples[i, j]) < abs(grid[j][hi] - samples[i, j])
                    seedlist[i, j] = lo
                else
                    seedlist[i, j] = hi
                end
            end
        end
    end

    psi = tt_cross(A, maxr, tol, maxiter)
    # psi = tt_cross(A, maxr, tol, maxiter, seedlist)
    @show psi

    sites = siteinds(psi)
    oneslist = [ITensor(ones(dim(sites[i])), sites[i]) for i in 1:d]
    norm = psi[1] * oneslist[1]
    for i in 2:d
        norm *= psi[i] * oneslist[i]
    end
    psi /= norm[]

    domprod = (X10_dom[2] - X10_dom[1]) * (X20_dom[2] - X20_dom[1]) * (X30_dom[2] - X30_dom[1]) * (α1_dom[2] - α1_dom[1]) * (α2_dom[2] - α2_dom[1]) * (α3_dom[2] - α3_dom[1]) * (m_dom[2] - m_dom[1]) * (η_dom[2] - η_dom[1])
    println(offset - log(norm[] * domprod / nbins^d))

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
        open("tt_repressilator_marginal_$pos.txt", "w") do file
            for i in 1:dim(sites[pos])
                for j in 1:dim(sites[pos+1])
                    # write(file, "$(grid_full[pos][i]) $(grid_full[pos + 1][j]) $(result[sites[pos]=>i, sites[pos+1]=>j])\n")
                    write(file, "$(grid[pos][i]) $(grid[pos + 1][j]) $(result[sites[pos]=>i, sites[pos+1]=>j])\n")
                end
            end
        end
    end

    # vec1list = [ITensor(grid_full[i], sites[i]) for i in 1:d]
    vec1list = [ITensor(grid[i], sites[i]) for i in 1:d]
    meanlist = zeros(d)
    for i in 1:d
        mean = psi[1] * (i == 1 ? vec1list[1] : oneslist[1])
        for k in 2:d
            mean *= psi[k] * (i == k ? vec1list[k] : oneslist[k])
        end
        meanlist[i] = mean[]
    end
    println(meanlist)

    # cov0 = undef
    # open("repressilator0cov.txt", "r") do file
    #     cov0 = eval(Meta.parse(readline(file)))
    # end

    # vec2list = [ITensor(grid_full[i] .- meanlist[i], sites[i]) for i in 1:d]
    # vec22list = [ITensor((grid_full[i] .- meanlist[i]).^2, sites[i]) for i in 1:d]
    vec2list = [ITensor(grid[i] .- meanlist[i], sites[i]) for i in 1:d]
    vec22list = [ITensor((grid[i] .- meanlist[i]).^2, sites[i]) for i in 1:d]
    varlist = zeros(d, d)
    for i in 1:d
        for j in i:d
            var = psi[1]
            if i == 1
                if i == j
                    var *= vec22list[1]
                else
                    var *= vec2list[1]
                end
            else
                var *= oneslist[1]
            end
            for k in 2:d
                var *= psi[k]
                if i == k || j == k
                    if i == j
                        var *= vec22list[k]
                    else
                        var *= vec2list[k]
                    end
                else
                    var *= oneslist[k]
                end
            end
            varlist[i, j] = varlist[j, i] = var[]
        end
    end
    display(varlist)
    # println(LinearAlgebra.norm(varlist - cov0) / LinearAlgebra.norm(cov0))
    flush(stdout)
end

d = 8
maxr = 100
tol = 1.0e-4
maxiter = 10
nbins = 100
nsamples = 10^4

start_time = time()
tt_repressilator()
end_time = time()
elapsed_time = end_time - start_time
println("Elapsed time: $elapsed_time seconds")

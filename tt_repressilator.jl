using DifferentialEquations
using StatsBase
using Clustering
using DelimitedFiles

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
    α1 = r[1]
    α2 = r[2]
    α3 = r[3]
    m = r[4]
    η = r[5]
    X10 = r[6]
    X20 = r[7]
    X30 = r[8]
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
    diff = [α1, α2, α3, m, η, X10, X20, X30] - mu
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

    mu = [15.0, 15.0, 15.0, 5.0, 5.0, 2.0, 2.0, 2.0]
    sigma = [25.0, 25.0, 25.0, 25.0, 25.0, 4.0, 4.0, 4.0]
    neglogposterior(α1, α2, α3, m, η, X10, X20, X30) = V([α1, α2, α3, m, η, X10, X20, X30], tspan, nsteps, data, mu, sigma)

    α1_dom = (0.5, 25.0)
    α2_dom = (0.5, 25.0)
    α3_dom = (0.5, 25.0)
    m_dom = (3.0, 5.0)
    η_dom = (0.95, 1.05)
    X10_dom = (0.5, 3.5)
    X20_dom = (0.5, 3.5)
    X30_dom = (0.5, 3.5)

    grid_full = (
        collect(LinRange(α1_dom..., nbins + 1)),
        collect(LinRange(α2_dom..., nbins + 1)),
        collect(LinRange(α3_dom..., nbins + 1)),
        collect(LinRange(m_dom..., nbins + 1)),
        collect(LinRange(η_dom..., nbins + 1)),
        collect(LinRange(X10_dom..., nbins + 1)),
        collect(LinRange(X20_dom..., nbins + 1)),
        collect(LinRange(X30_dom..., nbins + 1))
    )

    samples = zeros(nsamples, d)
    samples = readdlm("repressilator_samples.txt")
    R = kmeans(samples', 3)
    borders = []
    for i in 1:d
        if i == 1 || i == 2 || i == 3
            clusterborders = []
            for j in 1:3
                idx = findall(x -> x == j, assignments(R))
                avg = mean(samples[idx, i])
                sd = std(samples[idx, i])
                push!(clusterborders, (avg - 8 * sd, avg + 8 * sd))
            end
            push!(borders, clusterborders)
        else
            avg = mean(samples[:, i])
            sd = std(samples[:, i])
            push!(borders, [(avg - 8 * sd, avg + 8 * sd)])
        end
        println(borders[i])
    end

    grid = Tuple([Float64[] for _ in 1:d])
    for i in 1:d
        for border in borders[i]
            first = searchsortedlast(grid_full[i], border[1])
            if first < 1
                first = 1
            end
            last = searchsortedfirst(grid_full[i], border[2])
            if last > nbins
                last = nbins
            end
            append!(grid[i], grid_full[i][first:last])
        end
        unique!(grid[i])
        sort!(grid[i])
    end
    println([length(g) for g in grid])

    offset = minimum([neglogposterior(samples[i, :]...) for i in 1:nsamples])

    println("Starting TT cross...")
    flush(stdout)

    posterior(x...) = exp(offset - neglogposterior(x...))
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

    weights = deepcopy(grid)
    for i in 1:d
        weights[i][1] = (grid[i][2] - grid[i][1]) / 2
        for j in 2:length(grid[i])-1
            weights[i][j] = (grid[i][j + 1] - grid[i][j - 1]) / 2
        end
        weights[i][length(grid[i])] = (grid[i][length(grid[i])] - grid[i][length(grid[i]) - 1]) / 2
    end

    psi = tt_cross(A, maxr, tol, maxiter, seedlist)
    @show psi

    sites = siteinds(psi)
    oneslist = [ITensor(weights[i], sites[i]) for i in 1:d]
    norm = psi[1] * oneslist[1]
    for i in 2:d
        norm *= psi[i] * oneslist[i]
    end
    psi /= norm[]

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
        open("tt_repressilator_marginal_$pos.txt", "w") do file
            for i in 1:ITensors.dim(sites[pos])
                for j in 1:ITensors.dim(sites[pos+1])
                    write(file, "$(grid[pos][i]) $(grid[pos + 1][j]) $(result[sites[pos]=>i, sites[pos+1]=>j])\n")
                end
            end
        end
    end

    vec1list = [ITensor(weights[i] .* grid[i], sites[i]) for i in 1:d]
    meanlist = zeros(d)
    for i in 1:d
        mean = psi[1] * (i == 1 ? vec1list[1] : oneslist[1])
        for k in 2:d
            mean *= psi[k] * (i == k ? vec1list[k] : oneslist[k])
        end
        meanlist[i] = mean[]
    end
    println(meanlist)

    vec2list = [ITensor(weights[i] .* (grid[i] .- meanlist[i]), sites[i]) for i in 1:d]
    vec22list = [ITensor(weights[i] .* (grid[i] .- meanlist[i]).^2, sites[i]) for i in 1:d]
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
    flush(stdout)

    open("tt_repressilator_samples.txt", "w") do file
        for sampleid in 1:1000
            println("Collecting sample $sampleid...")
            sample = Vector{Float64}(undef, d)
            sampleidx = Vector{Int64}(undef, d)

            for count in 1:d
                Renv = undef
                if count != d
                    ind = ITensor(weights[d], sites[d])
                    Renv = psi[d] * ind
                    for i in d-1:-1:count+1
                        ind = ITensor(weights[i], sites[i])
                        Renv *= psi[i] * ind
                    end
                end
                u = rand()
                println("u_$count = $u")
                flush(stdout)
                a = 1
                b = ITensors.dim(sites[count])

                ind = ITensor(weights[count], sites[count])
                normi = psi[count] * ind
                for i in count-1:-1:1
                    ind = ITensor(sites[i])
                    ind[sites[i]=>sampleidx[i]] = 1.0
                    normi *= psi[i] * ind
                end
                if count != d
                    normi *= Renv
                end

                cdfi = 0.0
                while b - a > 1
                    mid = div(a + b, 2)
                    indvec = zeros(ITensors.dim(sites[count]))
                    indvec[1:mid-1] .= weights[count][1:mid-1]
                    indvec[mid] = 0.5 * (grid[count][mid] - grid[count][mid - 1])
                    ind = ITensor(indvec, sites[count])
                    cdfi = psi[count] * ind
                    for i in count-1:-1:1
                        ind = ITensor(sites[i])
                        ind[sites[i]=>sampleidx[i]] = 1.0
                        cdfi *= psi[i] * ind
                    end
                    if count != d
                        cdfi *= Renv
                    end
                    if cdfi[] / normi[] < u
                        a = mid
                    else
                        b = mid
                    end
                end

                # Boundary CDF up to x_a (half-left at a)
                Ca = 0.0
                if a > 1
                    indvec = zeros(ITensors.dim(sites[count]))
                    indvec[1:a-1] .= weights[count][1:a-1]
                    indvec[a] = 0.5 * (grid[count][a] - grid[count][a - 1])  # half-left at a
                    ind = ITensor(indvec, sites[count])
                    cdfi_a = psi[count] * ind
                    for i in count-1:-1:1
                        ind = ITensor(sites[i]); ind[sites[i] => sampleidx[i]] = 1.0
                        cdfi_a *= psi[i] * ind
                    end
                    if count != d
                        cdfi_a *= Renv
                    end
                    Ca = cdfi_a[]
                end

                # Node values at a and b (conditioned on previous dims; integrated over later dims)
                ind = ITensor(sites[count])
                ind[sites[count]=>a] = 1.0
                fa = psi[count] * ind
                for i in count-1:-1:1
                    ind = ITensor(sites[i])
                    ind[sites[i]=>sampleidx[i]] = 1.0
                    fa *= psi[i] * ind
                end
                if count != d
                    fa *= Renv
                end
                fa = fa[]

                ind = ITensor(sites[count])
                ind[sites[count] => b] = 1.0
                fb = psi[count] * ind
                for i in count-1:-1:1
                    ind = ITensor(sites[i])
                    ind[sites[i]=>sampleidx[i]] = 1.0
                    fb *= psi[i] * ind
                end
                if count != d
                    fb *= Renv
                end
                fb = fb[]

                fa = max(fa, 0.0)
                fb = max(fb, 0.0)

                # Correct discrete split inside the cell
                h = grid[count][b] - grid[count][a]
                threshold = Ca + 0.5 * h * fa

                if u * normi[] <= threshold
                    sampleidx[count] = a
                    sample[count] = grid[count][a]
                else
                    sampleidx[count] = b
                    sample[count] = grid[count][b]
                end
            end

            write(file, "$(sample[1]) $(sample[2]) $(sample[3]) $(sample[4]) $(sample[5]) $(sample[6]) $(sample[7]) $(sample[8])\n")
        end
    end
end

d = 8
maxr = 300
tol = 1.0e-4
maxiter = 10
nbins = 50
nsamples = 1000

start_time = time()
tt_repressilator()
end_time = time()
elapsed_time = end_time - start_time
println("Elapsed time: $elapsed_time seconds")

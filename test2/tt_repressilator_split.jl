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

    dom = (α1_dom, α2_dom, α3_dom, m_dom, η_dom, X10_dom, X20_dom, X30_dom)

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
    nclusters = 3
    X = samples'
    R = kmeans(X, nclusters)

    offset = minimum([neglogposterior(samples[i, :]...) for i in 1:nsamples])
    posterior(x...) = exp(offset - neglogposterior(x...))

    normtot = 0.0
    psilist = []
    rangelist = Vector{Any}(undef, nclusters)
    gridlist = []
    weightslist = []
    for cidx in 1:nclusters
        idx = findall(x -> x == cidx, assignments(R))
        borders = []
        for i in 1:d
            avg = mean(samples[idx, i])
            sd = max((maximum(samples[idx, i]) - minimum(samples[idx, i])) / 2, 0.05 * (dom[i][2] - dom[i][1]))
            push!(borders, (avg - 1.8 * sd, avg + 1.8 * sd))
        end
        println("Cluster $cidx")
        println(borders)

        rangelist[cidx] = []
        push!(gridlist, Tuple([Float64[] for _ in 1:d]))
        for i in 1:d
            first = searchsortedlast(grid_full[i], borders[i][1])
            if first < 1
                first = 1
            end
            last = searchsortedfirst(grid_full[i], borders[i][2])
            if last > nbins
                last = nbins
            end
            append!(gridlist[cidx][i], grid_full[i][first:last])
            push!(rangelist[cidx], first:last)
        end
        println([length(g) for g in gridlist[cidx]])
        
        println("Starting TT cross...")
        flush(stdout)

        A = ODEArray(posterior, gridlist[cidx])
        
        seedlist = []
        for i in 1:nsamples
            seed = zeros(Int64, d)
            valid = true
            for j in 1:d
                hi = searchsortedfirst(gridlist[cidx][j], samples[i, j])
                if hi == 1 || hi == length(gridlist[cidx][j]) + 1
                    valid = false
                    break
                else
                    lo = hi - 1
                    if abs(gridlist[cidx][j][lo] - samples[i, j]) < abs(gridlist[cidx][j][hi] - samples[i, j])
                        seed[j] = lo
                    else
                        seed[j] = hi
                    end
                end
            end
            if valid
                push!(seedlist, seed)
            end
        end
        seedlist = Matrix{Int64}(hcat(seedlist...)')

        push!(weightslist, deepcopy(gridlist[cidx]))
        for i in 1:d
            weightslist[cidx][i][1] = (gridlist[cidx][i][2] - gridlist[cidx][i][1]) / 2
            for j in 2:length(gridlist[cidx][i])-1
                weightslist[cidx][i][j] = (gridlist[cidx][i][j + 1] - gridlist[cidx][i][j - 1]) / 2
            end
            weightslist[cidx][i][length(gridlist[cidx][i])] = (gridlist[cidx][i][length(gridlist[cidx][i])] - gridlist[cidx][i][length(gridlist[cidx][i]) - 1]) / 2
        end

        push!(psilist, tt_cross(A, maxr, tol, maxiter, seedlist))
        @show psilist[cidx]

        sites = siteinds(psilist[cidx])
        oneslist = [ITensor(weightslist[cidx][i], sites[i]) for i in 1:d]
        norm = psilist[cidx][1] * oneslist[1]
        for i in 2:d
            norm *= psilist[cidx][i] * oneslist[i]
        end
        normtot += norm[]
    end
    println(offset - log(normtot))

    psilist ./= normtot

    for pos in 1:d-1
        marginals2D = zeros(nbins + 1, nbins + 1)
        for cidx in 1:nclusters
            sites = siteinds(psilist[cidx])
            oneslist = [ITensor(weightslist[cidx][i], sites[i]) for i in 1:d]
            Lenv = undef
            Renv = undef
            if pos != 1
                Lenv = psilist[cidx][1] * oneslist[1]
                for i in 2:pos-1
                    Lenv *= psilist[cidx][i] * oneslist[i]
                end
            end
            if pos != d - 1
                Renv = psilist[cidx][d] * oneslist[d]
                for i in d-1:-1:pos+2
                    Renv *= psilist[cidx][i] * oneslist[i]
                end
            end
            result = undef
            if pos == 1
                result = psilist[cidx][1] * psilist[cidx][2] * Renv
            elseif pos + 1 == d
                result = Lenv * psilist[cidx][d - 1] * psilist[cidx][d]
            else
                result = Lenv * psilist[cidx][pos] * psilist[cidx][pos + 1] * Renv
            end
            for i in 1:ITensors.dim(sites[pos])
                i2 = rangelist[cidx][pos][i]
                for j in 1:ITensors.dim(sites[pos+1])
                    j2 = rangelist[cidx][pos + 1][j]
                    marginals2D[i2, j2] += result[sites[pos]=>i, sites[pos+1]=>j]
                end
            end
            open("tt_repressilator_marginal_$(pos)_cluster$(cidx).txt", "w") do file
                for i in 1:ITensors.dim(sites[pos])
                    i2 = rangelist[cidx][pos][i]
                    for j in 1:ITensors.dim(sites[pos+1])
                        j2 = rangelist[cidx][pos + 1][j]
                        write(file, "$(grid_full[pos][i2]) $(grid_full[pos + 1][j2]) $(result[sites[pos]=>i, sites[pos+1]=>j])\n")
                    end
                end
            end
        end
        open("tt_repressilator_marginal_$pos.txt", "w") do file
            for i in 1:nbins+1
                for j in 1:nbins+1
                    write(file, "$(grid_full[pos][i]) $(grid_full[pos + 1][j]) $(marginals2D[i, j])\n")
                end
            end
        end
    end

    meanlist = zeros(d)
    for cidx in 1:nclusters
        sites = siteinds(psilist[cidx])
        oneslist = [ITensor(weightslist[cidx][i], sites[i]) for i in 1:d]
        vec1list = [ITensor(weightslist[cidx][i] .* gridlist[cidx][i], sites[i]) for i in 1:d]
        for i in 1:d
            mean = psilist[cidx][1] * (i == 1 ? vec1list[1] : oneslist[1])
            for k in 2:d
                mean *= psilist[cidx][k] * (i == k ? vec1list[k] : oneslist[k])
            end
            meanlist[i] += mean[]
        end
    end
    println(meanlist)

    varlist = zeros(d, d)
    for cidx in 1:nclusters
        sites = siteinds(psilist[cidx])
        oneslist = [ITensor(weightslist[cidx][i], sites[i]) for i in 1:d]
        vec2list = [ITensor(weightslist[cidx][i] .* (gridlist[cidx][i] .- meanlist[i]), sites[i]) for i in 1:d]
        vec22list = [ITensor(weightslist[cidx][i] .* (gridlist[cidx][i] .- meanlist[i]).^2, sites[i]) for i in 1:d]
        for i in 1:d
            for j in i:d
                var = psilist[cidx][1]
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
                    var *= psilist[cidx][k]
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
                varlist[i, j] += var[]
                varlist[j, i] += var[]
            end
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
                Renvlist = Vector{Any}(undef, nclusters)
                if count != d
                    for cidx in 1:nclusters
                        sites = siteinds(psilist[cidx])
                        ind = ITensor(weightslist[cidx][d], sites[d])
                        Renvlist[cidx] = psilist[cidx][d] * ind
                        for i in d-1:-1:count+1
                            ind = ITensor(weightslist[cidx][i], sites[i])
                            Renvlist[cidx] *= psilist[cidx][i] * ind
                        end
                    end
                end
                u = rand()
                println("u_$count = $u")
                flush(stdout)
                target = 0

                normi = 0.0
                for cidx in 1:nclusters
                    sites = siteinds(psilist[cidx])
                    include_cluster = true
                    for i in 1:count-1
                        if !(sampleidx[i] in rangelist[cidx][i])
                            include_cluster = false
                        end
                    end
                    if include_cluster
                        ind = ITensor(weightslist[cidx][count], sites[count])
                        normi_cluster = psilist[cidx][count] * ind
                        for i in count-1:-1:1
                            ind = ITensor(sites[i])
                            ind[sites[i]=>(sampleidx[i]-first(rangelist[cidx][i])+1)] = 1.0
                            normi_cluster *= psilist[cidx][i] * ind
                        end
                        if count != d
                            normi_cluster *= Renvlist[cidx]
                        end
                        normi += normi_cluster[]
                    end
                end

                cdfi = 0.0
                for a in 2:nbins+1
                    if target > 0
                        break
                    end

                    cdfi = 0.0
                    for cidx in 1:nclusters
                        sites = siteinds(psilist[cidx])
                        include_cluster = true
                        for i in 1:count-1
                            if !(sampleidx[i] in rangelist[cidx][i])
                                include_cluster = false
                            end
                        end
                        if include_cluster
                            cdfi_cluster = undef
                            if a - 1 in rangelist[cidx][count] && a in rangelist[cidx][count]
                                indvec = zeros(ITensors.dim(sites[count]))
                                a_cluster = a - first(rangelist[cidx][count]) + 1
                                indvec[1:a_cluster-1] .= weightslist[cidx][count][1:a_cluster-1]
                                indvec[a_cluster] = 0.5 * (gridlist[cidx][count][a_cluster] - gridlist[cidx][count][a_cluster - 1])
                                ind = ITensor(indvec, sites[count])
                                cdfi_cluster = psilist[cidx][count] * ind
                            elseif a > last(rangelist[cidx][count])
                                ind = ITensor(weightslist[cidx][count], sites[count])
                                cdfi_cluster = psilist[cidx][count] * ind
                            else
                                continue
                            end
                            for i in count-1:-1:1
                                ind = ITensor(sites[i])
                                ind[sites[i]=>(sampleidx[i]-first(rangelist[cidx][i])+1)] = 1.0
                                cdfi_cluster *= psilist[cidx][i] * ind
                            end
                            if count != d
                                cdfi_cluster *= Renvlist[cidx]
                            end
                            cdfi += cdfi_cluster[]
                            if cdfi / normi > u
                                target = a - 1
                            end
                        end
                    end
                end

                # Boundary CDF up to x_a (half-left at a)
                Ca = 0.0
                if target > 1
                    for cidx in 1:nclusters
                        sites = siteinds(psilist[cidx])
                        include_cluster = true
                        for i in 1:count-1
                            if !(sampleidx[i] in rangelist[cidx][i])
                                include_cluster = false
                            end
                        end
                        if include_cluster
                            cdfi_cluster = undef
                            if target - 1 in rangelist[cidx][count] && target in rangelist[cidx][count]
                                indvec = zeros(ITensors.dim(sites[count]))
                                a_cluster = target - first(rangelist[cidx][count]) + 1
                                indvec[1:a_cluster-1] .= weightslist[cidx][count][1:a_cluster-1]
                                indvec[a_cluster] = 0.5 * (gridlist[cidx][count][a_cluster] - gridlist[cidx][count][a_cluster - 1])
                                ind = ITensor(indvec, sites[count])
                                cdfi_cluster = psilist[cidx][count] * ind
                            elseif target > last(rangelist[cidx][count])
                                ind = ITensor(weightslist[cidx][count], sites[count])
                                cdfi_cluster = psilist[cidx][count] * ind
                            else
                                continue
                            end
                            for i in count-1:-1:1
                                ind = ITensor(sites[i])
                                ind[sites[i]=>(sampleidx[i]-first(rangelist[cidx][i])+1)] = 1.0
                                cdfi_cluster *= psilist[cidx][i] * ind
                            end
                            if count != d
                                cdfi_cluster *= Renvlist[cidx]
                            end
                            Ca += cdfi_cluster[]
                        end
                    end
                end

                # Node values at a and b (conditioned on previous dims; integrated over later dims)
                fa = 0.0
                for cidx in 1:nclusters
                    sites = siteinds(psilist[cidx])
                    include_cluster = target in rangelist[cidx][count]
                    for i in 1:count-1
                        if !(sampleidx[i] in rangelist[cidx][i])
                            include_cluster = false
                        end
                    end
                    if include_cluster
                        ind = ITensor(sites[count])
                        ind[sites[count]=>(target-first(rangelist[cidx][count])+1)] = 1.0
                        fa_cluster = psilist[cidx][count] * ind
                        for i in count-1:-1:1
                            ind = ITensor(sites[i])
                            ind[sites[i]=>(sampleidx[i]-first(rangelist[cidx][i])+1)] = 1.0
                            fa_cluster *= psilist[cidx][i] * ind
                        end
                        if count != d
                            fa_cluster *= Renvlist[cidx]
                        end
                        fa += fa_cluster[]
                    end
                end
                fa = max(fa, 0.0)

                # Correct discrete split inside the cell
                h = grid_full[count][target + 1] - grid_full[count][target]
                threshold = Ca + 0.5 * h * fa

                if u * normi <= threshold
                    sampleidx[count] = target
                    sample[count] = grid_full[count][target]
                else
                    sampleidx[count] = target + 1
                    sample[count] = grid_full[count][target + 1]
                end
            end

            write(file, "$(sample[1]) $(sample[2]) $(sample[3]) $(sample[4]) $(sample[5]) $(sample[6]) $(sample[7]) $(sample[8])\n")
        end
    end
end

d = 8
maxr = 50
tol = 1.0e-4
maxiter = 10
nbins = 50
nsamples = 1000

start_time = time()
tt_repressilator()
end_time = time()
elapsed_time = end_time - start_time
println("Elapsed time: $elapsed_time seconds")

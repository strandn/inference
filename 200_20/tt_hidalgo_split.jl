using StatsBase
using Clustering
using DelimitedFiles
using Distributions

include("tt_cross.jl")

function V(r, data)
    K = 3
    mu = r[1:3]
    lambda = exp.(r[4:6])
    q = [r[7:8]; 1.0 - r[7] - r[8]]
    beta = r[9]

    # ---- Hyperparameters ----
    M = mean(data)
    R = maximum(data) - minimum(data)
    κ = 4 / R^2
    α = 2.0
    g = 0.2
    h = 100 * g / (α * R^2)

    # ---- Priors ----
    # Means
    logprior_mu = sum(logpdf(Normal(M, sqrt(1/κ)), μk) for μk in mu)

    # Precisions (Gamma(α, β) with shape α, rate β)
    logprior_lambda = sum(logpdf(Gamma(α, beta), λk) for λk in lambda)

    # Hyperprior on β
    logprior_beta = logpdf(Gamma(g, h), beta)

    # Mixture weights (Dirichlet(1,...,1))
    # log Dirichlet(1,...,1) = 0 if q in simplex, -Inf otherwise
    logprior_q = (all(q .>= 0) && isapprox(sum(q), 1.0; atol=1e-8)) ? 0.0 : -Inf

    logprior = logprior_mu + logprior_lambda + logprior_beta + logprior_q

    # ---- Likelihood ----
    loglik = 0.0
    for yi in data
        # each mixture component density
        comps = [q[k] * pdf(Normal(mu[k], 1 / sqrt(lambda[k])), yi) for k in 1:K]
        pyi = sum(comps)
        if pyi <= 0
            return Inf  # guard against underflow/invalid parameters
        end
        loglik += log(pyi)
    end

    # ---- Posterior ----
    logpost = logprior + loglik
    return -logpost  # negative log posterior
end

function tt_hidalgo()
    data = parse.(Float64, filter(!isempty, readlines("hidalgo_stamp_thicknesses.csv")))
    data *= 100
    neglogposterior(mu1, mu2, mu3, ll1, ll2, ll3, q1, q2, beta) = V([mu1, mu2, mu3, ll1, ll2, ll3, q1, q2, beta], data)

    mu1_dom = (6.0, 12.0)
    mu2_dom = (6.0, 12.0)
    mu3_dom = (6.0, 12.0)
    ll1_dom = (-2.0, 5.0)
    ll2_dom = (-2.0, 5.0)
    ll3_dom = (-2.0, 5.0)
    q1_dom = (0.0, 0.6)
    q2_dom = (0.0, 0.6)
    beta_dom = (1.0, 4.0)

    dom = (mu1_dom, mu2_dom, mu3_dom, ll1_dom, ll2_dom, ll3_dom, q1_dom, q2_dom, beta_dom)

    grid_full = (
        collect(LinRange(mu1_dom..., nbins + 1)),
        collect(LinRange(mu2_dom..., nbins + 1)),
        collect(LinRange(mu3_dom..., nbins + 1)),
        collect(LinRange(ll1_dom..., nbins + 1)),
        collect(LinRange(ll2_dom..., nbins + 1)),
        collect(LinRange(ll3_dom..., nbins + 1)),
        collect(LinRange(q1_dom..., nbins + 1)),
        collect(LinRange(q2_dom..., nbins + 1)),
        collect(LinRange(beta_dom..., nbins + 1))
    )

    samples = zeros(nsamples, d)
    samples = readdlm("hidalgo_samples.txt")
    nclusters = 3
    X = samples'
    # X = (X .- mean(X, dims=2)) ./ std(X, dims=2)
    R = kmeans(X, nclusters)
    # R = kmeans(X, nclusters; init=:rand)

    offset = minimum([neglogposterior(samples[i, :]...) for i in 1:nsamples])
    # offset = 756.5821387651788
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
            # sd = max(std(samples[idx, i]), 0.01 * (dom[i][2] - dom[i][1]))
            # push!(borders, (avg - 3 * sd, avg + 3 * sd))
            sd = max(maximum(samples[idx, i]) - minimum(samples[idx, i]), 0.05 * (dom[i][2] - dom[i][1]))
            push!(borders, (avg - 1.0 * sd, avg + 1.0 * sd))
            # println("$avg $(std(samples[idx, i])) $(0.01 * (dom[i][2] - dom[i][1]))")
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
        # f = h5open("tt_cross_10_$cidx.h5", "r")
        # push!(psilist, read(f, "factor", MPS))
        # close(f)
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
            open("tt_hidalgo_marginal_$(pos)_cluster$(cidx).txt", "w") do file
                for i in 1:ITensors.dim(sites[pos])
                    i2 = rangelist[cidx][pos][i]
                    for j in 1:ITensors.dim(sites[pos+1])
                        j2 = rangelist[cidx][pos + 1][j]
                        write(file, "$(grid_full[pos][i2]) $(grid_full[pos + 1][j2]) $(result[sites[pos]=>i, sites[pos+1]=>j])\n")
                    end
                end
            end
        end
        open("tt_hidalgo_marginal_$pos.txt", "w") do file
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

    open("tt_hidalgo_samples.txt", "w") do file
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

            write(file, "$(sample[1]) $(sample[2]) $(sample[3]) $(sample[4]) $(sample[5]) $(sample[6]) $(sample[7]) $(sample[8]) $(sample[9])\n")
        end
    end
end

d = 9
maxr = 20
tol = 1.0e-4
maxiter = 10
nbins = 200
nsamples = 1000

start_time = time()
tt_hidalgo()
end_time = time()
elapsed_time = end_time - start_time
println("Elapsed time: $elapsed_time seconds")

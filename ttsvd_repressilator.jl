using DifferentialEquations
using ITensors
using ITensorMPS

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

function ttsvd_repressilator()
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

    grid = (
        collect(LinRange(X10_dom..., nbins + 1)),
        collect(LinRange(X20_dom..., nbins + 1)),
        collect(LinRange(X30_dom..., nbins + 1)),
        collect(LinRange(α1_dom..., nbins + 1)),
        collect(LinRange(α2_dom..., nbins + 1)),
        collect(LinRange(α3_dom..., nbins + 1)),
        collect(LinRange(m_dom..., nbins + 1)),
        collect(LinRange(η_dom..., nbins + 1))
    )

    println("Populating tensor...")
    flush(stdout)
    
    nlA = zeros(Float64, nbins + 1, nbins + 1, nbins + 1, nbins + 1, nbins + 1, nbins + 1, nbins + 1, nbins + 1)
    for i1 in 1:nbins+1
        for i2 in 1:nbins+1
            for i3 in 1:nbins+1
                for i4 in 1:nbins+1
                    for i5 in 1:nbins+1
                        for i6 in 1:nbins+1
                            for i7 in 1:nbins+1
                                for i8 in 1:nbins+1
                                    nlA[i1, i2, i3, i4, i5, i6, i7, i8] = neglogposterior(grid[1][i1], grid[2][i2], grid[3][i3], grid[4][i4], grid[5][i5], grid[6][i6], grid[7][i7], grid[8][i8])
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    offset = minimum(nlA)

    A = zeros(Float64, nbins + 1, nbins + 1, nbins + 1, nbins + 1, nbins + 1, nbins + 1, nbins + 1, nbins + 1)
    A[:, :, :, :, :, :, :, :] = exp(offset - nlA[:, :, :, :, :, :, :, :])

    psivec = Vector{ITensor}(undef, d)
    sites = siteinds(nbins + 1, d)

    println("Computing posterior TT...\n")
    flush(stdout)

    psivec[1], S, Vt = svd(ITensor(A, sites...), sites[1]; cutoff=cutoff)
    for i in 2:d-1
        link = commonindex(psivec[i - 1], S)
        psivec[i], S, Vt = svd(S * Vt, link, sites[i]; cutoff=cutoff)
    end
    psivec[d] = S * Vt

    psi = MPS(psivec)
    @show psi

    sites = siteinds(psi)
    oneslist = [ITensor(ones(nbins + 1), sites[i]) for i in 1:d]
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
        open("ttsvd_repressilator_marginal_$pos.txt", "w") do file
            for i in 1:nbins+1
                for j in 1:nbins+1
                    write(file, "$(grid[pos][i]) $(grid[pos + 1][j]) $(result[sites[pos] => i, sites[pos + 1] => j])\n")
                end
            end
        end
    end

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

    open("ttsvd_repressilator_samples.txt", "w") do file
        for sampleid in 1:1000
            println("Collecting sample $sampleid...")
            sample = Vector{Float64}(undef, d)
            sampleidx = Vector{Int64}(undef, d)

            for count in 1:d
                Renv = undef
                if count != d
                    ind = ITensor(ones(nbins + 1), sites[d])
                    Renv = psi[d] * ind
                    for i in d-1:-1:count+1
                        ind = ITensor(ones(nbins + 1), sites[i])
                        Renv *= psi[i] * ind
                    end
                end
                u = rand()
                println("u_$count = $u")
                flush(stdout)
                a = 1
                b = nbins + 1

                ind = ITensor(ones(nbins + 1), sites[count])
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
                while true
                    mid = div(a + b, 2)
                    if a == mid
                        break
                    end
                    indvec = zeros(nbins + 1)
                    indvec[1:mid] .= 1.0
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
                
                indvec = zeros(dim(sites[count]))
                indvec[1:b] .= 1.0
                ind = ITensor(indvec, sites[count])
                cdfi_b = psi[count] * ind
                for i in count-1:-1:1
                    ind = ITensor(sites[i])
                    ind[sites[i]=>sampleidx[i]] = 1.0
                    cdfi_b *= psi[i] * ind
                end
                if count != d
                    cdfi_b *= Renv
                end

                if abs(cdfi[] / normi[] - u) < abs(cdfi_b[] / normi[] - u)
                    sample[count] = grid[count][a]
                    sampleidx[count] = a
                else
                    sample[count] = grid[count][b]
                    sampleidx[count] = b
                end
            end

            write(file, "$(sample[1]) $(sample[2]) $(sample[3]) $(sample[4]) $(sample[5]) $(sample[6]) $(sample[7]) $(sample[8])\n")
        end
    end
end

d = 8
nbins = 10
cutoff = 1.0e-6
start_time = time()
ttsvd_repressilator()
end_time = time()
elapsed_time = end_time - start_time
println("Elapsed time: $elapsed_time seconds")

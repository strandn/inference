using LinearAlgebra
using ITensors
using ITensorMPS

function radialvelocity(v0, K, φ0, lnP, t)
    Ω = 2 * pi / exp(lnP)
    return v0 + K * cos(Ω * t + φ0)
end

function nlp(r, tspan, nsteps, data, mu, sigma)
    tlist = LinRange(tspan..., nsteps + 1)
    v0 = r[1]
    K = r[2]
    φ0 = r[3]
    lnP = r[4]
    obs = []
    for t in tlist
        push!(obs, radialvelocity(v0, K, φ0, lnP, t))
    end

    s2 = 3.24
    diff = [v0, K, φ0, lnP] - mu
    result = 1 / 2 * sum((diff .^ 2) ./ sigma)
    for i in 1:nsteps+1
        result += 1 / 2 * log(2 * pi * s2) + (data[i] - obs[i]) ^ 2 / (2 * s2)
    end
    return result
end

function ttsvd_exoplanet()
    tspan = (0.0, 200.0)
    nsteps = 50
    dt = (tspan[2] - tspan[1]) / nsteps
    v0_true = 1.0
    K_true = 10.0
    φ0_true = 5.0
    lnP_true = 4.2

    data = zeros(nsteps + 1)

    for i in 1:nsteps+1
        t = (i - 1) * dt
        data[i] = radialvelocity(v0_true, K_true, φ0_true, lnP_true, t)
    end
    data += sqrt(3.24) * randn(nsteps + 1)

    mu = [0.0, 5.0, 3.0, 4.0]
    sigma = [1.0, 9.0, 2.25, 0.25]
    neglogposterior(x0, K, φ0, lnP) = nlp([x0, K, φ0, lnP], tspan, nsteps, data, mu, sigma)

    v0_dom = (-3.0, 3.0)
    K_dom = (0.5, 14.0)
    φ0_dom = (0.0, 2 * pi)
    lnP_dom = (3.0, 5.0)

    nbins = 100
    grid = (LinRange(v0_dom..., nbins + 1), LinRange(K_dom..., nbins + 1), LinRange(φ0_dom..., nbins + 1), LinRange(lnP_dom..., nbins + 1))

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

    oneslist = [ITensor(ones(nbins), sites[i]) for i in 1:d]
    norm = psi[1] * oneslist[1]
    for i in 2:d
        norm *= psi[i] * oneslist[i]
    end

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
        open("ttsvd_exoplanet_marginal_$pos.txt", "w") do file
            for i in 1:nbins
                for j in 1:nbins
                    write(file, "$(grid[pos][i]) $(grid[pos + 1][j]) $(result[sites[pos] => i, sites[pos + 1] => j])\n")
                end
            end
        end
    end
end

d = 4
cutoff = 1.0e-8
ttsvd_exoplanet()

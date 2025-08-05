include("tt_cross.jl")

function radialvelocity(v0, K, φ0, lnP, t)
    Ω = 2 * pi / exp(lnP)
    return v0 + K * cos(Ω * t + φ0)
end

function V(r, tspan, nsteps, data, mu, sigma)
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
    diff = [v0, K] - mu
    result = 1 / 2 * sum((diff .^ 2) ./ sigma)
    for i in 1:nsteps+1
        result += 1 / 2 * log(2 * pi * s2) + (data[i] - obs[i]) ^ 2 / (2 * s2)
    end
    return result
end

function dmrg_exoplanet()
    tspan = (0.0, 200.0)
    nsteps = 6

    data = []
    open("exoplanet_data.txt", "r") do file
        for line in eachline(file)
            cols = split(line)
            push!(data, parse(Float64, cols[2]))
        end
    end

    mu = [0.0, 5.0]
    sigma = [1.0, 9.0]
    neglogposterior(x0, K, φ0, lnP) = V([x0, K, φ0, lnP], tspan, nsteps, data, mu, sigma)

    v0_true = 0.0
    K_true = 10.0
    φ0_true = 5.3
    lnP_true = 4.24

    v0_dom = (-5.0, 5.0)
    K_dom = (0.5, 20.0)
    φ0_dom = (0.0, 2 * pi)
    lnP_dom = (3.0, 5.0)

    nbins = 100
    grid = (
        LinRange(v0_dom..., nbins + 1),
        LinRange(K_dom..., nbins + 1),
        LinRange(φ0_dom..., nbins + 1),
        LinRange(lnP_dom..., nbins + 1)
    )
    v0_idx = searchsortedfirst(grid[1], v0_true)
    K_idx = searchsortedfirst(grid[2], K_true)
    φ0_idx = searchsortedfirst(grid[3], φ0_true)
    lnP_idx = searchsortedfirst(grid[4], lnP_true)

    offset = neglogposterior(v0_true, K_true, φ0_true, lnP_true)

    println("Starting DMRG cross...")
    flush(stdout)

    posterior(x...) = exp(offset - neglogposterior(x...))
    A = ODEArray(posterior, grid)
    seedlist = [
        [v0_idx, K_idx, φ0_idx, lnP_idx]
    ]
    psi = dmrg_cross(A, maxr, cutoff, tol, maxiter)
    # psi = dmrg_cross(A, maxr, cutoff, tol, maxiter, seedlist)

    sites = siteinds(psi)
    oneslist = [ITensor(ones(nbins), sites[i]) for i in 1:d]
    norm = psi[1] * oneslist[1]
    for i in 2:d
        norm *= psi[i] * oneslist[i]
    end
    psi /= norm[]

    domprod = (v0_dom[2] - v0_dom[1]) * (K_dom[2] - K_dom[1]) * (φ0_dom[2] - φ0_dom[1]) * (lnP_dom[2] - lnP_dom[1])
    println(offset - log(norm[] * domprod / 100^d))

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
        open("dmrg_exoplanet_marginal_$pos.txt", "w") do file
            for i in 1:nbins
                for j in 1:nbins
                    write(file, "$(grid[pos][i]) $(grid[pos + 1][j]) $(result[sites[pos] => i, sites[pos + 1] => j])\n")
                end
            end
        end
    end

    vec1list = [ITensor(collect(grid[i][1:nbins]), sites[i]) for i in 1:d]
    meanlist = zeros(d)
    for i in 1:d
        mean = psi[1] * (i == 1 ? vec1list[1] : oneslist[1])
        for k in 2:d
            mean *= psi[k] * (i == k ? vec1list[k] : oneslist[k])
        end
        meanlist[i] = mean[]
    end
    println(meanlist)

    cov0 = undef
    open("exoplanet0cov.txt", "r") do file
        cov0 = eval(Meta.parse(readline(file)))
    end

    vec2list = [ITensor(collect(grid[i][1:nbins] .- meanlist[i]), sites[i]) for i in 1:d]
    vec22list = [ITensor(collect((grid[i][1:nbins] .- meanlist[i]).^2), sites[i]) for i in 1:d]
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
    println(LinearAlgebra.norm(varlist - cov0) / LinearAlgebra.norm(cov0))
    flush(stdout)
end

d = 4
maxr = 100
cutoff = 1.0e-6
tol = 1.0e-4
maxiter = 3

for _ in 1:2
    start_time = time()
    dmrg_exoplanet()
    end_time = time()
    elapsed_time = end_time - start_time
    println("Elapsed time: $elapsed_time seconds")
    flush(stdout)
end

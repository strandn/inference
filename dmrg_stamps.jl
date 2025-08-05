using Distributions

include("tt_cross.jl")

function hidalgo_like(x...)
    centers = [
        [65.0, 85.0, 115.0, log(2.0^2), log(3.0^2), log(4.0^2), 0.0, 0.0, 0.0],  # equal weights
        [115.0, 65.0, 85.0, log(4.0^2), log(2.0^2), log(3.0^2), 0.5, 0.0, -0.5],
        [85.0, 115.0, 65.0, log(3.0^2), log(4.0^2), log(2.0^2), -0.5, 0.5, 0.0],
    ]

    # Covariance: moderate noise around each mode (diagonal)
    Σ = Diagonal([9.0, 9.0, 9.0, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0])

    # Define mixture
    ps = [MvNormal(μ, Σ) for μ in centers]

    # Evaluate density (unnormalized)
    return 10.0 - log(sum(pdf(p, [elt for elt in x]) for p in ps))
end

function dmrg_stamps()
    m1_true = 65.0
    m2_true = 85.0
    m3_true = 115.0
    ls1_true = log(2.0^2)
    ls2_true = log(3.0^2)
    ls3_true = log(4.0^2)
    a1_true = 0.0
    a2_true = 0.0
    a3_true = 0.0

    m1_dom = (50.0, 140.0)
    m2_dom = (50.0, 140.0)
    m3_dom = (50.0, 140.0)
    ls1_dom = (-2.0, 5.0)
    ls2_dom = (-2.0, 5.0)
    ls3_dom = (-2.0, 5.0)
    a1_dom = (-5.0, 5.0)
    a2_dom = (-5.0, 5.0)
    a3_dom = (-5.0, 5.0)

    nbins = 100
    grid = (
        LinRange(m1_dom..., nbins + 1),
        LinRange(m2_dom..., nbins + 1),
        LinRange(m3_dom..., nbins + 1),
        LinRange(ls1_dom..., nbins + 1),
        LinRange(ls2_dom..., nbins + 1),
        LinRange(ls3_dom..., nbins + 1),
        LinRange(a1_dom..., nbins + 1),
        LinRange(a2_dom..., nbins + 1),
        LinRange(a3_dom..., nbins + 1)
    )
    m1_idx = searchsortedfirst(grid[1], m1_true)
    m2_idx = searchsortedfirst(grid[2], m2_true)
    m3_idx = searchsortedfirst(grid[3], m3_true)
    ls1_idx = searchsortedfirst(grid[4], ls1_true)
    ls2_idx = searchsortedfirst(grid[5], ls2_true)
    ls3_idx = searchsortedfirst(grid[5], ls3_true)
    a1_idx = searchsortedfirst(grid[6], a1_true)
    a2_idx = searchsortedfirst(grid[7], a2_true)
    a3_idx = searchsortedfirst(grid[8], a3_true)

    offset = hidalgo_like(m1_true, m2_true, m3_true, ls1_true, ls2_true, ls3_true, a1_true, a2_true, a3_true)

    println("Starting DMRG cross...")
    flush(stdout)

    posterior(x...) = exp(offset - hidalgo_like(x...))
    A = ODEArray(posterior, grid)
    seedlist = [
        [m1_idx, m2_idx, m3_idx, ls1_idx, ls2_idx, ls3_idx, a1_idx, a2_idx, a3_idx],
        [m3_idx, m1_idx, m2_idx, ls1_idx, ls2_idx, ls3_idx, a1_idx, a2_idx, a3_idx],
        [m2_idx, m3_idx, m1_idx, ls1_idx, ls2_idx, ls3_idx, a1_idx, a2_idx, a3_idx]
    ]
    # psi = dmrg_cross(A, maxr, cutoff, tol, maxiter)
    psi = dmrg_cross(A, maxr, cutoff, tol, maxiter, seedlist)

    sites = siteinds(psi)
    oneslist = [ITensor(ones(nbins), sites[i]) for i in 1:d]
    norm = psi[1] * oneslist[1]
    for i in 2:d
        norm *= psi[i] * oneslist[i]
    end
    psi /= norm[]

    domprod = (m1_dom[2] - m1_dom[1]) * (m2_dom[2] - m2_dom[1]) * (m3_dom[2] - m3_dom[1]) * (ls1_dom[2] - ls1_dom[1]) * (ls2_dom[2] - ls2_dom[1]) * (ls3_dom[2] - ls3_dom[1]) * (a1_dom[2] - a1_dom[1]) * (a2_dom[2] - a2_dom[1]) * (a3_dom[2] - a3_dom[1])
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
        open("dmrg_stamps_marginal_$pos.txt", "w") do file
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
    open("stamps0cov.txt", "r") do file
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

d = 9
maxr = 100
cutoff = 1.0e-10
tol = 0.01
maxiter = 10

for _ in 1:2
    start_time = time()
    dmrg_stamps()
    end_time = time()
    elapsed_time = end_time - start_time
    println("Elapsed time: $elapsed_time seconds")
    flush(stdout)
end

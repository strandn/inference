include("tt_aca.jl")

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

function aca_stamps()
    m1_dom = (50.0, 140.0)
    m2_dom = (50.0, 140.0)
    m3_dom = (50.0, 140.0)
    ls1_dom = (-2.0, 5.0)
    ls2_dom = (-2.0, 5.0)
    ls3_dom = (-2.0, 5.0)
    a1_dom = (-5.0, 5.0)
    a2_dom = (-5.0, 5.0)
    a3_dom = (-5.0, 5.0)

    F = ResFunc(hidalgo_like, (m1_dom, m2_dom, m3_dom, ls1_dom, ls2_dom, ls3_dom, a1_dom, a2_dom, a3_dom), 0.0, Tuple(fill(false, 9)))

    open("stamps_IJ.txt", "r") do file
        F.I, F.J = eval(Meta.parse(readline(file)))
        F.offset = parse(Float64, readline(file))
    end

    cov0 = undef
    open("stamps0cov.txt", "r") do file
        cov0 = eval(Meta.parse(readline(file)))
    end

    grid = (
        collect(LinRange(m1_dom..., nbins + 1)),
        collect(LinRange(m2_dom..., nbins + 1)),
        collect(LinRange(m3_dom..., nbins + 1)),
        collect(LinRange(ls1_dom..., nbins + 1)),
        collect(LinRange(ls2_dom..., nbins + 1)),
        collect(LinRange(ls3_dom..., nbins + 1)),
        collect(LinRange(a1_dom..., nbins + 1)),
        collect(LinRange(a2_dom..., nbins + 1)),
        collect(LinRange(a3_dom..., nbins + 1))
    )

    weights = deepcopy(grid)
    for i in 1:d
        weights[i][1] = (grid[i][2] - grid[i][1]) / 2
        for j in 2:length(grid[i])-1
            weights[i][j] = (grid[i][j + 1] - grid[i][j - 1]) / 2
        end
        weights[i][length(grid[i])] = (grid[i][length(grid[i])] - grid[i][length(grid[i]) - 1]) / 2
    end

    psi = build_tt(F, grid)
    @show psi

    sites = siteinds(psi)
    oneslist = [ITensor(weights[i], sites[i]) for i in 1:d]
    norm = psi[1] * oneslist[1]
    for i in 2:d
        norm *= psi[i] * oneslist[i]
    end
    psi /= norm[]

    println(F.offset - log(norm[]))

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
        open("stamps_marginal_$pos.txt", "w") do file
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
    println(LinearAlgebra.norm(varlist - cov0) / LinearAlgebra.norm(cov0))
    flush(stdout)

    open("stamps_samples.txt", "w") do file
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

            write(file, "$(sample[1]) $(sample[2]) $(sample[3]) $(sample[4]) $(sample[5]) $(sample[6]) $(sample[7]) $(sample[8]) $(sample[9])\n")
        end
    end
end

d = 9
nbins = 100

start_time = time()
aca_stamps()
end_time = time()
elapsed_time = end_time - start_time
println("Elapsed time: $elapsed_time seconds")

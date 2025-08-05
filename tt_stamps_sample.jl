using Distributions
using ITensors
using ITensorMPS
using HDF5

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

function tt_stamps()
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

    f = h5open("tt_cross_$iter.h5", "r")
    psi = read(f, "factor", MPS)
    close(f)

    sites = siteinds(psi)
    open("tt_stamps_samples.txt", "w") do file
        for sampleid in 1:30
            println("Collecting sample $sampleid...")
            sample = Vector{Float64}(undef, d)

            for count in 1:d
                Renv = undef
                if count != d
                    oneslist = ITensor(ones(nbins), sites[d])
                    Renv = psi[d] * oneslist
                    for i in d-1:-1:count+1
                        oneslist = ITensor(ones(nbins), sites[i])
                        Renv *= psi[i] * oneslist
                    end
                end
                u = rand()
                println("u_$count = $u")
                flush(stdout)
                a = 1
                b = nbins

                oneslist = ITensor(ones(nbins), sites[count])
                normi = psi[count] * oneslist
                for i in count-1:-1:1
                    oneslist = ITensor(sites[i])
                    oneslist[sites[i]=>sample[i]] = 1.0
                    normi *= psi[i] * oneslist
                end
                if count != d
                    normi *= Renv
                end

                while a != b
                    mid = div(a + b, 2)
                    ind = zeros(nbins)
                    ind[1:mid] .= 1.0
                    oneslist = ITensor(ind, sites[count])
                    cdfi = psi[count] * oneslist
                    for i in count-1:-1:1
                        oneslist = ITensor(sites[i])
                        oneslist[sites[i]=>sample[i]] = 1.0
                        cdfi *= psi[i] * oneslist
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
                sample[count] = grid[count][a]
            end

            write(file, "$(sample[1]) $(sample[2]) $(sample[3]) $(sample[4]) $(sample[5]) $(sample[6]) $(sample[7]) $(sample[8]) $(sample[9])\n")
        end
    end
end

d = 9
iter = 3

tt_stamps()
end_time = time()
elapsed_time = end_time - start_time
println("Elapsed time: $elapsed_time seconds")

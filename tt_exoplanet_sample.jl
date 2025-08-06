using ITensors
using ITensorMPS
using HDF5

function tt_exoplanet()
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

    f = h5open("tt_cross_$iter.h5", "r")
    psi = read(f, "factor", MPS)
    close(f)

    sites = siteinds(psi)
    open("tt_exoplanet_samples.txt", "w") do file
        for sampleid in 1:30
            println("Collecting sample $sampleid...")
            sample = Vector{Float64}(undef, d)
            sampleidx = Vector{Int64}(undef, d)

            for count in 1:d
                Renv = undef
                if count != d
                    ones = ITensor(ones(nbins), sites[d])
                    Renv = psi[d] * ones
                    for i in d-1:-1:count+1
                        ones = ITensor(ones(nbins), sites[i])
                        Renv *= psi[i] * ones
                    end
                end
                u = rand()
                println("u_$count = $u")
                flush(stdout)
                a = 1
                b = nbins

                ones = ITensor(ones(nbins), sites[count])
                normi = psi[count] * ones
                for i in count-1:-1:1
                    ones = ITensor(sites[i])
                    ones[sites[i]=>sampleidx[i]] = 1.0
                    normi *= psi[i] * ones
                end
                if count != d
                    normi *= Renv
                end

                while true
                    mid = div(a + b, 2)
                    if a == mid
                        break
                    end
                    ind = zeros(nbins)
                    ind[1:mid] .= 1.0
                    ones = ITensor(ind, sites[count])
                    cdfi = psi[count] * ones
                    for i in count-1:-1:1
                        ones = ITensor(sites[i])
                        ones[sites[i]=>sampleidx[i]] = 1.0
                        cdfi *= psi[i] * ones
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
                sampleidx[count] = a
            end

            write(file, "$(sample[1]) $(sample[2]) $(sample[3]) $(sample[4])\n")
        end
    end
end

d = 4
iter = 5

start_time = time()
tt_exoplanet()
end_time = time()
elapsed_time = end_time - start_time
println("Elapsed time: $elapsed_time seconds")

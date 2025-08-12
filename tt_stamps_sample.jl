using ITensors
using ITensorMPS
using HDF5

function tt_stamps()
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

    f = h5open("tt_cross_$iter.h5", "r")
    psi = read(f, "factor", MPS)
    close(f)

    sites = siteinds(psi)
    open("tt_stamps_samples.txt", "w") do file
        for sampleid in 1:30
            println("Collecting sample $sampleid...")
            sample = Vector{Float64}(undef, d)
            sampleidx = Vector{Int64}(undef, d)

            for count in 1:d
                Renv = undef
                if count != d
                    ind = ITensor(ones(nbins), sites[d])
                    Renv = psi[d] * ind
                    for i in d-1:-1:count+1
                        ind = ITensor(ones(nbins), sites[i])
                        Renv *= psi[i] * ind
                    end
                end
                u = rand()
                println("u_$count = $u")
                flush(stdout)
                a = 1
                b = nbins

                ind = ITensor(ones(nbins), sites[count])
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
                    indvec = zeros(nbins)
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
                
                indvec = zeros(nbins)
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

            write(file, "$(sample[1]) $(sample[2]) $(sample[3]) $(sample[4]) $(sample[5]) $(sample[6]) $(sample[7]) $(sample[8]) $(sample[9])\n")
        end
    end
end

d = 9
iter = 3

start_time = time()
tt_stamps()
end_time = time()
elapsed_time = end_time - start_time
println("Elapsed time: $elapsed_time seconds")

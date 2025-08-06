using ITensors
using ITensorMPS
using HDF5

function tt_repressilator()
    X10_dom = (0.5, 3.5)
    X20_dom = (0.5, 3.5)
    X30_dom = (0.5, 3.5)
    α1_dom = (0.5, 25.0)
    α2_dom = (0.5, 25.0)
    α3_dom = (0.5, 25.0)
    m_dom = (3.0, 5.0)
    η_dom = (0.95, 1.05)

    nbins = 100
    grid = (
        LinRange(X10_dom..., nbins + 1),
        LinRange(X20_dom..., nbins + 1),
        LinRange(X30_dom..., nbins + 1),
        LinRange(α1_dom..., nbins + 1),
        LinRange(α2_dom..., nbins + 1),
        LinRange(α3_dom..., nbins + 1),
        LinRange(m_dom..., nbins + 1),
        LinRange(η_dom..., nbins + 1)
    )

    f = h5open("dmrg_cross_$iter.h5", "r")
    psi = read(f, "factor", MPS)
    close(f)

    sites = siteinds(psi)
    open("dmrg_repressilator_samples.txt", "w") do file
        for sampleid in 1:30
            println("Collecting sample $sampleid...")
            sample = Vector{Float64}(undef, d)
            sampleidx = Vector{Int64}(undef, d)

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
                    oneslist[sites[i]=>sampleidx[i]] = 1.0
                    normi *= psi[i] * oneslist
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
                    oneslist = ITensor(ind, sites[count])
                    cdfi = psi[count] * oneslist
                    for i in count-1:-1:1
                        oneslist = ITensor(sites[i])
                        oneslist[sites[i]=>sampleidx[i]] = 1.0
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
                sampleidx[count] = a
            end

            write(file, "$(sample[1]) $(sample[2]) $(sample[3]) $(sample[4]) $(sample[5]) $(sample[6]) $(sample[7]) $(sample[8])\n")
        end
    end
end

d = 8
iter = 10

start_time = time()
tt_repressilator()
end_time = time()
elapsed_time = end_time - start_time
println("Elapsed time: $elapsed_time seconds")

using ITensors
using ITensorMPS
using HDF5

function dmrg_repressilator()
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

    X10_true = X20_true = X30_true = 2.0
    α1_true = 10.0
    α2_true = 15.0
    α3_true = 20.0
    m_true = 4.0
    η_true = 1.0

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

    offset = neglogposterior(X10_true, X20_true, X30_true, α1_true, α2_true, α3_true, m_true, η_true)

    f = h5open("dmrg_cross_$iter.h5", "r")
    psi = read(f, "factor", MPS)
    close(f)

    sites = siteinds(psi)
    oneslist = [ITensor(ones(nbins), sites[i]) for i in 1:d]
    norm = psi[1] * oneslist[1]
    for i in 2:d
        norm *= psi[i] * oneslist[i]
    end
    psi /= norm[]

    domprod = (X10_dom[2] - X10_dom[1]) * (X20_dom[2] - X20_dom[1]) * (X30_dom[2] - X30_dom[1]) * (α1_dom[2] - α1_dom[1]) * (α2_dom[2] - α2_dom[1]) * (α3_dom[2] - α3_dom[1]) * (m_dom[2] - m_dom[1]) * (η_dom[2] - η_dom[1])
    println(offset - log(norm[] * domprod / 100^d))

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

    # cov0 = undef
    # open("repressilator0cov.txt", "r") do file
    #     cov0 = eval(Meta.parse(readline(file)))
    # end

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
    # println(LinearAlgebra.norm(varlist - cov0) / LinearAlgebra.norm(cov0))
    flush(stdout)

    open("dmrg_repressilator_samples.txt", "w") do file
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

            write(file, "$(sample[1]) $(sample[2]) $(sample[3]) $(sample[4]) $(sample[5]) $(sample[6]) $(sample[7]) $(sample[8])\n")
        end
    end
end

d = 8
iter = 10

start_time = time()
dmrg_repressilator()
end_time = time()
elapsed_time = end_time - start_time
println("Elapsed time: $elapsed_time seconds")

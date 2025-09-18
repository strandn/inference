using DifferentialEquations

include("tt_aca.jl")

function lv!(du, u, p, t)
    x, y = u
    a, b, c, d = p
    du[1] = a * x - b * x * y
    du[2] = -c * y + d * x * y
end

function V(r, tspan, nsteps, data_hare, data_lynx, mu, sigma)
    dt = (tspan[2] - tspan[1]) / nsteps
    x0 = r[1]
    y0 = r[2]
    a = r[3]
    b = r[4]
    c = r[5]
    d = r[6]
    prob = ODEProblem(lv!, [x0, y0], tspan, [a, b, c, d])
    obs_hare = undef
    obs_lynx = undef
    try
        sol = solve(prob, Tsit5(), saveat=dt)
        if sol.retcode == ReturnCode.Success
            obs_hare = sol[1, :]
            obs_lynx = sol[2, :]
        else
            throw(ErrorException("ODE solver failed"))
        end
    catch e
        obs_hare = fill(Inf, nsteps + 1)
        obs_lynx = fill(Inf, nsteps + 1)
    end

    s2 = 100.0
    diff = [x0, y0, a, b, c, d] - mu
    result = 1 / 2 * sum((diff .^ 2) ./ sigma)
    for i in 1:nsteps+1
        result += log(2 * pi * s2) + (data_hare[i] - obs_hare[i]) ^ 2 / (2 * s2) + (data_lynx[i] - obs_lynx[i]) ^ 2 / (2 * s2)
    end
    return result
end

function aca_lv()
    tspan = (1900.0, 1920.0)
    nsteps = 20

    data_hare = [30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4, 27.1, 40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7]
    data_lynx = [4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1, 7.4, 8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6]

    mu = [35.0, 4.0, 0.5, 0.02, 1.0, 0.03]
    sigma = [1.0, 1.0, 0.01, 1.0e-4, 0.01, 1.0e-4]
    neglogposterior(x0, y0, a, b, c, d) = V([x0, y0, a, b, c, d], tspan, nsteps, data_hare, data_lynx, mu, sigma)

    x0_dom = (20.0, 50.0)
    y0_dom = (0.0, 9.0)
    a_dom = (0.2, 0.7)
    b_dom = (0.01, 0.05)
    c_dom = (0.4, 1.4)
    d_dom = (0.01, 0.05)

    F = ResFunc(neglogposterior, (x0_dom, y0_dom, a_dom, b_dom, c_dom, d_dom), 0.0, Tuple(fill(false, d)))

    open("lv_IJ.txt", "r") do file
        F.I, F.J = eval(Meta.parse(readline(file)))
        F.offset = parse(Float64, readline(file))
    end

    grid = (
        collect(LinRange(x0_dom..., nbins + 1)),
        collect(LinRange(y0_dom..., nbins + 1)),
        collect(LinRange(a_dom..., nbins + 1)),
        collect(LinRange(b_dom..., nbins + 1)),
        collect(LinRange(c_dom..., nbins + 1)),
        collect(LinRange(d_dom..., nbins + 1))
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

    # for pos in 1:d-1
    #     Lenv = undef
    #     Renv = undef
    #     if pos != 1
    #         Lenv = psi[1] * oneslist[1]
    #         for i in 2:pos-1
    #             Lenv *= psi[i] * oneslist[i]
    #         end
    #     end
    #     if pos != d - 1
    #         Renv = psi[d] * oneslist[d]
    #         for i in d-1:-1:pos+2
    #             Renv *= psi[i] * oneslist[i]
    #         end
    #     end
    #     result = undef
    #     if pos == 1
    #         result = psi[1] * psi[2] * Renv
    #     elseif pos + 1 == d
    #         result = Lenv * psi[d - 1] * psi[d]
    #     else
    #         result = Lenv * psi[pos] * psi[pos + 1] * Renv
    #     end
    #     open("lv_marginal_$pos.txt", "w") do file
    #         for i in 1:ITensors.dim(sites[pos])
    #             for j in 1:ITensors.dim(sites[pos+1])
    #                 write(file, "$(grid[pos][i]) $(grid[pos + 1][j]) $(result[sites[pos]=>i, sites[pos+1]=>j])\n")
    #             end
    #         end
    #     end
    # end

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
    flush(stdout)

    open("lv_samples.txt", "w") do file
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

            write(file, "$(sample[1]) $(sample[2]) $(sample[3]) $(sample[4]) $(sample[5]) $(sample[6])\n")
        end
    end
end

d = 6
nbins = 500

start_time = time()
aca_lv()
end_time = time()
elapsed_time = end_time - start_time
println("Elapsed time: $elapsed_time seconds")

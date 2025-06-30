using LinearAlgebra

include("tt_aca.jl")

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
    diff = [v0, K, φ0, lnP] - mu
    result = 1 / 2 * sum((diff .^ 2) ./ sigma)
    for i in 1:nsteps+1
        result += 1 / 2 * log(2 * pi * s2) + (data[i] - obs[i]) ^ 2 / (2 * s2)
    end
    return result
end

function aca_exoplanet()
    tspan = (0.0, 200.0)
    nsteps = 10

    data = []
    open("exoplanet_data.txt", "r") do file
        for line in eachline(file)
            cols = split(line)
            push!(data, parse(Float64, cols[2]))
        end
    end

    mu = [0.0, 5.0, 3.0, 4.0]
    sigma = [1.0, 9.0, 2.25, 0.25]
    neglogposterior(x0, K, φ0, lnP) = V([x0, K, φ0, lnP], tspan, nsteps, data, mu, sigma)

    v0_dom = (-3.0, 3.0)
    K_dom = (0.5, 14.0)
    φ0_dom = (0.0, 2 * pi)
    lnP_dom = (3.0, 5.0)

    F = ResFunc(neglogposterior, (v0_dom, K_dom, φ0_dom, lnP_dom), 0.0, mu, sigma)

    open("exoplanet_IJ.txt", "r") do file
        F.I, F.J = eval(Meta.parse(readline(file)))
        F.offset = parse(Float64, readline(file))
    end

    norm = compute_norm(F)
    println("norm = $norm")

    nbins = 100
    grid = (LinRange(v0_dom..., nbins + 1), LinRange(K_dom..., nbins + 1), LinRange(φ0_dom..., nbins + 1), LinRange(lnP_dom..., nbins + 1))

    for count in 1:3
        dens = compute_marginal(F, count, norm)
        open("exoplanet_marginal_$count.txt", "w") do file
            for i in 1:nbins
                for j in 1:nbins
                    write(file, "$(grid[count][i]) $(grid[count + 1][j]) $(dens[i, j])\n")
                end
            end
        end
    end

    open("exoplanet_samples.txt", "w") do file
        for i in 1:10
            println("Collecting sample $i...")
            sample = sample_from_tt(F)
            write(file, "$(sample[1]) $(sample[2]) $(sample[3]) $(sample[4])\n")
        end
    end
end

start_time = time()
aca_exoplanet()
end_time = time()
elapsed_time = end_time - start_time
println("Elapsed time: $elapsed_time seconds")

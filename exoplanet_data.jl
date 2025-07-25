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

function aca_exoplanet()
    println("Generating data...")

    tspan = (0.0, 200.0)
    nsteps = 6
    dt = (tspan[2] - tspan[1]) / nsteps
    tlist = LinRange(tspan..., nsteps + 1)
    v0_true = 0.0
    K_true = 10.0
    φ0_true = 5.3
    lnP_true = 4.24

    data = zeros(nsteps + 1)
    for i in 1:nsteps+1
        t = (i - 1) * dt
        data[i] = radialvelocity(v0_true, K_true, φ0_true, lnP_true, t)
    end
    data += sqrt(3.24) * randn(nsteps + 1)

    mu = [0.0, 5.0]
    sigma = [1.0, 9.0]
    neglogposterior(x0, K, φ0, lnP) = V([x0, K, φ0, lnP], tspan, nsteps, data, mu, sigma)

    open("exoplanet_data.txt", "w") do file
        for i in 1:nsteps+1
            write(file, "$(tlist[i]) $(data[i])\n")
        end
    end
end

aca_exoplanet()

using DifferentialEquations

include("tt_aca.jl")

function lv!(du, u, p, t)
    x, y = u
    a, b, c, d = p
    du[1] = a * x - b * x * y
    du[2] = -c * y + d * x * y
end

function V(r, tspan, nsteps, data, mu, sigma)
    dt = (tspan[2] - tspan[1]) / nsteps
    x0 = r[1]
    y0 = r[2]
    a = r[3]
    b = r[4]
    c = r[5]
    d = r[6]
    prob = ODEProblem(lv!, [x0, y0], tspan, [a, b, c, d])
    obs = undef
    try
        sol = solve(prob, Tsit5(), saveat=dt)
        if sol.retcode == ReturnCode.Success
            obs = sol[1, :] + sol[2, :]
        else
            throw(ErrorException("ODE solver failed"))
        end
    catch e
        obs = fill(Inf, nsteps + 1)
    end

    s2 = 12.25
    diff = [x0, y0, a, b, c, d] - mu
    result = 1 / 2 * sum((diff .^ 2) ./ sigma)
    for i in 1:nsteps+1
        result += 1 / 2 * log(2 * pi * s2) + (data[i] - obs[i]) ^ 2 / (2 * s2)
    end
    return result
end

function aca_lv()
    tspan = (0.0, 50.0)
    nsteps = 30

    data = []
    open("lv_data.txt", "r") do file
        for line in eachline(file)
            cols = split(line)
            push!(data, parse(Float64, cols[5]))
        end
    end

    mu = [15.0, 15.0, 0.7, 0.05, 0.7, 0.05]
    sigma = [225.0, 225.0, 0.04, 0.0025, 0.04, 0.0025]
    neglogposterior(x0, y0, a, b, c, d) = V([x0, y0, a, b, c, d], tspan, nsteps, data, mu, sigma)

    x0_dom = (5.0, 50.0)
    y0_dom = (5.0, 50.0)
    a_dom = (0.1, 1.0)
    b_dom = (0.01, 0.15)
    c_dom = (0.1, 1.1)
    d_dom = (0.01, 0.15)

    F = ResFunc(neglogposterior, (x0_dom, y0_dom, a_dom, b_dom, c_dom, d_dom), 0.0, Tuple(fill(false, d)))

    open("lv_IJ.txt", "r") do file
        F.I, F.J = eval(Meta.parse(readline(file)))
        F.offset = parse(Float64, readline(file))
    end

    norm, integrals, skeleton, links = compute_norm(F)
    println("norm = $norm\n")
    println(F.offset - log(norm))
    println()
    flush(stdout)

    nbins = 100
    grid = (
        LinRange(x0_dom..., nbins + 1),
        LinRange(y0_dom..., nbins + 1),
        LinRange(a_dom..., nbins + 1),
        LinRange(b_dom..., nbins + 1),
        LinRange(c_dom..., nbins + 1),
        LinRange(d_dom..., nbins + 1)
    )

    for count in 1:d-1
        dens = compute_marginal(F, integrals, skeleton, links, count)
        open("lv_marginal_$count.txt", "w") do file
            for i in 1:nbins
                for j in 1:nbins
                    write(file, "$(grid[count][i]) $(grid[count + 1][j]) $(dens[i, j] / norm)\n")
                end
            end
        end
    end

    open("lv_samples.txt", "w") do file
        for i in 1:30
            println("Collecting sample $i...")
            sample = sample_from_tt(F, integrals, skeleton, links)
            write(file, "$(sample[1]) $(sample[2]) $(sample[3]) $(sample[4]) $(sample[5]) $(sample[6])\n")
        end
    end
end

d = 6

start_time = time()
aca_lv()
end_time = time()
elapsed_time = end_time - start_time
println("Elapsed time: $elapsed_time seconds")

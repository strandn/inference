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

    F = ResFunc(hidalgo_like, (m1_dom, m2_dom, m3_dom, ls1_dom, ls2_dom, ls3_dom, a1_dom, a2_dom, a3_dom), 0.0, [], [])

    open("stamps_IJ.txt", "r") do file
        F.I, F.J = eval(Meta.parse(readline(file)))
        F.offset = parse(Float64, readline(file))
    end

    norm = compute_norm(F)
    println("norm = $norm")
    println(F.offset - log(norm))

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

    for count in 1:8
        dens = compute_marginal(F, count, norm)
        open("stamps_marginal_$count.txt", "w") do file
            for i in 1:nbins
                for j in 1:nbins
                    write(file, "$(grid[count][i]) $(grid[count + 1][j]) $(dens[i, j])\n")
                end
            end
        end
    end

    open("stamps_samples.txt", "w") do file
        for i in 1:30
            println("Collecting sample $i...")
            sample = sample_from_tt(F)
            write(file, "$(sample[1]) $(sample[2]) $(sample[3]) $(sample[4]) $(sample[5]) $(sample[6]) $(sample[7]) $(sample[8]) $(sample[9])\n")
        end
    end
end

start_time = time()
aca_stamps()
end_time = time()
elapsed_time = end_time - start_time
println("Elapsed time: $elapsed_time seconds")

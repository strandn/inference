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

    F = ResFunc(hidalgo_like, (m1_dom, m2_dom, m3_dom, ls1_dom, ls2_dom, ls3_dom, a1_dom, a2_dom, a3_dom), 0.0, fill(false, d))

    open("stamps_IJ.txt", "r") do file
        F.I, F.J = eval(Meta.parse(readline(file)))
        F.offset = parse(Float64, readline(file))
    end

    cov0 = undef
    open("stamps0cov.txt", "r") do file
        cov0 = eval(Meta.parse(readline(file)))
    end

    norm, integrals, skeleton, links = compute_norm(F)
    println("norm = $norm")
    println(F.offset - log(norm))
    println()
    flush(stdout)

    mu = zeros(d)
    for i in 1:d
        mu[i] = compute_mu(F, integrals, skeleton, links, i) / norm
    end
    println(mu)
    cov = zeros(d, d)
    for i in 1:d
        for j in i:d
            cov[i, j] = cov[j, i] = compute_cov(F, integrals, skeleton, links, mu, i, j) / norm
        end
    end
    display(cov)
    println(LinearAlgebra.norm(cov - cov0) / LinearAlgebra.norm(cov0))
    flush(stdout)
end

d = 9
start_time = time()
aca_stamps()
end_time = time()
elapsed_time = end_time - start_time
println("Elapsed time: $elapsed_time seconds")

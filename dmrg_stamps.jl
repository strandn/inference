using DifferentialEquations

include("tt_cross.jl")

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

function dmrg_repressilator()
    m1_true = 65.0
    m2_true = 85.0
    m3_true = 115.0
    ls1_true = log(2.0^2)
    ls2_true = log(3.0^2)
    ls3_true = log(4.0^2)
    a1_true = 0.0
    a2_true = 0.0
    a3_true = 0.0

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
    m1_idx = searchsortedfirst(grid[1], m1_true)
    m2_idx = searchsortedfirst(grid[2], m2_true)
    m3_idx = searchsortedfirst(grid[3], m3_true)
    ls1_idx = searchsortedfirst(grid[4], ls1_true)
    ls2_idx = searchsortedfirst(grid[5], ls2_true)
    ls3_idx = searchsortedfirst(grid[5], ls3_true)
    a1_idx = searchsortedfirst(grid[6], a1_true)
    a2_idx = searchsortedfirst(grid[7], a2_true)
    a3_idx = searchsortedfirst(grid[8], a3_true)

    offset = hidalgo_like(m1_true, m2_true, m3_true, ls1_true, ls2_true, ls3_true, a1_true, a2_true, a3_true)

    println("Starting DMRG cross...")
    flush(stdout)

    posterior(x...) = exp(offset - hidalgo_like(x...))
    A = ODEArray(posterior, grid)
    dmrg_cross(A, maxr, cutoff, tol, maxiter)
end

d = 8
maxr = 100
cutoff = 1.0e-4
tol = 0.01
maxiter = 10

dmrg_repressilator()

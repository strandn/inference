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

    F = ResFunc(hidalgo_like, (m1_dom, m2_dom, m3_dom, ls1_dom, ls2_dom, ls3_dom, a1_dom, a2_dom, a3_dom), cutoff, Tuple(fill(false, d)))

    if mpi_rank == 0
        println("Starting TT-cross ACA...")
    end

    IJ = continuous_aca(F, fill(maxr, d - 1), n_chains, n_samples, jump_width, mpi_comm)

    cov0 = undef
    open("stamps0cov.txt", "r") do file
        cov0 = eval(Meta.parse(readline(file)))
    end

    grid = (
        collect(LinRange(m1_dom..., nbins + 1)),
        collect(LinRange(m2_dom..., nbins + 1)),
        collect(LinRange(m3_dom..., nbins + 1)),
        collect(LinRange(ls1_dom..., nbins + 1)),
        collect(LinRange(ls2_dom..., nbins + 1)),
        collect(LinRange(ls3_dom..., nbins + 1)),
        collect(LinRange(a1_dom..., nbins + 1)),
        collect(LinRange(a2_dom..., nbins + 1)),
        collect(LinRange(a3_dom..., nbins + 1))
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

    sites = siteinds(psi)
    oneslist = [ITensor(weights[i], sites[i]) for i in 1:d]
    norm = psi[1] * oneslist[1]
    for i in 2:d
        norm *= psi[i] * oneslist[i]
    end
    psi /= norm[]

    println(F.offset - log(norm[]))

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
    println(LinearAlgebra.norm(varlist - cov0) / LinearAlgebra.norm(cov0))
    flush(stdout)
end

MPI.Init()
mpi_comm = MPI.COMM_WORLD
mpi_rank = MPI.Comm_rank(mpi_comm)
mpi_size = MPI.Comm_size(mpi_comm)

d = 9
maxr = 50
n_chains = 20
n_samples = 500
jump_width = 0.01
cutoff = 1.0e-6
nbins = 500

for _ in 1:20
    start_time = time()
    aca_stamps()
    end_time = time()
    elapsed_time = end_time - start_time
    if mpi_rank == 0
        println("Elapsed time: $elapsed_time seconds")
        flush(stdout)
    end
end

MPI.Finalize()

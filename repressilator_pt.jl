using MPI
using Random
using LinearAlgebra
using DifferentialEquations

function repressilator!(du, u, p, t)
    X1, X2, X3 = u
    α1, α2, α3, m, η = p
    du[1] = α1 / (1 + X2 ^ m) - η * X1
    du[2] = α2 / (1 + X3 ^ m) - η * X2
    du[3] = α3 / (1 + X1 ^ m) - η * X3
end

function V(r, tspan, nsteps, data, mu, sigma)
    dt = (tspan[2] - tspan[1]) / nsteps
    X10 = r[1]
    X20 = r[2]
    X30 = r[3]
    α1 = r[4]
    α2 = r[5]
    α3 = r[6]
    m = r[7]
    η = r[8]
    prob = ODEProblem(repressilator!, [X10, X20, X30], tspan, [α1, α2, α3, m, η])
    obs = undef
    try
        sol = solve(prob, Tsit5(), saveat=dt)
        if sol.retcode == ReturnCode.Success
            obs = sol[1, :] + sol[2, :] + sol[3, :]
        else
            throw(ErrorException("ODE solver failed"))
        end
    catch e
        obs = fill(Inf, nsteps + 1)
    end

    s2 = 0.25
    diff = [X10, X20, X30, α1, α2, α3, m, η] - mu
    result = 1 / 2 * sum((diff .^ 2) ./ sigma)
    for i in 1:nsteps+1
        result += 1 / 2 * log(2 * pi * s2) + (data[i] - obs[i]) ^ 2 / (2 * s2)
    end
    return result
end

# --------------------- Welford stats (vector) -------------------
mutable struct WelfordVec
    n::Int
    μ::Vector{Float64}
    M2::Matrix{Float64}
end
function WelfordVec(d::Int)
    WelfordVec(0, zeros(d), zeros(d, d))
end
function update!(W::WelfordVec, x::AbstractVector{<:Real})
    W.n += 1
    x = Vector{Float64}(x)
    δ = x .- W.μ
    W.μ .+= δ ./ W.n
    W.M2 .+= δ * (x .- W.μ)'          # outer product
end
mean_cov(W::WelfordVec) = (W.μ, W.n > 1 ? W.M2 ./ (W.n - 1) : fill(NaN, size(W.M2)))

# ------------------- Parallel Tempering (MPI) -------------------
"""
Run replica-exchange MCMC with one replica per MPI rank.

Arguments
- E(x)::Function     : energy function
- x0::Vector{Float64}: initial point for all replicas

Keywords
- nsteps::Int        : total local MH steps per replica
- burnin::Int        : discarded steps (β=1 chain)
- swap_every::Int    : neighbor exchange frequency
- betas::Vector{Float64} (length == nprocs) OR `nothing` to auto-build geometric
- baseσ::Float64     : proposal scale at β=1; actual σ_r = baseσ / √β_r
- seed::Int          : RNG seed base (per-rank offset added internally)

Returns (on rank 0): mean, covariance, and basic diagnostics printed.
"""
function pt_mpi(
    E::Function,
    x0::Vector{Float64};
    nsteps::Int,
    burnin::Int=0,
    swap_every::Int=10,
    betas::Union{Nothing,Vector{Float64}}=nothing,
    baseσ::Float64=0.5,
    seed::Int=42,
    save_every::Int=1,        # keep one every save_every post-burnin steps
    comm = MPI.COMM_WORLD
)
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    d = length(x0)

    # Build β ladder
    if betas === nothing
        βmax, βmin = 1.0, 0.02           # adjust as needed
        # geometric ladder (sorted high->low β so rank 0 is β≈1)
        betas = βmax .* (βmin/βmax) .^ (collect(0:nprocs-1) ./ (nprocs-1))
    else
        length(betas) == nprocs || error("length(betas) must equal number of ranks")
    end
    β = betas[rank+1]

    # RNG per-rank
    rng = MersenneTwister(seed + rank + 1)

    # Replica state on this rank
    x = copy(x0) .+ 1e-6 .* randn(rng, d)  # tiny jitter
    e = E(x...)
    σ = baseσ / sqrt(β)

    # Counters
    acc_moves = 0
    prop_moves = 0
    acc_swaps = 0
    prop_swaps = 0

    # Rank 0 will stream stats for β=1 chain (which always lives on rank 0)
    W = rank == 0 ? WelfordVec(d) : nothing

    # Local MH step
    function mh_step!()
        xnew = x .+ σ .* randn(rng, d)
        enew = E(xnew...)
        logα = -β * (enew - e)
        if log(rand(rng)) < logα
            x = xnew
            e = enew
            acc_moves += 1
        end
        prop_moves += 1
        return nothing
    end

    # One neighbor-swap pass (phase=0 for even-lower pairs, 1 for odd-lower)
    function swap_pass!(phase::Int)
        # Decide partner (if any) for this rank in this phase
        has_partner = false
        partner = -1
        lower = false
        if phase == 0
            if iseven(rank) && rank+1 < nprocs
                has_partner = true; partner = rank+1; lower = true
            elseif isodd(rank) && rank-1 >= 0
                has_partner = true; partner = rank-1; lower = false
            end
        else
            if isodd(rank) && rank+1 < nprocs
                has_partner = true; partner = rank+1; lower = true
            elseif iseven(rank) && rank-1 >= 0
                has_partner = true; partner = rank-1; lower = false
            end
        end
        if !has_partner
            MPI.Barrier(comm)
            return
        end

        # Exchange partner energy
        tagE = 100 + phase
        e_peer_ref = Ref{Float64}(0.0)
        MPI.Sendrecv!(Ref(e), partner, tagE, e_peer_ref, partner, tagE, comm)
        e_peer = e_peer_ref[]

        # Compute log acceptance: logα = (β_self - β_peer) * (E_peer - E_self)
        β_peer = betas[partner+1]
        logα = (β - β_peer) * (e_peer - e)

        # Lower rank draws uniform to decide; sends to partner
        tagU = 200 + phase
        u = 0.0
        if lower
            u = rand(rng)
            MPI.Send(Ref(u), partner, tagU, comm)
        else
            u_ref = Ref{Float64}(0.0)
            MPI.Recv!(u_ref, partner, tagU, comm)
            u = u_ref[]
        end

        # Decide acceptance
        accept = (log(u) < logα)
        prop_swaps += 1
        if accept
            # Swap states: sendrecv vectors
            tagX = 300 + phase
            tmp = similar(x)
            MPI.Sendrecv!(x, partner, tagX, tmp, partner, tagX, comm)
            x .= tmp
            e = e_peer
            acc_swaps += 1
        end

        MPI.Barrier(comm)
        return
    end

    # Main loop
    for t in 1:nsteps
        mh_step!()
        if (t % swap_every) == 0
            swap_pass!(0)  # even-lower pairs
            swap_pass!(1)  # odd-lower pairs
        end
        if rank == 0 && t > burnin && ((t - burnin) % save_every == 0)
            update!(W, x)
        end
    end

    # Reduce diagnostics to rank 0 (sums)
    tot_acc_moves = MPI.Reduce(acc_moves, +, 0, comm)
    tot_prop_moves = MPI.Reduce(prop_moves, +, 0, comm)
    tot_acc_swaps = MPI.Reduce(acc_swaps, +, 0, comm)
    tot_prop_swaps = MPI.Reduce(prop_swaps, +, 0, comm)

    if rank == 0
        μ, Σ = mean_cov(W)
        println("β ladder (first 6): ", round.(betas[1:min(end,6)]; digits=4))
        println("Local move acceptance (avg over all replicas): ",
                round(tot_acc_moves / max(1, tot_prop_moves), digits=3))
        println("Swap acceptance (avg over all attempted edges): ",
                round(tot_acc_swaps / max(1, tot_prop_swaps), digits=3))
        println()
        println(μ)
        display(Σ)
        return μ, Σ
    else
        return nothing
    end
end

# --------------------------- Driver -----------------------------
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

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

x0 = [X10_true, X20_true, X30_true, α1_true, α2_true, α3_true, m_true, η_true]
nprocs = MPI.Comm_size(comm)

# You can also pass your own `betas` vector (length == nprocs)
pt_mpi(neglogposterior, x0;
       nsteps=10^8,
       burnin=10^4,
       swap_every=100,
       baseσ=0.5,
       betas=nothing,        # auto geometric ladder
       seed=1234,
       save_every=100,
       comm=comm)

MPI.Barrier(comm)
MPI.Finalize()

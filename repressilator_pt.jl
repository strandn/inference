using MPI
using Random
using LinearAlgebra
using DifferentialEquations
using Statistics

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

# Reflect x into [lo, hi] (billiard reflection), handles multi-hop if step is large
@inline function reflect_into!(x::Vector{Float64}, domains::Vector{Tuple{Float64,Float64}})
    @inbounds for i in eachindex(x)
        lo, hi = domains[i]
        L = hi - lo
        xi = x[i]
        if xi < lo || xi > hi
            # map to [0, L] then reflect via mod
            y = xi - lo
            ymod = mod(y, 2L)
            if ymod <= L
                xi_ref = lo + ymod
            else
                xi_ref = hi - (ymod - L)
            end
            x[i] = xi_ref
        end
    end
    return x
end

# Build per-dim base σ from fractions of domain length
@inline function sigma_base_from_domains(domains, fracσ)
    @assert length(domains) == length(fracσ)
    σ = Vector{Float64}(undef, length(domains))
    @inbounds for i in eachindex(domains)
        lo, hi = domains[i]
        σ[i] = fracσ[i] * (hi - lo)
    end
    return σ
end

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
    seed::Int=42,
    save_every::Int=1,        # keep one every save_every post-burnin steps
    domains::Vector{Tuple{Float64,Float64}},
    fracσ::Vector{Float64},
    comm = MPI.COMM_WORLD
)

    # how many thinned samples we will keep on rank 0
    function n_kept(nsteps, burnin, save_every)
        kept = nsteps - burnin
        kept <= 0 && return 0
        return (kept + (save_every - 1)) ÷ save_every  # ceiling division
    end

    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    d = length(x0)
    @assert length(domains) == d
    @assert length(fracσ) == d

    Nkeep = n_kept(nsteps, burnin, save_every)
    samples = (rank == 0 && Nkeep > 0) ? Matrix{Float64}(undef, d, Nkeep) : nothing
    keepidx = 0

    # Build β ladder
    if betas === nothing
        βmax, βmin = 1.0, 0.001           # adjust as needed
        # geometric ladder (sorted high->low β so rank 0 is β≈1)
        betas = βmax .* (βmin/βmax) .^ (collect(0:nprocs-1) ./ (nprocs-1))
    else
        length(betas) == nprocs || error("length(betas) must equal number of ranks")
    end
    β = betas[rank+1]

    # RNG per-rank
    rng = MersenneTwister(seed + rank + 1)

    # Reflect initial state inside domain (just in case) and jitter a hair
    x = copy(x0)
    reflect_into!(x, domains)
    x .+= 1e-9 .* randn(rng, d)

    # Per-dim base σ and β-scaled σ
    σ_base = sigma_base_from_domains(domains, fracσ)   # size d
    σ_vec  = σ_base ./ sqrt(β)                         # per-dim scaling

    e = E(x...)   # your energy takes splatted tuple

    acc_moves = 0; prop_moves = 0
    acc_swaps = 0; prop_swaps = 0

    function mh_step!()
        xnew = @views x .+ σ_vec .* randn(rng, d)   # per-dim step
        reflect_into!(xnew, domains)                # keep inside box by reflection
        enew = E(xnew...)
        logα = -β * (enew - e)
        if log(rand(rng)) < logα
            x .= xnew
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
            keepidx += 1
            samples[:, keepidx] = x
        end
    end

    # Reduce diagnostics to rank 0 (sums)
    tot_acc_moves = MPI.Reduce(acc_moves, +, 0, comm)
    tot_prop_moves = MPI.Reduce(prop_moves, +, 0, comm)
    tot_acc_swaps = MPI.Reduce(acc_swaps, +, 0, comm)
    tot_prop_swaps = MPI.Reduce(prop_swaps, +, 0, comm)

    if rank == 0
        if Nkeep == 0
            println("No samples kept (check burnin/save_every).")
            return nothing
        end

        μ = vec(mean(samples; dims=2))
        Σ = cov(samples')

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

# 8D: [X10, X20, X30, α1, α2, α3, m, η]
domains = [
    (0.5, 3.5),  # X10
    (0.5, 3.5),  # X20
    (0.5, 3.5),  # X30
    (0.5, 25.0),  # α1
    (0.5, 25.0),  # α2
    (0.5, 25.0),  # α3
    (3.0, 5.0),   # m  (≥1)
    (0.95, 1.05)    # η
]

# fraction of domain length for base step at β=1 (tune later)
fracσ = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002]  # 2% of each span

# You can also pass your own `betas` vector (length == nprocs)
pt_mpi(neglogposterior, x0;
       nsteps=10^7,
       burnin=10^5,
       swap_every=100,
       betas=nothing,        # auto geometric ladder
       seed=Int64(round(time())),
       save_every=100,
       domains=domains,
       fracσ=fracσ,
       comm=comm)

MPI.Barrier(comm)
MPI.Finalize()

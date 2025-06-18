using DifferentialEquations
using Plots

gr()

# Define the system
function damped_oscillator!(du, u, p, t)
    x, v = u
    ω, γ = p
    du[1] = v
    du[2] = -ω^2 * x - γ * v
end

# Initial conditions and parameters
u0 = [1.0, 0.0]          # x(0)=1, v(0)=0
p = (1.0, 0.4)            # (ω, γ)
tspan = (0.0, 10.0)

# Create ODE problem
prob = ODEProblem(damped_oscillator!, u0, tspan, p)

sol = solve(prob, Tsit5(), saveat=0.01)

plt = plot(sol, xlabel="Time (t)", ylabel="x(t), v(t)", title="Damped Oscillator")
display(plt)

using LinearAlgebra

function aca_stamps()
    centers = [
        [65.0, 85.0, 115.0, log(2.0^2), log(3.0^2), log(4.0^2), 0.0, 0.0, 0.0],
        [115.0, 65.0, 85.0, log(4.0^2), log(2.0^2), log(3.0^2), 0.5, 0.0, -0.5],
        [85.0, 115.0, 65.0, log(3.0^2), log(4.0^2), log(2.0^2), -0.5, 0.5, 0.0],
    ]

    Σ = Diagonal([9.0, 9.0, 9.0, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0])

    weights = fill(1 / 3, 3)
    μ̄ = sum(weights[k] * centers[k] for k in 1:3)

    Σ_total = zeros(9, 9)
    for k in 1:3
        μk = centers[k]
        diff = μk .- μ̄
        Σ_total += weights[k] * (Matrix(Σ) + diff * diff')
    end

    println(μ̄)
    open("stamps0cov.txt", "w") do file
        write(file, "$Σ_total\n")
    end
    display(Σ_total)
end

aca_stamps()

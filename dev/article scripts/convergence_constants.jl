using GLMakie

multifactorial(v::AbstractArray) = prod(Base.factorial.(v))
multifactorial(t::Tuple) = prod(Base.factorial.(t))

# double factoriel n!! avec convention (-1)!! = 1
function double_factorial(n::Int)
    n <= 0 && return 1
    return prod(n:-2:1)
end


function homogeneous_polynom_constant(λ::Vector{Int})
    m = sum(λ)
    d = length(λ)
    λfact = multifactorial(λ)    # bornes pour k_i
    ranges = (0:(λ[i] ÷ 2) for i in 1:d)
    S = 0.0
    for ktup in Iterators.product(ranges...)
        k = collect(ktup)
        ksum = sum(k)
        # (2m - 2|k| - 3)!!
        dfact = double_factorial(2*m - 2*ksum - 3)
        # k!
        kfact = multifactorial(k)
        # (λ-2k)!
        λminus2k = λ .- 2 .* k
        denom_fact = multifactorial(λminus2k)
        term = dfact / (2.0^(ksum) * kfact * denom_fact)
        S += term
    end
    return λfact * S
end

fig = Figure()
d = 3
ax = Axis(fig[1, 1], title = "Homogeneous Polynomial Constants", xlabel = "λ", ylabel = "Constant Value")
ns = 0:20

function λ_1(n; d = 3)
    res = zeros(d) .|> Int
    res[1] = n
    res[2:end] .= 0
    return res
end

function λ_2(d = 3)
    res = [1 for i in 1:d]
    return res
end

constants_1 = [homogeneous_polynom_constant(λ_1(n; d = 3)) for n in ns]
lines!(ax, ns, constants_1, label = "λ = [n, 0, ⋯, 0], d = 3")

axislegend(ax, position = :lt)

fig |> display
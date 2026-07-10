using StaticArrays
using Inti
using GLMakie
using GuiggianiRichardsonDuffy

n = 20

x̂ = SVector(0.5, 0.5)

y1 = SVector(0.0, 0.0, 0.0)
y2 = SVector(2.0, 0.0, 0.0)
y3 = SVector(0.0, 1.0, 0.0)
y4 = SVector(1.0, 1.0, 0.0)

el = Inti.LagrangeSquare(y1, y2, y3, y4)

function in_angle_interval(θ, θ_min, θ_max)
    θ     = mod(θ, 2π)
    θ_min = mod(θ_min, 2π)
    θ_max = mod(θ_max, 2π)

    if θ_min <= θ_max
        θ_min < θ <= θ_max
    else
        θ > θ_min || θ <= θ_max
    end
end

function lagrange_basis(i, ρ_list)
    n = length(ρ_list)
    L = 1.0
    for j in 1:n
        if j != i
            L *= ρ_list[j] / (ρ_list[j] - ρ_list[i])
        end
    end
    return L
end

function W_minus_1(n)
    return 2*sum(1/i for i in 1:n)
end

function W_minus_2(n, ρ_func_max, θ)
    return 2/ρ_func_max(θ) * n * (n + 1)
end

function A_and_B_i(θ, i, x̂, el, n)
    ref_el = Inti.reference_domain(el)
    decompo = Inti.polar_decomposition(ref_el, x̂)
    x, _ = Inti.GaussLegendre(n)()
    x = collect([_x[1] for _x in x])
    id = findfirst(t -> in_angle_interval(θ, t[1], t[2]), decompo)
    _, _, ρ_func = decompo[id]
    rho_max = ρ_func(θ)
    ρ_i = rho_max * x[i]
    ρ_list = [ρ_func(θ) * x[j] for j in 1:n]
    c_i = lagrange_basis(i, ρ_list)
    P_n = sum(1/ρ_j for ρ_j in ρ_list)
    A_i = c_i * ρ_i^2
    B_i = c_i * ρ_i * (1 - ρ_i * P_n)
    return A_i, B_i
end

function divergent_parts(θ, x̂, el, n, i)
    ref_el = Inti.reference_domain(el)
    decompo = Inti.polar_decomposition(ref_el, x̂)
    x, _ = Inti.GaussLegendre(n)()
    x = collect([_x[1] for _x in x])
    id = findfirst(t -> in_angle_interval(θ, t[1], t[2]), decompo)
    _, _, ρ_func = decompo[id]
    W_m_1 = W_minus_1(n)
    W_m_2 = W_minus_2(n, ρ_func, θ)
    A, B = A_and_B_i(θ, i, x̂, el, n)
    # A_i pairs with W_-2 and B_i with W_-1, as in the weight() function above.
    return A * W_m_2 + B * W_m_1
end

function weight(θ, i, x̂, el, n)
    A, B = A_and_B_i(θ, i, x̂, el, n)
    W_m_1 = W_minus_1(n)
    x, w = Inti.GaussLegendre(n)()
    x = collect([_x[1] for _x in x])
    id = findfirst(t -> in_angle_interval(θ, t[1], t[2]), Inti.polar_decomposition(Inti.reference_domain(el), x̂))
    _, _, ρ_func = Inti.polar_decomposition(Inti.reference_domain(el), x̂)[id]
    W_m_2 = W_minus_2(n, ρ_func, θ)
    # Inti.GaussLegendre integrates over [0, 1] (weights sum to 1), so mapping to
    # [0, ρ̂] has Jacobian ρ̂ : w_i = ρ̂ * w_i^{[0,1]}  (NOT ρ̂/2, which is the [-1, 1] convention).
    w_i = ρ_func(θ) * w[i]
    # NB: β and γ from the Guiggiani decomposition are taken here as 1 and 0. Per the tex they
    # enter as A*(W_-2 + γ/β² + 1/ρ̂) and B*(ln|ρ̂/β| - W_-1); add them once available.
    return w_i - A * (W_m_2 + 1/ρ_func(θ)) + B * (log(ρ_func(θ)) - W_m_1)
end

function quadrature(θ, K, x̂, el, n)
    û = ξ -> 1.0
    decompo = Inti.polar_decomposition(Inti.reference_domain(el), x̂)
    x, _ = Inti.GaussLegendre(n)()
    x = collect([_x[1] for _x in x])
    id = findfirst(t -> in_angle_interval(θ, t[1], t[2]), decompo)
    _, _, ρ_func = decompo[id]
    F = GuiggianiRichardsonDuffy.polar_kernel_fun(K, el, û, x̂, 1)
    rho_i_list = [ρ_func(θ) * x[i] for i in 1:n]
    F_i_list = [F(ρ_i, θ) for ρ_i in rho_i_list]
    w_i_list = [weight(θ, i, x̂, el, n) for i in 1:n]
    return sum(F_i_list[i] * w_i_list[i] for i in 1:n)
end

op = Inti.Laplace(; dim = 3)
K = Inti.HyperSingularKernel(op)

N_points = 1000
fig = Figure()
axA = Axis(fig[1, 1], title = "A_i(θ)", xlabel = "θ", ylabel = "value")
axB = Axis(fig[1, 2], title = "B_i(θ)", xlabel = "θ", ylabel = "value")
axN = Axis(fig[2, 1], title = "Sign of A_i(θ) and B_i(θ) with i (n = $n)", xlabel = "θ", ylabel = "value")
ax_sum = Axis(fig[2, 2], title = "Sum of A_i(θ) and B_i(θ)", xlabel = "θ", ylabel = "value")
ax_quad = Axis(fig[1:3, 3], title = "Quadrature values", xlabel = "θ", ylabel = "value")
# vertical lines
decompo = Inti.polar_decomposition(Inti.reference_domain(el), x̂)
for (θ_min, θ_max, _) in decompo
    vlines!(ax_quad, [θ_min, θ_max], color = :black)
end
xs = range(0, 2π, length = N_points)
signAs = Float64[]
signBs = Float64[]
quads = [quadrature(θ, K, x̂, el, n) for θ in xs]
for i in 1:n
    A_and_B_i_values = [A_and_B_i(θ, i, x̂, el, n) for θ in xs]
    A_values = [val[1] for val in A_and_B_i_values]
    B_values = [val[2] for val in A_and_B_i_values]
    lines!(axA, xs, A_values, label = "A_i(θ), i = $i")
    lines!(axB, xs, B_values, label = "B_i(θ), i = $i")
    signA = sign(A_values[1])
    signB = sign(B_values[1])
    push!(signAs, signA)
    push!(signBs, signB)
end

lines!(ax_quad, xs, quads, label = "Quadrature values")

sumA = [sum([A_and_B_i(θ, i, x̂, el, n)[1] for i in 1:n]) for θ in xs]
sumB = [sum([A_and_B_i(θ, i, x̂, el, n)[2] for i in 1:n]) for θ in xs]
lines!(axN, 1:n, signAs, label = "Sign of A_i(θ)")
lines!(axN, 1:n, signBs, label = "Sign of B_i(θ)")
lines!(ax_sum, xs, sumA, label = "Sum of A_i(θ)")
lines!(ax_sum, xs, sumB, label = "Sum of B_i(θ)")
axislegend(axA)
axislegend(axB)
axislegend(axN)
axislegend(ax_quad)
axislegend(ax_sum)

ax_div = Axis(fig[3, 1], title = "Apparent arithmetic divergence", xlabel = "θ", ylabel = "value")
for i in 1:n
    div_values = [divergent_parts(θ, x̂, el, n, i) for θ in xs]
    lines!(ax_div, xs, div_values, label = "Divergent part for i = $i")
end

ax_weight = Axis(fig[3, 2], title = "Quadrature weights", xlabel = "θ", ylabel = "value")
for i in 1:n
    weight_values = [weight(θ, i, x̂, el, n) for θ in xs]
    lines!(ax_weight, xs, weight_values, label = "Weight for i = $i")
end
axislegend(ax_div)
axislegend(ax_weight)
display(fig)

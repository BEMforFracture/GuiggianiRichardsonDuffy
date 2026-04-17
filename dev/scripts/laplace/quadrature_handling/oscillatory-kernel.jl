using Inti
using StaticArrays
using GLMakie
using LinearAlgebra

## input parameters
η = SVector(0.5, 0.5)
ρ = 0.5

y¹ = SVector(0.0, 1.0, 0.0)
y² = SVector(1.0, 0.0, 0.0)
y³ = SVector(1.0, 1.0, 0.0)
y⁴ = SVector(0.0, 1.0, 0.0)
nodes = (y¹, y², y³, y⁴)
el = Inti.LagrangeSquare(nodes)

function A_vec(ρ, θ, Dτ, D²τ)
    uθ = SVector(cos(θ), sin(θ))
    A = Dτ * uθ
    B = ntuple(i -> transpose(uθ) * D²τ[i, :, :] * uθ, 3) |> SVector
    return A + ρ * B
end

function Dτ(ξ)
    return Inti.jacobian(el, ξ)
end

function D²τ(ξ)
    return Inti.hessian(el, ξ)
end

function A_vec(ρ, θ, ξ)
    return A_vec(ρ, θ, Dτ(ξ), D²τ(ξ))
end

function vᵢⱼ(ρ, θ, ξ, i, j)
    vecA = A_vec(ρ, θ, ξ)
    Aᵢ = vecA[i]
    Aⱼ = vecA[j]
    nA = norm(vecA)
    return Aᵢ * Aⱼ / nA^2
end

function integrate_gauss(ρ, ξ, i, j; rtol = eps(), atol = 1e-14, nmax = 1000, nmin = 3)
    n = 1
    ϵ = Inf
    prev_integral = 0.0
    while n <= nmax && (n < nmin || ϵ > rtol)
        quad_theta = Inti.GaussLegendre(n)
        integral = 2π * quad_theta() do (θ,)
            θ_ref = θ * 2π  # transform [0,1] to [0,2π]
            return vᵢⱼ(ρ, θ_ref, ξ, i, j)
        end
        ϵ_abs = abs(integral - prev_integral)
        ϵ_rel = abs(integral) > atol ? ϵ_abs / abs(integral) : ϵ_abs
        ϵ = ϵ_rel
        prev_integral = integral
        n += 1
    end
    if n > nmax
        @warn "Maximum quadrature order reached without convergence for i=$i, j=$j"
    end
    return prev_integral, n - 1
end

function integrate_trapezoid(ρ, ξ, i, j; rtol = eps(), atol = 1e-14, nmax = 1000, nmin = 10)
    N = 2
    ϵ = Inf
    prev_integral = 0.0
    while N <= nmax && (N < nmin || ϵ > rtol)
        quad_theta = Inti.Trapezoid(N)
        integral = 2π * quad_theta() do (θ,)
            θ_ref = θ * 2π
            return vᵢⱼ(ρ, θ_ref, ξ, i, j)
        end
        ϵ_abs = abs(integral - prev_integral)
        ϵ_rel = abs(integral) > atol ? ϵ_abs / abs(integral) : ϵ_abs
        ϵ = ϵ_rel
        prev_integral = integral
        N += 1
    end
    return prev_integral, N-1
end

function integrate(ρ, ξ, i, j, quad::Symbol; rtol = eps(), atol = 1e-14, nmax = 1000, nmin = nothing)
    # Set starting point and default nmin based on quadrature type
    if quad == :gauss
        n = 1
        nmin = isnothing(nmin) ? 3 : nmin
    elseif quad == :trapezoid
        n = 2  # Start at 2 for trapezoid to use actual trapezoidal rule
        nmin = isnothing(nmin) ? 10 : nmin  # Higher for trapezoid due to oscillatory functions
    else
        error("Unknown quadrature type: $quad")
    end
    
    ϵ = Inf
    prev_integral = 0.0
    while n <= nmax && (n < nmin || ϵ > rtol)
        if quad == :gauss
             quad_theta = Inti.GaussLegendre(n)
        elseif quad == :trapezoid
             quad_theta = Inti.Trapezoid(n)
        end
        integral = 2π * quad_theta() do (θ,)
            θ_ref = θ * 2π
            return vᵢⱼ(ρ, θ_ref, ξ, i, j)
        end
        ϵ_abs = abs(integral - prev_integral)
        ϵ_rel = abs(integral) > atol ? ϵ_abs / abs(integral) : ϵ_abs
        ϵ = ϵ_rel
        prev_integral = integral
        n += 1
    end
    return prev_integral, n - 1, ϵ
end

function visualize_v_vec(ρ, η; N = 1000, rtol = eps())
    quad_theta = Inti.GaussLegendre(10)

    I_g = zeros(3, 3)
    I_trap = zeros(3, 3)
    ns_g = zeros(Int, 3, 3)
    ns_trap = zeros(Int, 3, 3)
    for i in 1:3
        for j in 1:3
            res_g = integrate(ρ, η, i, j, :gauss; rtol = rtol)
            res_trap = integrate(ρ, η, i, j, :trapezoid; rtol = rtol)
            I_g[i, j] = res_g[1]
            ns_g[i, j] = res_g[2]
            I_trap[i, j] = res_trap[1]
            ns_trap[i, j] = res_trap[2]
            @info "Integral I[$i, $j] = $(I_g[i, j]) with $(ns_g[i, j]) gauss points for θ integration"
            @info "Integral I[$i, $j] = $(I_trap[i, j]) with $(ns_trap[i, j]) trapezoidal points for θ integration"
        end
    end
    θs = range(0, 2π, length=N)
    fig = Figure()
    for i in 1:3
        for j in 1:3
            ax_ij = Axis(fig[i, j], title="v_$i$j(ρ=$ρ, η=$(η))")
            v_vecs = [vᵢⱼ(ρ, θ, η, i, j) for θ in θs]
            lines!(ax_ij, θs, v_vecs, label="v_$i$j")
            text!(
                ax_ij,
                "Integral: $(I_g[i, j])\nNumber of Gauss points: $(ns_g[i, j])";
                position = (0.02, 0.98),
                space = :relative,
                align = (:left, :top),
            )
            text!(
                ax_ij,
                "Integral (trapezoid): $(I_trap[i, j])\nNumber of trapezoidal points: $(ns_trap[i, j])";
                position = (0.02, 0.02),
                space = :relative,
                align = (:left, :bottom),
            )
        end
    end
    return fig
end

function n_points_vs_precision(ρ, ξ, i, j; ϵs = [1e-3, 1e-4, 1e-5, 1e-6])
    data_g = Dict()
    data_trap = Dict()
    for ϵ in ϵs
        _, n_g = integrate(ρ, ξ, i, j, :gauss; rtol = ϵ)
        _, n_trap = integrate(ρ, ξ, i, j, :trapezoid; rtol = ϵ)
        data_g[ϵ] = n_g
        data_trap[ϵ] = n_trap
    end
    return data_g, data_trap
end

function plot_n_points_vs_precision(data_g, data_trap)
    ϵs = collect(keys(data_g))
    n_gs = collect(values(data_g))
    n_traps = collect(values(data_trap))
    fig = Figure()
    for i in 1:3
        for j in 1:3
             ax = Axis(fig[i, j], title="Number of Gauss points vs Precision for v_$i$j")
             lines!(ax, ϵs, n_gs[i, j], label="v_$i$j", xscale=:log10, xreversed=true)
             lines!(ax, ϵs, n_traps[i, j], label="v_$i$j (Trapezoid)", xscale=:log10, xreversed=true)
             axislegend(ax)
        end
    end
    return fig
end

function plot_n_points_vs_precision(ρ, ξ)
    data = [n_points_vs_precision(ρ, ξ, i, j; ϵs = ϵs) for i in 1:3, j in 1:3]
    fig = Figure()
    for i in 1:3
        for j in 1:3
            data_g, data_trap = data[i, j]
            ordered_ϵs = sort(collect(keys(data_g)); rev = true)
            n_gs = [data_g[ϵ] for ϵ in ordered_ϵs]
            n_traps = [data_trap[ϵ] for ϵ in ordered_ϵs]

            ax = Axis(fig[i, j], title="Number of points vs precision for v_$i$j")
            lines!(ax, ordered_ϵs, n_gs, label="Gauss")
            lines!(ax, ordered_ϵs, n_traps, label="Trapezoid")
            ax.xscale = log10
            ax.xreversed = true
            axislegend(ax, position = :lt)
        end
    end
    return fig
end

## run the visualization for a specific η and ρ

fig = visualize_v_vec(ρ, η)
window_1 = display(GLMakie.Screen(), fig)

ϵs = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]

fig = plot_n_points_vs_precision(ρ, η)
window_2 = display(GLMakie.Screen(), fig)
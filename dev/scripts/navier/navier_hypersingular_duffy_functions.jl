import GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using LinearAlgebra
using GLMakie
using ForwardDiff

# INPUTS
quad1D = Inti.GaussLegendre(2)
x̂ = quad1D.nodes[1][1]
x̂ = SVector(x̂, x̂)
# x̂ = SVector(0.3, 0.3) # source point in reference coordinates
ori = 1               # element orientation

E = 210e9
ν = 0.3
μ = E / (2 * (1 + ν))
λ = E * ν / ((1 + ν) * (1 - 2ν))

# Quadrature parameters
n_a = 5
quad_a = Inti.GaussLegendre(n_a)
target_rel_tol = 1e-8
nmin_quad = 2
nmax_quad = 120

# Setup element
δ = 0.5
z = 0.0
y¹ = SVector(-1.0, -1.0, z)
y² = SVector(1.0 + δ, -1.0, z)
y³ = SVector(-1.0, 1.0, z)
y⁴ = SVector(1.0 - δ, 1.0, z)
nodes = (y¹, y², y³, y⁴)

el = Inti.LagrangeSquare(nodes)
ref_domain = Inti.reference_domain(el)
û = ξ -> 1.0

# Kernel setup
op = Inti.Elastostatic(; μ = μ, λ = λ, dim = 3)
K_base = Inti.HyperSingularKernel(op)
SK = GRD.SplitKernel(K_base)

x = el(x̂)
Dτ = Inti.jacobian(el, x̂)
nx = Inti._normal(Dτ, ori)
D²τ = Inti.hessian(el, x̂)
qx = (coords = x, normal = nx)
N = length(x)

function return_vertices(τ)
	if τ == 1
		return SVector(1.0, 0.0), SVector(1.0, 1.0)
	elseif τ == 2
		return SVector(1.0, 1.0), SVector(0.0, 1.0)
	elseif τ == 3
		return SVector(0.0, 1.0), SVector(0.0, 0.0)
	else
		return SVector(0.0, 0.0), SVector(1.0, 0.0)
	end
end

function duffy_decomposition(::Inti.ReferenceSquare)
	ξᴵ, ξᴵᴵ = return_vertices(1)
	_, ξᴵᴵᴵ = return_vertices(2)
	_, ξᴵⱽ = return_vertices(3)
	return (
		(ξᴵ, ξᴵᴵ, 1),
		(ξᴵᴵ, ξᴵᴵᴵ, 2),
		(ξᴵᴵᴵ, ξᴵⱽ, 3),
		(ξᴵⱽ, ξᴵ, 4),
	)
end

function local_geometry_data(ξᴵ, ξᴵᴵ, b)
	c = b * (ξᴵᴵ - ξᴵ) + ξᴵ - x̂
	A = Dτ * c
	nA = norm(A)
	B = ntuple(i -> transpose(c) * D²τ[i, :, :] * c, N) |> SVector
	β = 1 / nA
	γ_over_β_squared = -(A ⋅ B) / nA^2
	return c, A, B, β, γ_over_β_squared
end

function _func(a, c, A, B, surface)
	ŷ = x̂ + a * c
	jac_y = Inti.jacobian(el, ŷ)
	ny = Inti._normal(jac_y, ori)
	y = el(ŷ)
	qy = (coords = y, normal = ny)
	μy = Inti._integration_measure(jac_y)

	AB = A + a / 2 * B
	Â = AB / norm(AB)
	_, K̂ = SK(qx, qy, Â)
	v = û(ŷ)

	return K̂ * v * surface * μy / norm(AB)^3
end

function singular_coeffs(f)
	f₋₂ = f(0.0)
	f₋₁ = similar(f₋₂)
	for I in eachindex(f₋₂)
		f₋₁[I] = ForwardDiff.derivative(a -> f(a)[I], 0.0)
	end
	return f₋₂, f₋₁
end

function G₁(ξᴵ, ξᴵᴵ, b)
	surface = (ξᴵ[1] - x̂[1]) * (ξᴵᴵ[2] - ξᴵ[2]) - (ξᴵᴵ[1] - ξᴵ[1]) * (ξᴵ[2] - x̂[2])
	c, A, B, _, _ = local_geometry_data(ξᴵ, ξᴵᴵ, b)

	f = a -> _func(a, c, A, B, surface)
	f₋₂, f₋₁ = singular_coeffs(f)

	return quad_a() do (a,)
		1 / a^2 * (f(a) - f₋₂ - a * f₋₁)
	end
end

function G₂(ξᴵ, ξᴵᴵ, b)
	surface = (ξᴵ[1] - x̂[1]) * (ξᴵᴵ[2] - ξᴵ[2]) - (ξᴵᴵ[1] - ξᴵ[1]) * (ξᴵ[2] - x̂[2])
	c, A, B, β, γ_over_β_squared = local_geometry_data(ξᴵ, ξᴵᴵ, b)

	f = a -> _func(a, c, A, B, surface)
	f₋₂, f₋₁ = singular_coeffs(f)

	return -f₋₁ * log(abs(β)) - f₋₂ * (γ_over_β_squared + 1)
end

function dG₁(ξᴵ, ξᴵᴵ, b)
	G = G₁(ξᴵ, ξᴵᴵ, b)
	dG = similar(G)
	for I in eachindex(G)
		dG[I] = ForwardDiff.derivative(t -> G₁(ξᴵ, ξᴵᴵ, t)[I], b)
	end
	return dG
end

function dG₂(ξᴵ, ξᴵᴵ, b)
	G = G₂(ξᴵ, ξᴵᴵ, b)
	dG = similar(G)
	for I in eachindex(G)
		dG[I] = ForwardDiff.derivative(t -> G₂(ξᴵ, ξᴵᴵ, t)[I], b)
	end
	return dG
end

function integrate_gauss_interval(f, a, b, n)
	quad = Inti.GaussLegendre(n)
	mid = (a + b) / 2
	half = (b - a) / 2
	return quad() do (t,)
		half * f(mid + half * t)
	end
end

function min_quad_points_successive(f, a, b; tol = 1e-8, nmin = 2, nmax = 120)
	I_prev = integrate_gauss_interval(f, a, b, nmin)
	last_rel = Inf
	for n in (nmin + 1):nmax
		I_curr = integrate_gauss_interval(f, a, b, n)
		last_rel = norm(I_curr - I_prev) / max(norm(I_curr), eps())
		if last_rel < tol
			return n, last_rel
		end
		I_prev = I_curr
	end
	return nmax, last_rel
end

Nplot = 1000
bs = range(0.0, stop = 1.0, length = Nplot)
decompo = duffy_decomposition(ref_domain)

fig = Figure(size = (2000, 900))
colors = [[:blue, :orange, :green], [:red, :purple, :brown], [:pink, :gray, :cyan], [:black, :teal, :goldenrod]]

for (k, (ξᴵ, ξᴵᴵ, τ)) in enumerate(decompo)
	ax1 = Axis(fig[k, 1]; xlabel = "b", ylabel = "G₁(b)", title = k == 1 ? "Secteur τ = $τ | tol=$(target_rel_tol)" : "Secteur τ = $τ")
	ax2 = Axis(fig[k, 2]; xlabel = "b", ylabel = "G₂(b)", title = "Secteur τ = $τ")
	ax3 = Axis(fig[k, 3]; xlabel = "b", ylabel = "G₁'(b)", title = "Secteur τ = $τ")
	ax4 = Axis(fig[k, 4]; xlabel = "b", ylabel = "G₂'(b)", title = "Secteur τ = $τ")

	nb_G1, relb_G1 = min_quad_points_successive(
		b -> G₁(ξᴵ, ξᴵᴵ, b),
		0.0,
		1.0;
		tol = target_rel_tol,
		nmin = nmin_quad,
		nmax = nmax_quad,
	)
	nb_G2, relb_G2 = min_quad_points_successive(
		b -> G₂(ξᴵ, ξᴵᴵ, b),
		0.0,
		1.0;
		tol = target_rel_tol,
		nmin = nmin_quad,
		nmax = nmax_quad,
	)
	println("Secteur τ=$τ | tol=$target_rel_tol | n_b(G1)=$nb_G1 (rel=$relb_G1) | n_b(G2)=$nb_G2 (rel=$relb_G2)")

	for i in 1:3
		for j in 1:3
			g1 = [G₁(ξᴵ, ξᴵᴵ, b)[i, j] for b in bs]
			g2 = [G₂(ξᴵ, ξᴵᴵ, b)[i, j] for b in bs]
			dg1 = [dG₁(ξᴵ, ξᴵᴵ, b)[i, j] for b in bs]
			dg2 = [dG₂(ξᴵ, ξᴵᴵ, b)[i, j] for b in bs]
			col = colors[k][j]
			lines!(ax1, bs, g1, color = col, label = "G₁[$i,$j]")
			lines!(ax2, bs, g2, color = col, label = "G₂[$i,$j]")
			lines!(ax3, bs, dg1, color = col, label = "G₁'[$i,$j]")
			lines!(ax4, bs, dg2, color = col, label = "G₂'[$i,$j]")
		end
	end
	text!(ax1, 0.02, 0.98; text = "nG1=$nb_G1", space = :relative, align = (:left, :top), fontsize = 10, color = :black)
	text!(ax2, 0.02, 0.98; text = "nG2=$nb_G2", space = :relative, align = (:left, :top), fontsize = 10, color = :black)
end

display(GLMakie.Screen(), fig)
# GLMakie.save("./dev/figures/navier/navier_hypersingular_duffy_functions.png", fig)

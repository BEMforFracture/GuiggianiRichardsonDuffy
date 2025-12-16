import GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using LinearAlgebra
using ForwardDiff
using GLMakie

# Configuration
n_a = 10
n_b = 50
quad_a = Inti.GaussLegendre(n_a)
quad_b = Inti.GaussLegendre(n_b)

n_rho = 10
n_theta = 40
quad_rho = Inti.GaussLegendre(n_rho)
quad_theta = Inti.GaussLegendre(n_theta)

ϵs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

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

# Density function
û = ξ -> 1.0

# Kernel setup
K_base = Inti.HyperSingularKernel(Inti.Laplace(dim = 3))
SK = GRD.SplitKernel(K_base)

s = Inti.singularity_order(SK)
S = s + 1
sorder = Val(S)

function return_vertices(τ)
	if τ == 1
		return SVector(1, 0), SVector(1, 1)
	elseif τ == 2
		return SVector(1, 1), SVector(0, 1)
	elseif τ == 3
		return SVector(0, 1), SVector(0, 0)
	else
		return SVector(0, 0), SVector(1, 0)
	end
end

function duffy_decomposition(::Inti.ReferenceSquare)
	ξᴵ, ξᴵᴵ = return_vertices(1)
	_, ξᴵᴵᴵ = return_vertices(2)
	_, ξᴵⱽ = return_vertices(3)
	return (ξᴵ, ξᴵᴵ, 1),
	(ξᴵᴵ, ξᴵᴵᴵ, 2),
	(ξᴵᴵᴵ, ξᴵⱽ, 3),
	(ξᴵⱽ, ξᴵ, 4)
end

# Initialize accumulator
# T = SMatrix{3, 3, Float64}

function duffy_guiggiani_singular_integral(SK, û, x̂, el, ori, quad_a, quad_b)
	x = el(x̂)
	Dτ = Inti.jacobian(el, x̂)
	nx = Inti._normal(Dτ, ori)
	D²τ = Inti.hessian(el, x̂)
	qx = (coords = x, normal = nx)
	N = length(x)

	T = Float64
	duffy = zero(T)
	# Integrate over each angular sector
	for (ξᴵ, ξᴵᴵ, τ) in duffy_decomposition(ref_domain)
		surface = (ξᴵ[1] - x̂[1]) ⋅ (ξᴵᴵ[2] - ξᴵ[2]) - (ξᴵᴵ[1] - ξᴵ[1]) ⋅ (ξᴵ[2] - x̂[2])

		_func = (a, b, c, A, B) -> begin
			ŷ = x̂ + a * c
			jac_y = Inti.jacobian(el, ŷ)
			ny = Inti._normal(jac_y, ori)
			y = el(ŷ)
			qy = (coords = y, normal = ny)
			μ = Inti._integration_measure(jac_y)
			AB = A + a / 2 * B
			Â = AB / norm(AB)
			_, K̂ = SK(qx, qy, Â) <
					v = û(ŷ)
			map(v -> K̂ * v, v) * surface * μ / norm(AB)^3
		end
		I_ab = quad_b() do (b,)
			c = b * (ξᴵᴵ - ξᴵ) + ξᴵ - x̂
			A = Dτ * c
			nA = norm(A)
			B = ntuple(i -> transpose(c) * D²τ[i, :, :] * c, N) |> SVector
			β = 1 / nA
			γ_over_β_squared = -A ⋅ B / nA^2

			coeffs_ = func -> begin
				F2 = func(0.0)
				F1 = ForwardDiff.derivative(func, 0.0)
				return (F1, F2)
			end

			func = t -> _func(t, b, c, A, B)

			f₋₁, f₋₂ = coeffs_(func)

			I_a = quad_a() do (a,)
				return 1 / a^2 * (func(a) - f₋₂ - a * f₋₁)
			end

			return I_a - f₋₁ * log(abs(β)) - f₋₂ * (γ_over_β_squared + 1)
		end
		duffy += I_ab
	end
	return duffy
end

method = GRD.AutoDiffExpansion()

qorder = 3
quad = Inti.GaussLegendre(qorder)
prod_quad = Inti.TensorProductQuadrature(quad, quad)

function _nodes(TPQ::Inti.TensorProductQuadrature{2, Tuple{Inti.GaussLegendre{N, T}, Inti.GaussLegendre{N, T}}}) where {N, T}
	qnodes = [T[getindex.(qrule.nodes, 1)...] for qrule in TPQ.quads1d]
	q1 = qnodes[1]
	q2 = qnodes[2]
	nds = [SVector(x, y) for (x, y) in Iterators.product(q1, q2)]
	return SVector{length(nds), SVector{2, T}}(nds...)
end

qnodes = _nodes(prod_quad)

nθ_data = [Dict{Float64, Int64}() for _ in 1:9]
for i in 1:9
	x̂ = qnodes[i]
	I_exp = GRD.hypersingular_laplace_integral_on_plane_element(el(x̂), el)

	for ϵ in ϵs
		nθ = 1
		quad_θ = Inti.GaussLegendre(nθ)
		I_num = GRD.guiggiani_singular_integral(K_base, û, x̂, el, 1, quad_rho, quad_θ, method)
		error = norm(I_num - I_exp) / norm(I_exp)

		while error > ϵ
			nθ += 1
			quad_θ = Inti.GaussLegendre(nθ)
			I_num = GRD.guiggiani_singular_integral(K_base, û, x̂, el, 1, quad_rho, quad_θ, method)
			error = norm(I_num - I_exp) / norm(I_exp)
		end

		nθ_data[i][ϵ] = nθ
	end
end

nρ_data = [Dict{Float64, Int64}() for _ in 1:9]
for i in 1:9
	x̂ = qnodes[i]
	I_exp = GRD.hypersingular_laplace_integral_on_plane_element(el(x̂), el)

	for ϵ in ϵs
		nρ = 1
		quad_ρ = Inti.GaussLegendre(nρ)
		I_num = GRD.guiggiani_singular_integral(K_base, û, x̂, el, 1, quad_ρ, quad_theta, method)
		error = norm(I_num - I_exp) / norm(I_exp)

		while error > ϵ
			nρ += 1
			quad_ρ = Inti.GaussLegendre(nρ)
			I_num = GRD.guiggiani_singular_integral(K_base, û, x̂, el, 1, quad_ρ, quad_theta, method)
			error = norm(I_num - I_exp) / norm(I_exp)
		end

		nρ_data[i][ϵ] = nρ
	end
end

na_data = [Dict{Float64, Int64}() for _ in 1:9]
for i in 1:9
	x̂ = qnodes[i]
	I_exp = GRD.hypersingular_laplace_integral_on_plane_element(el(x̂), el)

	for ϵ in ϵs
		n_a = 1
		_quad_a = Inti.GaussLegendre(n_a)
		I_num = duffy_guiggiani_singular_integral(SK, û, x̂, el, 1, _quad_a, quad_b)
		error = norm(I_num - I_exp) / norm(I_exp)

		while error > ϵ
			n_a += 1
			_quad_a = Inti.GaussLegendre(n_a)
			I_num = duffy_guiggiani_singular_integral(SK, û, x̂, el, 1, _quad_a, quad_b)
			error = norm(I_num - I_exp) / norm(I_exp)
		end
		na_data[i][ϵ] = n_a
	end
end

nb_data = [Dict{Float64, Int64}() for _ in 1:9]
for i in 1:9
	x̂ = qnodes[i]
	I_exp = GRD.hypersingular_laplace_integral_on_plane_element(el(x̂), el)

	for ϵ in ϵs
		n_b = 1
		_quad_b = Inti.GaussLegendre(n_b)
		I_num = duffy_guiggiani_singular_integral(SK, û, x̂, el, 1, quad_a, _quad_b)
		error = norm(I_num - I_exp) / norm(I_exp)

		while error > ϵ
			n_b += 1
			_quad_b = Inti.GaussLegendre(n_b)
			I_num = duffy_guiggiani_singular_integral(SK, û, x̂, el, 1, quad_a, _quad_b)
			error = norm(I_num - I_exp) / norm(I_exp)
		end
		nb_data[i][ϵ] = n_b
	end
end

# Créer une grille 3x3 de subplots
fig = Figure(size = (1400, 1400))

for i in 1:9
	# Calculer la position dans la grille (ligne, colonne)
	row = div(i - 1, 3) + 1
	col = mod(i - 1, 3) + 1

	ax = Axis(fig[row, col],
		xlabel = row == 3 ? "Précision ε" : "",
		ylabel = col == 1 ? "nombre de points de Gauss" : "",
		title = "x̂[$i] = $(qnodes[i])",
		xscale = log10,
		xreversed = true,
		xticks = (ϵs,
			string.(ϵs)),
	)

	# Extraire et trier les données
	sorted_data_theta = sort(collect(nθ_data[i]), by = first, rev = true)
	epsilons_theta = [ϵ for (ϵ, _) in sorted_data_theta]
	nthetas = [Int(nθ) for (_, nθ) in sorted_data_theta]

	sorted_data_rho = sort(collect(nρ_data[i]), by = first, rev = true)
	epsilons_rho = [ϵ for (ϵ, _) in sorted_data_rho]
	nrhos = [Int(nρ) for (_, nρ) in sorted_data_rho]

	sorted_data_a = sort(collect(na_data[i]), by = first, rev = true)
	epsilons_a = [ϵ for (ϵ, _) in sorted_data_a]
	nas = [Int(n_a) for (_, n_a) in sorted_data_a]

	sorted_data_b = sort(collect(nb_data[i]), by = first, rev = true)
	epsilons_b = [ϵ for (ϵ, _) in sorted_data_b]
	nbs = [Int(n_b) for (_, n_b) in sorted_data_b]

	lines!(ax, epsilons_theta, nthetas, label = "Polar : θ", color = :black)
	lines!(ax, epsilons_rho, nrhos, label = "Polar : ρ", color = :red)
	lines!(ax, epsilons_a, nas, color = :green, label = "Duffy : a")
	lines!(ax, epsilons_b, nbs, color = :blue, label = "Duffy : b")

	axislegend(ax, position = :lt)
end

display(fig)

# GLMakie.save("dev/figures/laplace/laplace_hypersingular_duffy_polar_comparison.png", fig)

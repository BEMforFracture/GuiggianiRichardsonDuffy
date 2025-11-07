import GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using LinearAlgebra
using GLMakie
using DelaunayTriangulation
using GeometryBasics
using Statistics

Nθ = 1:40
Nρ = 1:20

qorder = 3
quad = Inti.GaussLegendre(qorder)
prod_quad = Inti.TensorProductQuadrature(quad, quad)

function nodes(TPQ::Inti.TensorProductQuadrature{2, Tuple{Inti.GaussLegendre{N, T}, Inti.GaussLegendre{N, T}}}) where {N, T}
	qnodes = [T[getindex.(qrule.nodes, 1)...] for qrule in TPQ.quads1d]
	q1 = qnodes[1]
	q2 = qnodes[2]
	nds = [SVector(x, y) for (x, y) in Iterators.product(q1, q2)]
	return SVector{length(nds), SVector{2, T}}(nds...)
end

qnodes = nodes(prod_quad)

method = GRD.AutoDiffExpansion()

K = Inti.HyperSingularKernel(Inti.Laplace(dim = 3))

# Setup element
δ = 0.5
z = 0.0
y¹ = SVector(-1.0, -1.0, z)
y² = SVector(1.0 + δ, -1.0, z)
y³ = SVector(-1.0, 1.0, z)
y⁴ = SVector(1.0 - δ, 1.0, z)
nnodes = (y¹, y², y³, y⁴)
el = Inti.LagrangeSquare(nnodes)

# Density function
û = ξ -> 1.0

errors = [Dict{Tuple{Int64, Int64}, Float64}() for _ in 1:length(qnodes)]

for nθ in Nθ, nρ in Nρ
	quad_θ = Inti.GaussLegendre(nθ)
	quad_ρ = Inti.GaussLegendre(nρ)

	for (i, x̂) in enumerate(qnodes)
		I_exp = GRD.hypersingular_laplace_integral_on_plane_element(el(x̂), el)
		I_num = GRD.guiggiani_singular_integral(K, û, x̂, el, 1, quad_ρ, quad_θ, method)
		ϵ = norm(I_num - I_exp) / norm(I_exp)
		errors[i][(nρ, nθ)] = ϵ  # Inverser la clé et la valeur
	end
end

# Préparer les données pour tous les points
ρ_vals = sort(unique([ρ for d in errors for ((ρ, θ), _) in d]))
θ_vals = sort(unique([θ for d in errors for ((ρ, θ), _) in d]))

# Créer une grille 3x3 de subplots
fig = Figure(size = (1400, 1400))

# Collecter toutes les erreurs pour la colorbar commune
all_errors = Float64[]
for d in errors
	append!(all_errors, collect(values(d)))
end

for i in 1:9
	data = errors[i]

	Z = fill(NaN, length(ρ_vals), length(θ_vals))
	for ((ρ, θ), err) in data
		iρ = findfirst(==(ρ), ρ_vals)
		iθ = findfirst(==(θ), θ_vals)
		Z[iρ, iθ] = err
	end

	# Calculer la position dans la grille (ligne, colonne)
	row = div(i - 1, 3) + 1
	col = mod(i - 1, 3) + 1

	ax = Axis(fig[row, col],
		xlabel = "nρ",
		ylabel = "nθ",
		title = "x̂[$i] = $(qnodes[i])",
	)

	hm = heatmap!(ax, ρ_vals, θ_vals, log10.(abs.(Z));
		colormap = :viridis,
		nan_color = :white,
	)
end

# Ajouter une colorbar commune à droite
Colorbar(fig[:, 4], limits = extrema(log10.(abs.(all_errors))),
	colormap = :viridis, label = "log₁₀(erreur)")

fig

GLMakie.save("dev/figures/laplace/laplace_hypersingular_fixed_accuracy_number_of_quad_points.png", fig)

nρ = 5
quad_ρ = Inti.GaussLegendre(nρ)

ϵs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]

# Calculer nθ pour chaque point et chaque précision
nθ_data = [Dict{Float64, Int64}() for _ in 1:9]

for i in 1:9
	x̂ = qnodes[i]
	I_exp = GRD.hypersingular_laplace_integral_on_plane_element(el(x̂), el)

	for ϵ in ϵs
		nθ = 1
		quad_θ = Inti.GaussLegendre(nθ)
		I_num = GRD.guiggiani_singular_integral(K, û, x̂, el, 1, quad_ρ, quad_θ, method)
		error = norm(I_num - I_exp) / norm(I_exp)

		while error > ϵ
			nθ += 1
			quad_θ = Inti.GaussLegendre(nθ)
			I_num = GRD.guiggiani_singular_integral(K, û, x̂, el, 1, quad_ρ, quad_θ, method)
			error = norm(I_num - I_exp) / norm(I_exp)
		end

		nθ_data[i][ϵ] = nθ
	end
end

# Créer une grille 3x3 de subplots
fig = Figure(size = (1400, 1400))

for i in 1:9
	# Calculer la position dans la grille (ligne, colonne)
	row = div(i - 1, 3) + 1
	col = mod(i - 1, 3) + 1

	ax = Axis(fig[row, col],
		xlabel = "Précision ε",
		ylabel = "nθ requis",
		title = "x̂[$i] = $(qnodes[i])",
		xscale = log10,
		xreversed = true,
		xticks = ([1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-0],
			["10⁻¹²", "10⁻¹⁰", "10⁻⁸", "10⁻⁶", "10⁻⁴", "10⁻²", "10⁻⁰"]),
	)

	# Extraire et trier les données
	sorted_data = sort(collect(nθ_data[i]), by = first, rev = true)
	epsilons = [ϵ for (ϵ, _) in sorted_data]
	nthetas = [Int(nθ) for (_, nθ) in sorted_data]

	lines!(ax, epsilons, nthetas)
end

fig

GLMakie.save("dev/figures/laplace/laplace_hypersingular_fixed_accuracy_theta.png", fig)

import GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using LinearAlgebra
using GLMakie
using DelaunayTriangulation
using GeometryBasics

# Configuration
N = 10  # Number of points in each direction (total N² points)

# Richardson parameters
rich_params = GRD.RichardsonParams(
	first_contract = 1e-2,
	contract = 0.5,
	breaktol = Inf,
	atol = 0.0,
	rtol = 0.0,
	maxeval = 5,
)

# Setup element
δ = 0.5
z = 0.0
y¹ = SVector(-1.0, -1.0, z)
y² = SVector(1.0 + δ, -1.0, z)
y³ = SVector(-1.0, 1.0, z)
y⁴ = SVector(1.0 - δ, 1.0, z)
nodes = (y¹, y², y³, y⁴)
el = Inti.LagrangeSquare(nodes)

# Density function
û = ξ -> 1.0

# Kernel setup
K_base = Inti.HyperSingularKernel(Inti.Laplace(dim = 3))
K = GRD.SplitKernel(K_base)

# Methods to test
methods = [
	("AutoDiff", GRD.AutoDiffExpansion(), K),
	("SemiRichardson", GRD.SemiRichardsonExpansion(rich_params), K),
	("FullRichardson", GRD.FullRichardsonExpansion(rich_params), K),
]

# Create grid of points (avoid boundaries)
ξ_min = 1 / N
ξ_max = 1 - 1 / N
ξ_range = range(ξ_min, ξ_max; length = N)
η_range = range(ξ_min, ξ_max; length = N)

# Storage for errors
errors = Dict(
	m[1] => (F₋₂ = zeros(N, N), F₋₁ = zeros(N, N))
	for m in methods
)

# Angular samples for evaluation
N_θ = 1000
θs = range(0, 2π; length = N_θ)

@info "Computing Laurent coefficients for $(N^2) points with $(length(methods)) methods..."

# Loop over all grid points
for (i, ξ) in enumerate(ξ_range)
	for (j, η) in enumerate(η_range)
		x̂ = SVector(ξ, η)

		# Analytical reference
		ℒ_ana = GRD.laurents_coeffs(K_base, el, û, x̂, GRD.AnalyticalExpansion())
		vals_ana = [ℒ_ana(θ) for θ in θs]
		F₋₂_ana = [v[1] for v in vals_ana]
		F₋₁_ana = [v[2] for v in vals_ana]

		# Compute errors for each method
		for (method_name, method, K_to_use) in methods
			ℒ = GRD.laurents_coeffs(K_to_use, el, û, x̂, method)
			vals_test = [ℒ(θ) for θ in θs]
			F₋₂_test = [v[1] for v in vals_test]
			F₋₁_test = [v[2] for v in vals_test]

			# Relative errors in norm 2
			error_F₋₂ = norm(F₋₂_ana - F₋₂_test, 2) / norm(F₋₂_ana, 2)
			error_F₋₁ = norm(F₋₁_ana - F₋₁_test, 2) / norm(F₋₁_ana, 2)

			errors[method_name].F₋₂[i, j] = error_F₋₂
			errors[method_name].F₋₁[i, j] = error_F₋₁
		end

		if (i - 1) * N + j % 100 == 0
			@info "Progress: $((i-1)*N + j)/$(N^2) points computed"
		end
	end
end

@info "Computation complete. Creating plots..."

# Get physical coordinates
x_flat = [el(SVector(ξ, η))[1] for ξ in ξ_range for η in η_range]
y_flat = [el(SVector(ξ, η))[2] for ξ in ξ_range for η in η_range]

# Create Delaunay triangulation
points = [(x_flat[i], y_flat[i]) for i in 1:length(x_flat)]
tri = triangulate(points)
connectivity = [TriangleFace(triangle...) for triangle in each_solid_triangle(tri)]

# Element boundary
corners_x = [y¹[1], y²[1], y⁴[1], y³[1], y¹[1]]
corners_y = [y¹[2], y²[2], y⁴[2], y³[2], y¹[2]]

# Create figure with 2 rows × 3 columns
fig = Figure(; size = (2100, 1000))

# Plot F₋₂ errors (first row)
for (col, (method_name, _, _)) in enumerate(methods)
	ax = Axis(fig[1, col*2-1];
		xlabel = "x",
		ylabel = "y",
		title = "F₋₂: $method_name",
		aspect = DataAspect(),
	)

	errors_flat = vec(errors[method_name].F₋₂')

	hm = mesh!(ax,
		Point2f.(x_flat, y_flat),
		connectivity,
		color = log10.(errors_flat),
		colormap = :turbo,
		shading = NoShading,
	)

	Colorbar(fig[1, col*2], hm; label = "log₁₀(Error)", width = 15)
	lines!(ax, corners_x, corners_y; color = :black, linewidth = 2)
end

# Plot F₋₁ errors (second row)
for (col, (method_name, _, _)) in enumerate(methods)
	ax = Axis(fig[2, col*2-1];
		xlabel = "x",
		ylabel = "y",
		title = "F₋₁: $method_name",
		aspect = DataAspect(),
	)

	errors_flat = vec(errors[method_name].F₋₁')

	hm = mesh!(ax,
		Point2f.(x_flat, y_flat),
		connectivity,
		color = log10.(errors_flat),
		colormap = :turbo,
		shading = NoShading,
	)

	Colorbar(fig[2, col*2], hm; label = "log₁₀(Error)", width = 15)
	lines!(ax, corners_x, corners_y; color = :black, linewidth = 2)
end

# Add title
Label(fig[0, :],
	"Laplace Hypersingular Laurent Coefficients Error (maxeval = $(rich_params.maxeval))",
	fontsize = 20, font = :bold)

# Print statistics
for (method_name, _, _) in methods
	@info "Method: $method_name"
	@info "  Min error F₋₂: $(minimum(errors[method_name].F₋₂))"
	@info "  Max error F₋₂: $(maximum(errors[method_name].F₋₂))"
	@info "  Min error F₋₁: $(minimum(errors[method_name].F₋₁))"
	@info "  Max error F₋₁: $(maximum(errors[method_name].F₋₁))"
end

display(fig)
# GLMakie.save("./dev/figures/laplace/laplace_hypersingular_laurent_coeffs_convergence_map.png", fig)

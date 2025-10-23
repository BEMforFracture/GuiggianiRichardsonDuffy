import GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using LinearAlgebra
using GLMakie
using DelaunayTriangulation
using GeometryBasics

# INPUTS

N = 10 # number of points in each direction (total N² points)

### Richardson extrapolation parameters
maxeval = 5
rtol = 0.0
atol = 0.0
contract = 0.5
first_contract = 1e-2
breaktol = Inf

# END INPUTS

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

K = GRD.SplitLaplaceHypersingular

# Create grid of points on the reference element [0, 1]²
ξ_min = 1 / N
ξ_max = 1 - 1 / N
ξ_range = range(ξ_min, ξ_max, length = N)  # avoid boundaries
η_range = range(ξ_min, ξ_max, length = N)

# Dictionary to store errors for each method
methods = [:auto_diff, :semi_richardson, :full_richardson]
errors = Dict(
	method => (F₋₂ = zeros(N, N), F₋₁ = zeros(N, N))
	for method in methods
)

# Number of angular samples for Laurent coefficient evaluation
N_θ = 1000
θs = range(0, 2π, length = N_θ)

@info "Computing Laurent coefficients for $(N^2) points with $(length(methods)) methods..."

# Loop over all grid points
for (i, ξ) in enumerate(ξ_range)
	for (j, η) in enumerate(η_range)
		x̂ = SVector(ξ, η)
		x = el(x̂)

		# Analytical reference
		F₋₂_ana, F₋₁_ana = GRD.laurents_coeffs(K, el, û, x̂; expansion = :analytical, name = :LaplaceHypersingular)

		# Compute errors for each method
		for method in methods
			if method == :auto_diff
				F₋₂, F₋₁ = GRD.laurents_coeffs(K, el, û, x̂; expansion = :auto_diff)
			elseif method == :semi_richardson
				F₋₂, F₋₁ = GRD.laurents_coeffs(K, el, û, x̂; expansion = :semi_richardson, maxeval = maxeval, rtol = rtol, first_contract = first_contract, breaktol = breaktol, contract = contract, atol = atol)
			elseif method == :full_richardson
				F₋₂, F₋₁ = GRD.laurents_coeffs(K, el, û, x̂; expansion = :full_richardson, maxeval = maxeval, rtol = rtol, first_contract = first_contract, breaktol = breaktol, contract = contract, atol = atol)
			end

			# Compute relative errors in norm 2
			error_F₋₂ = norm(F₋₂_ana.(θs) - F₋₂.(θs), 2) / norm(F₋₂_ana.(θs), 2)
			error_F₋₁ = norm(F₋₁_ana.(θs) - F₋₁.(θs), 2) / norm(F₋₁_ana.(θs), 2)

			errors[method].F₋₂[i, j] = error_F₋₂
			errors[method].F₋₁[i, j] = error_F₋₁
		end

		if (i - 1) * N + j % 100 == 0
			@info "Progress: $((i-1)*N + j)/$(N^2) points computed"
		end
	end
end

@info "Computation complete. Creating plots..."

# Get physical coordinates as flat vectors
x_flat = [el(SVector(ξ, η))[1] for ξ in ξ_range for η in η_range]
y_flat = [el(SVector(ξ, η))[2] for ξ in ξ_range for η in η_range]

# Create Delaunay triangulation
points = [(x_flat[i], y_flat[i]) for i in 1:length(x_flat)]
tri = triangulate(points)

# Get triangulation connectivity as TriangleFace
connectivity = [TriangleFace(triangle...) for triangle in each_solid_triangle(tri)]

# Element boundary
corners_x = [y¹[1], y²[1], y⁴[1], y³[1], y¹[1]]
corners_y = [y¹[2], y²[2], y⁴[2], y³[2], y¹[2]]

# Create figure with 3x2 grid (3 methods × 2 coefficients)
fig = Figure(; size = (2100, 1000))

method_names = Dict(
	:auto_diff => "Auto Diff",
	:semi_richardson => "Semi Richardson",
	:full_richardson => "Full Richardson",
)

# Plot F₋₂ errors (first row)
for (col, method) in enumerate(methods)
	# Calculate positions: col*2-1 for axis, col*2 for colorbar
	ax = Axis(fig[1, col*2-1];
		xlabel = "x",
		ylabel = "y",
		title = "F₋₂: $(method_names[method])",
		aspect = DataAspect(),
	)

	errors_flat = vec(errors[method].F₋₂')

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
for (col, method) in enumerate(methods)
	# Calculate positions: col*2-1 for axis, col*2 for colorbar
	ax = Axis(fig[2, col*2-1];
		xlabel = "x",
		ylabel = "y",
		title = "F₋₁: $(method_names[method])",
		aspect = DataAspect(),
	)

	errors_flat = vec(errors[method].F₋₁')

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
Label(fig[0, :], "Laplace Hypersingular Laurent Coefficients Error Comparison (maxeval = $maxeval)",
	fontsize = 20, font = :bold)

# Print statistics
for method in methods
	@info "Method: $method"
	@info "  Min error F₋₂: $(minimum(errors[method].F₋₂))"
	@info "  Max error F₋₂: $(maximum(errors[method].F₋₂))"
	@info "  Min error F₋₁: $(minimum(errors[method].F₋₁))"
	@info "  Max error F₋₁: $(maximum(errors[method].F₋₁))"
end

fig

# GLMakie.save("./dev/figures/laplace/laplace_hypersingular_laurent_coeffs_error_map.png", fig)

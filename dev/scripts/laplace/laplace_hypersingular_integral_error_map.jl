import GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using LinearAlgebra
using GLMakie
using DelaunayTriangulation
using GeometryBasics

# INPUTS

N = 60 # number of points in each direction (total N² points)

# Quadrature parameters
n_rho = 10
n_theta = 40

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

# Create list of points to test (grid points)
test_points = [(i, j, SVector(ξ, η)) for (i, ξ) in enumerate(ξ_range) for (j, η) in enumerate(η_range)]

# Dictionary to store errors for each method
methods = [:auto_diff, :semi_richardson, :full_richardson, :analytical]
errors = Dict(
	method => zeros(N, N)
	for method in methods
)

@info "Computing integrals for $(length(test_points)) points ($(N^2) grid) with $(length(methods)) methods..."

# Loop over all test points
for (point_idx, (i, j, x̂)) in enumerate(test_points)
	x = el(x̂)

	# Analytical reference
	expected_I = GRD.hypersingular_laplace_integral_on_plane_element(x, el)

	# Compute errors for each method
	for method in methods
		if method == :auto_diff
			I = GRD.guiggiani_singular_integral(
				K,
				û,
				x̂,
				el,
				n_rho,
				n_theta;
				sorder = Val(-2),
				expansion = :auto_diff,
			)
		elseif method == :semi_richardson
			I = GRD.guiggiani_singular_integral(
				K,
				û,
				x̂,
				el,
				n_rho,
				n_theta;
				sorder = Val(-2),
				expansion = :semi_richardson,
				maxeval = maxeval,
				rtol = rtol,
				first_contract = first_contract,
				breaktol = breaktol,
				contract = contract,
				atol = atol,
			)
		elseif method == :full_richardson
			I = GRD.guiggiani_singular_integral(
				K,
				û,
				x̂,
				el,
				n_rho,
				n_theta;
				sorder = Val(-2),
				expansion = :full_richardson,
				maxeval = maxeval,
				rtol = rtol,
				first_contract = first_contract,
				breaktol = breaktol,
				contract = contract,
				atol = atol,
			)
		elseif method == :analytical
			I = GRD.guiggiani_singular_integral(
				K,
				û,
				x̂,
				el,
				n_rho,
				n_theta;
				sorder = Val(-2),
				expansion = :analytical,
			)
		end

		# Compute relative error
		error = abs(I - expected_I) / abs(expected_I)

		# Store error in appropriate location
		errors[method][i, j] = error
	end

	if point_idx % 100 == 0
		@info "Progress: $point_idx/$(length(test_points)) points computed"
	end
end

@info "Computation complete. Creating plots..."

# Get physical coordinates as flat vectors
x_flat = [el(p[3])[1] for p in test_points]
y_flat = [el(p[3])[2] for p in test_points]

# Create Delaunay triangulation
points = [(x_flat[i], y_flat[i]) for i in 1:length(x_flat)]
tri = triangulate(points)

# Get triangulation connectivity as TriangleFace
connectivity = [TriangleFace(triangle...) for triangle in each_solid_triangle(tri)]

# Element boundary
corners_x = [y¹[1], y²[1], y⁴[1], y³[1], y¹[1]]
corners_y = [y¹[2], y²[2], y⁴[2], y³[2], y¹[2]]

# Create figure with 2x2 grid
fig = Figure(; size = (1600, 1400))

method_names = Dict(
	:auto_diff => "Auto Diff",
	:semi_richardson => "Semi Richardson",
	:full_richardson => "Full Richardson",
	:analytical => "Analytical",
)

# Plot integral errors in 2x2 grid
for (idx, method) in enumerate(methods)
	# Calculate row and column: (row-1)*2 + col gives positions 1,2,3,4
	row = div(idx - 1, 2) + 1
	col = mod(idx - 1, 2) + 1

	# Positions: row, col*2-1 for axis, col*2 for colorbar
	ax = Axis(fig[row, col*2-1];
		xlabel = "x",
		ylabel = "y",
		title = "$(method_names[method])",
		aspect = DataAspect(),
	)

	# Extract errors in the same order as test_points
	errors_flat = [errors[method][p[1], p[2]] for p in test_points]

	hm = mesh!(ax,
		Point2f.(x_flat, y_flat),
		connectivity,
		color = log10.(errors_flat),
		colormap = :turbo,
		shading = NoShading,
	)

	Colorbar(fig[row, col*2], hm; label = "log₁₀(Error)", width = 15)

	lines!(ax, corners_x, corners_y; color = :black, linewidth = 2)
end

# Add title
Label(fig[0, :], "Laplace Hypersingular Integral Error (n_rho = $n_rho, n_theta = $n_theta, maxeval = $maxeval)",
	fontsize = 20, font = :bold)

# Print statistics
for method in methods
	@info "Method: $method"
	@info "  Min error: $(minimum(errors[method]))"
	@info "  Max error: $(maximum(errors[method]))"
	@info "  Mean error: $(sum(errors[method]) / length(errors[method]))"
end

fig
GLMakie.display(fig)
GLMakie.save("./dev/figures/laplace/laplace_hypersingular_integral_error_map.png", fig)

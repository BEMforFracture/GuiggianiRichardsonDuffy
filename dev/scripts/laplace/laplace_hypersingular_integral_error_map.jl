import GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using LinearAlgebra
using GLMakie
using DelaunayTriangulation
using GeometryBasics

# Configuration
N = 60  # Number of points in each direction (total N² points)

# Quadrature parameters
n_rho = 10
n_theta = 40
quad_rho = Inti.GaussLegendre(n_rho)
quad_theta = Inti.GaussLegendre(n_theta)

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
K_base = Inti.HyperSingularKernel(Inti.Laplace(dim=3))
K = GRD.SplitKernel(K_base)

# Methods to test
methods = [
	("AutoDiff", GRD.AutoDiffExpansion(), K),
	("SemiRichardson", GRD.SemiRichardsonExpansion(rich_params), K),
	("FullRichardson", GRD.FullRichardsonExpansion(rich_params), K),
	("Analytical", GRD.AnalyticalExpansion(), K_base),
]

# Create grid of test points (avoid boundaries)
ξ_min = 1 / N
ξ_max = 1 - 1 / N
ξ_range = range(ξ_min, ξ_max; length = N)
η_range = range(ξ_min, ξ_max; length = N)

test_points = [(i, j, SVector(ξ, η)) for (i, ξ) in enumerate(ξ_range) for (j, η) in enumerate(η_range)]

# Storage for errors
errors = Dict(m[1] => zeros(N, N) for m in methods)

@info "Computing integrals for $(length(test_points)) points with $(length(methods)) methods..."

# Compute errors for all points and methods
for (point_idx, (i, j, x̂)) in enumerate(test_points)
	x = el(x̂)
	expected_I = GRD.hypersingular_laplace_integral_on_plane_element(x, el)
	
	for (method_name, method, K_to_use) in methods
		I = GRD.guiggiani_singular_integral(
			K_to_use, û, x̂, el, quad_rho, quad_theta, method
		)
		error = abs(I - expected_I) / abs(expected_I)
		errors[method_name][i, j] = error
	end
	
	if point_idx % 100 == 0
		@info "Progress: $point_idx/$(length(test_points)) points computed"
	end
end

@info "Computation complete. Creating plots..."

# Get physical coordinates
x_flat = [el(p[3])[1] for p in test_points]
y_flat = [el(p[3])[2] for p in test_points]

# Create Delaunay triangulation
points = [(x_flat[i], y_flat[i]) for i in 1:length(x_flat)]
tri = triangulate(points)
connectivity = [TriangleFace(triangle...) for triangle in each_solid_triangle(tri)]

# Element boundary
corners_x = [y¹[1], y²[1], y⁴[1], y³[1], y¹[1]]
corners_y = [y¹[2], y²[2], y⁴[2], y³[2], y¹[2]]

# Create figure
fig = Figure(; size = (1600, 1400))

# Plot each method in 2x2 grid
for (idx, (method_name, _, _)) in enumerate(methods)
	row = div(idx - 1, 2) + 1
	col = mod(idx - 1, 2) + 1
	
	ax = Axis(fig[row, col*2-1];
		xlabel = "x",
		ylabel = "y",
		title = method_name,
		aspect = DataAspect(),
	)
	
	# Extract errors
	errors_flat = [errors[method_name][p[1], p[2]] for p in test_points]
	
	# Create mesh plot
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
Label(fig[0, :], 
	"Laplace Hypersingular Integral Error (n_rho = $n_rho, n_theta = $n_theta, maxeval = $(rich_params.maxeval))",
	fontsize = 20, font = :bold)

# Print statistics
for (method_name, _, _) in methods
	@info "Method: $method_name"
	@info "  Min error: $(minimum(errors[method_name]))"
	@info "  Max error: $(maximum(errors[method_name]))"
	@info "  Mean error: $(sum(errors[method_name]) / length(errors[method_name]))"
end

display(fig)
# GLMakie.save("./dev/figures/laplace/laplace_hypersingular_integral_error_map.png", fig)

using Inti

include("correlations.jl")
include("nmax_vs_compactness.jl")
include("nmax_vs_min_angle.jl")
include("nmax_vs_side_ratio.jl")
include("nmax_vs_jacobian_ratio.jl")
include("nmax_vs_corner_distance.jl")

op_laplace = Inti.Laplace(; dim = 3)

E = 210e9
ν = 0.3
μ = E / (2 * (1 + ν))
λ = E * ν / ((1 + ν) * (1 - 2 * ν))
op_navier = Inti.Elastostatic(; μ = μ, λ = λ, dim = 3)

seed = 42
nb_elements = 100

X_quad, X_tri, names_quad, names_tri = correlation_matrices(op_laplace, seed, nb_elements)
fig_quad, fig_tri, fig_quad_partial_matrix, fig_tri_partial_matrix, fig_quad_nmax, fig_tri_nmax = plot_correlations(X_quad, X_tri, names_quad, names_tri)
fig_quad_partial, fig_tri_partial = nmax_partial_correlation_analysis(X_quad, X_tri, names_quad, names_tri)
fig_quad_compactness, fig_tri_compactness = plot_nmax_vs_compactness_by_distance(op_laplace, seed, nb_elements; epsilon_target = 1e-10)
fig_quad_min_angle, fig_tri_min_angle = plot_nmax_vs_min_angle_by_distance(op_laplace, seed, nb_elements; epsilon_target = 1e-10)
fig_quad_side_ratio, fig_tri_side_ratio = plot_nmax_vs_side_ratio_by_distance(op_laplace, seed, nb_elements; epsilon_target = 1e-10)
fig_quad_jacobian_ratio, fig_tri_jacobian_ratio = plot_nmax_vs_jacobian_ratio_by_distance(op_laplace, seed, nb_elements; epsilon_target = 1e-10)
fig_quad_corner_dist, fig_tri_corner_dist = plot_nmax_vs_corner_distance_by_distance(op_laplace, seed, nb_elements; epsilon_target = 1e-10)

function _gallery_screen(fig_pairs)
	labels = first.(fig_pairs)
	# colorbuffer needs axis permutation plus one axis flip for upright preview.
	buffers = [reverse(permutedims(collect(GLMakie.colorbuffer(fig)), (2, 1)); dims = 2) for (_, fig) in fig_pairs]

	gallery = GLMakie.Figure(size = (1400, 1100))
	menu = GLMakie.Menu(
		gallery[1, 1],
		options = [lbl => i for (i, lbl) in enumerate(labels)],
		default = 1,
		width = 700,
	)
	ax = GLMakie.Axis(gallery[2, 1], title = labels[1])
	img_obs = GLMakie.Observable(buffers[1])
	GLMakie.image!(ax, img_obs)
	ax.aspect = GLMakie.DataAspect()
	GLMakie.hidedecorations!(ax)
	GLMakie.hidespines!(ax)

	# Adjust row sizes: menu 8% of height, image 92%
	GLMakie.rowsize!(gallery.layout, 1, GLMakie.Relative(0.08))
	GLMakie.rowsize!(gallery.layout, 2, GLMakie.Relative(0.92))
	
	# Reduce gaps and padding to maximize plot area
	GLMakie.rowgap!(gallery.layout, 2)
	GLMakie.colgap!(gallery.layout, 5)
	GLMakie.tight_ticklabel_spacing!(ax)

	on(menu.selection) do selected
		idx = selected isa Pair ? last(selected) : selected
		img_obs[] = buffers[idx]
		ax.title = labels[idx]
		GLMakie.autolimits!(ax)  # Reinitialize axis limits for new image
	end

	return display(GLMakie.Screen(), gallery)
end

fig_pairs = [
	("Quadrangles - Correlation Matrix", fig_quad),
	("Triangles - Correlation Matrix", fig_tri),
	("Quadrangles - Partial Correlation Matrix | target_error", fig_quad_partial_matrix),
	("Triangles - Partial Correlation Matrix | target_error", fig_tri_partial_matrix),
	("Quadrangles - Correlation with nmax", fig_quad_nmax),
	("Triangles - Correlation with nmax", fig_tri_nmax),
	("Quadrangles - Partial Correlation Diagnostics", fig_quad_partial),
	("Triangles - Partial Correlation Diagnostics", fig_tri_partial),
	("Quadrangles - nmax vs compactness", fig_quad_compactness),
	("Triangles - nmax vs compactness", fig_tri_compactness),
	("Quadrangles - nmax vs min_angle", fig_quad_min_angle),
	("Triangles - nmax vs min_angle", fig_tri_min_angle),
	("Quadrangles - nmax vs side_ratio", fig_quad_side_ratio),
	("Triangles - nmax vs side_ratio", fig_tri_side_ratio),
	("Quadrangles - nmax vs jacobian_ratio", fig_quad_jacobian_ratio),
	("Triangles - nmax vs jacobian_ratio", fig_tri_jacobian_ratio),
	("Quadrangles - nmax vs corner_dist", fig_quad_corner_dist),
	("Triangles - nmax vs corner_dist", fig_tri_corner_dist),
]

gallery_window = _gallery_screen(fig_pairs)

# GLMakie.save("quadrangles_correlation_heatmap.png", fig_quad)
# GLMakie.save("triangles_correlation_heatmap.png", fig_tri)
# GLMakie.save("quadrangles_nmax_correlation.png", fig_quad_nmax)
# GLMakie.save("triangles_nmax_correlation.png", fig_tri_nmax)
# GLMakie.save("quadrangles_nmax_partial_correlation.png", fig_quad_partial)
# GLMakie.save("triangles_nmax_partial_correlation.png", fig_tri_partial)
# GLMakie.save("quadrangles_nmax_vs_compactness_epsilon_1e-10.png", fig_quad_compactness)
# GLMakie.save("triangles_nmax_vs_compactness_epsilon_1e-10.png", fig_tri_compactness)
# GLMakie.save("quadrangles_nmax_vs_min_angle_epsilon_1e-10.png", fig_quad_min_angle)
# GLMakie.save("triangles_nmax_vs_min_angle_epsilon_1e-10.png", fig_tri_min_angle)
# GLMakie.save("quadrangles_nmax_vs_side_ratio_epsilon_1e-10.png", fig_quad_side_ratio)
# GLMakie.save("triangles_nmax_vs_side_ratio_epsilon_1e-10.png", fig_tri_side_ratio)
# GLMakie.save("quadrangles_nmax_vs_jacobian_ratio_epsilon_1e-10.png", fig_quad_jacobian_ratio)
# GLMakie.save("triangles_nmax_vs_jacobian_ratio_epsilon_1e-10.png", fig_tri_jacobian_ratio)
# GLMakie.save("quadrangles_nmax_vs_corner_distance_epsilon_1e-10.png", fig_quad_corner_dist)
# GLMakie.save("triangles_nmax_vs_corner_distance_epsilon_1e-10.png", fig_tri_corner_dist)
# GLMakie.save("quadrangles_compactness_vs_elongation.png", fig_shape_quad)
# GLMakie.save("triangles_compactness_vs_elongation.png", fig_shape_tri)
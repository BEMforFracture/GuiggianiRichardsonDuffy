using Inti
using Serialization

include("correlations.jl")
include("nmax_vs_compactness.jl")
include("nmax_vs_min_angle.jl")
include("nmax_vs_side_ratio.jl")
include("nmax_vs_jacobian_ratio.jl")
include("nmax_vs_jacobian_at_point.jl")
include("nmax_vs_corner_distance.jl")

op_laplace = Inti.Laplace(; dim = 3)

E = 210e9
ν = 0.3
μ = E / (2 * (1 + ν))
λ = E * ν / ((1 + ν) * (1 - 2 * ν))
op_navier = Inti.Elastostatic(; μ = μ, λ = λ, dim = 3)

seed = 42
nb_elements = 100

fixed_epsilon = 1e-6

cache_tag = "seed$(seed)_n$(nb_elements)_eps$(replace(string(fixed_epsilon), "." => "p"))"
cache_path = joinpath(@__DIR__, "..", "..", "results", "calibration_cache_$(cache_tag).bin")

function _calibration_data_from_main()
	required = (
		:X_quad,
		:X_tri,
		:names_quad,
		:names_tri,
		:quads,
		:quad_params,
		:tris,
		:tri_params,
		:nmax_by_element,
		:all_cases,
	)
	all(isdefined(Main, sym) for sym in required) || return nothing

	return (
		X_quad = Main.X_quad,
		X_tri = Main.X_tri,
		names_quad = Main.names_quad,
		names_tri = Main.names_tri,
		quads = Main.quads,
		quad_params = Main.quad_params,
		tris = Main.tris,
		tri_params = Main.tri_params,
		nmax_by_element = Main.nmax_by_element,
		all_cases = Main.all_cases,
	)
end

function _load_or_compute_calibration_data(cache_path::AbstractString, op, seed::Int, nb_elements::Int)
	data = _calibration_data_from_main()
	if !isnothing(data)
		println("Using calibration data already present in memory; saving it to cache at ", cache_path)
		mkpath(dirname(cache_path))
		serialize(cache_path, data)
		return data
	end

	if isfile(cache_path)
		println("Loading cached calibration data from ", cache_path)
		return deserialize(cache_path)
	end

	println("Cache not found at ", cache_path)
	println("Computing calibration data once, then saving it for future runs.")
	data = let
		X_quad, X_tri, names_quad, names_tri, quads, quad_params, tris, tri_params, nmax_by_element, all_cases = correlation_matrices(op, seed, nb_elements)
		(
			X_quad = X_quad,
			X_tri = X_tri,
			names_quad = names_quad,
			names_tri = names_tri,
			quads = quads,
			quad_params = quad_params,
			tris = tris,
			tri_params = tri_params,
			nmax_by_element = nmax_by_element,
			all_cases = all_cases,
		)
	end
	mkpath(dirname(cache_path))
	serialize(cache_path, data)
	println("Saved cached calibration data to ", cache_path)
	return data
end

function save_calibration_cache_from_main(cache_path::AbstractString = cache_path)
	data = _calibration_data_from_main()
	isnothing(data) && error("No calibration data found in Main. Run the sampling once so X_quad, X_tri, etc. exist in the current session.")
	mkpath(dirname(cache_path))
	serialize(cache_path, data)
	println("Saved cached calibration data to ", cache_path)
	return cache_path
end

data = _load_or_compute_calibration_data(cache_path, op_laplace, seed, nb_elements)
X_quad = data.X_quad
X_tri = data.X_tri
names_quad = data.names_quad
names_tri = data.names_tri
quads = data.quads
quad_params = data.quad_params
tris = data.tris
tri_params = data.tri_params
nmax_by_element = data.nmax_by_element
all_cases = data.all_cases
fig_quad, fig_tri, fig_quad_partial_matrix, fig_tri_partial_matrix, fig_quad_nmax, fig_tri_nmax = plot_correlations(X_quad, X_tri, names_quad, names_tri)
fig_quad_partial, fig_tri_partial = nmax_partial_correlation_analysis(X_quad, X_tri, names_quad, names_tri)
# collision_report = nmax_parameter_collision_analysis_by_epsilon(X_quad, X_tri, names_quad, names_tri; rounded_digits = nothing, max_show = 4)
fig_quad_compactness, fig_tri_compactness = plot_nmax_vs_compactness_by_distance(op_laplace, quads, quad_params, tris, tri_params; epsilon_target = fixed_epsilon)
fig_quad_min_angle, fig_tri_min_angle = plot_nmax_vs_min_angle_by_distance(op_laplace, quads, quad_params, tris, tri_params; epsilon_target = fixed_epsilon)
fig_quad_side_ratio, fig_tri_side_ratio = plot_nmax_vs_side_ratio_by_distance(op_laplace, quads, quad_params, tris, tri_params; epsilon_target = fixed_epsilon)
fig_quad_jacobian_ratio, fig_tri_jacobian_ratio = plot_nmax_vs_jacobian_ratio_by_distance(op_laplace, quads, quad_params, tris, tri_params; epsilon_target = fixed_epsilon)
fig_quad_jacobian_at_point, fig_tri_jacobian_at_point = plot_nmax_vs_jacobian_at_point_by_distance(op_laplace, quads, quad_params, tris, tri_params; epsilon_target = fixed_epsilon)
fig_quad_corner_dist, fig_tri_corner_dist = plot_nmax_vs_corner_distance_by_distance(op_laplace, quads, quad_params, tris, tri_params; epsilon_target = fixed_epsilon)

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
	("Quadrangles - nmax vs jacobian_at_point", fig_quad_jacobian_at_point),
	("Triangles - nmax vs jacobian_at_point", fig_tri_jacobian_at_point),
	("Quadrangles - nmax vs corner_dist", fig_quad_corner_dist),
	("Triangles - nmax vs corner_dist", fig_tri_corner_dist),
]

gallery_window = _gallery_screen(fig_pairs)
epsilon_min = minimum(c.epsilon for c in all_cases)
cases_at_min_epsilon = [c for c in all_cases if c.epsilon == fixed_epsilon]
case_browser_window = case_browser(cases_at_min_epsilon; sort_by = :nmax, reverse_order = true)

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
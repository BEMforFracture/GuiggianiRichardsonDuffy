using GLMakie

@isdefined(generate_random_quadrangles) || include("quadrangle_generator.jl")
@isdefined(generate_random_triangles) || include("triangle_generator.jl")
@isdefined(build_quad1n_config) || include("source_points_generator.jl")
@isdefined(run_case_s) || include("case.jl")

function _distance_key(d::Float64; digits::Int = 8)
    return round(d; digits = digits)
end

function _push_case_point!(groups::Dict{Float64, Vector{NTuple{2, Float64}}}, c::Case)
    ref_el = Inti.reference_domain(c.el)
    d = _distance_key(dist_to_border(ref_el, c.point))
    compactness = get(c.quality_metrics, :compactness, NaN)
    nmax = float(maximum(c.n_thetas))

    if !isfinite(compactness) || !isfinite(nmax)
        return
    end

    if !haskey(groups, d)
        groups[d] = NTuple{2, Float64}[]
    end
    push!(groups[d], (compactness, nmax))
end

function _collect_quad_groups(op, seed::Int, nb_elements::Int, epsilon_target::Float64)
    quads, quad_params = generate_random_quadrangles(nb_elements, seed = seed)
    groups = Dict{Float64, Vector{NTuple{2, Float64}}}()

    for (i, el) in enumerate(quads)
        gen_params = (aspect_ratio = quad_params[i].aspect_ratio, shear = quad_params[i].shear, trapezoid = quad_params[i].trapezoid)
        configs = (
            build_quad1n_config(el),
            build_quad4n_config(el),
            build_quad9n_config(el),
        )

        for config in configs
            cases = run_case_s(config, epsilon_target, op; generation_params = gen_params)
            for c in cases
                _push_case_point!(groups, c)
            end
        end
    end

    return groups
end

function _collect_tri_groups(op, seed::Int, nb_elements::Int, epsilon_target::Float64)
    tris, tri_params = generate_random_triangles(nb_elements, seed = seed)
    groups = Dict{Float64, Vector{NTuple{2, Float64}}}()

    for (i, el) in enumerate(tris)
        gen_params = (aspect_ratio = tri_params[i].aspect_ratio, skewness = tri_params[i].skewness, elongation_gen = tri_params[i].elongation)
        configs = (
            build_tri1n_config(el),
            build_tri3n_config(el),
            build_tri6n_config(el),
        )

        for config in configs
            cases = run_case_s(config, epsilon_target, op; generation_params = gen_params)
            for c in cases
                _push_case_point!(groups, c)
            end
        end
    end

    return groups
end

function _plot_groups(groups::Dict{Float64, Vector{NTuple{2, Float64}}}, title_text::String)
    fig = GLMakie.Figure(size = (920, 520))
    ax = GLMakie.Axis(
        fig[1, 1],
        title = title_text,
        xlabel = "compactness",
        ylabel = "nmax",
    )

    dvals = sort(collect(keys(groups)))
    palette = GLMakie.cgrad(:Set2, max(length(dvals), 1); categorical = true)

    for (k, d) in enumerate(dvals)
        pts = groups[d]
        order = sortperm(first.(pts))
        x = first.(pts)[order]
        y = last.(pts)[order]
        label = "dist = $(round(d; digits = 4))"

        GLMakie.scatter!(ax, x, y; color = palette[k], markersize = 7, label = label)
    end

    GLMakie.axislegend(ax; position = :rb)
    return fig
end

function plot_nmax_vs_compactness_by_distance(op, seed::Int, nb_elements::Int; epsilon_target::Float64 = 1e-10)
    quad_groups = _collect_quad_groups(op, seed, nb_elements, epsilon_target)
    tri_groups = _collect_tri_groups(op, seed, nb_elements, epsilon_target)

    fig_quad = _plot_groups(
        quad_groups,
        "Quadrangles - nmax vs compactness by distance class (epsilon=$(epsilon_target), $(nb_elements) elements)",
    )
    fig_tri = _plot_groups(
        tri_groups,
        "Triangles - nmax vs compactness by distance class (epsilon=$(epsilon_target), $(nb_elements) elements)",
    )

    return fig_quad, fig_tri
end

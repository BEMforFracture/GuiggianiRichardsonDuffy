## This script computes all the correlations between numerical parameters, in perticular how each parameter affects nmax.

using MultivariateStats
using LinearAlgebra
using GLMakie
using Statistics

include("quadrangle_generator.jl")
include("triangle_generator.jl")
include("source_points_generator.jl")
include("case.jl")

function _print_progress(current::Int, total::Int; prefix::String = "Sampling", width::Int = 40)
    ratio = total == 0 ? 1.0 : current / total
    filled = round(Int, ratio * width)
    bar = repeat("=", filled) * repeat(" ", width - filled)
    percent = round(100 * ratio, digits = 1)
    print("\r", prefix, " [", bar, "] ", percent, "% (", current, "/", total, ")")
    flush(stdout)
    if current >= total
        println()
    end
end


function _nonconstant_columns(X, names)
    keep = vec(std(X, dims = 1)) .> 0
    return X[:, keep], names[keep]
end

function _rank_vector(x::AbstractVector)
    n = length(x)
    order = sortperm(x)
    ranks = zeros(Float64, n)

    i = 1
    while i <= n
        j = i
        while j < n && x[order[j]] == x[order[j + 1]]
            j += 1
        end
        rank_value = (i + j) / 2
        for k in i:j
            ranks[order[k]] = rank_value
        end
        i = j + 1
    end

    return ranks
end

function _spearman_correlation_matrix(X)
    ranked = similar(float.(X))
    for j in 1:size(X, 2)
        ranked[:, j] = _rank_vector(view(X, :, j))
    end
    return cor(ranked)
end

function _rank_matrix(X)
    ranked = similar(float.(X))
    for j in 1:size(X, 2)
        ranked[:, j] = _rank_vector(view(X, :, j))
    end
    return ranked
end

function _residualize(y::AbstractVector, Z::AbstractMatrix)
    n = length(y)
    Xreg = hcat(ones(n), Z)
    β = Xreg \ y
    return y - Xreg * β
end

# Partial Spearman correlation between columns i and j, controlling for controls:
# 1) rank all variables (done before calling this function),
# 2) residualize Xi and Yj against Z = controls,
# 3) corr(residual_Xi, residual_Yj).
#
# This is equivalent to the standard one-control closed form:
# r_{XY.Z} = (r_{XY} - r_{XZ}*r_{YZ}) / sqrt((1-r_{XZ}^2)*(1-r_{YZ}^2)).
function _partial_spearman(ranked::AbstractMatrix, i::Int, j::Int, controls::Vector{Int})
    xi = view(ranked, :, i)
    yj = view(ranked, :, j)
    if isempty(controls)
        return cor(xi, yj)
    end
    Z = ranked[:, controls]
    rx = _residualize(xi, Z)
    ry = _residualize(yj, Z)
    return cor(rx, ry)
end

function _nmax_correlation_diagnostics(X, names; target_name = "nmax", control_names = ["dist", "target_error"])
    X_corr, names_corr = _nonconstant_columns(X, names)
    ranked = _rank_matrix(X_corr)
    corr_matrix = cor(ranked)

    idx_target = findfirst(==(target_name), names_corr)
    isnothing(idx_target) && error("Target variable $(target_name) not found after filtering.")

    control_idx = Int[]
    for cname in control_names
        idx = findfirst(==(cname), names_corr)
        if !isnothing(idx)
            push!(control_idx, idx)
        end
    end

    rows = NamedTuple[]
    for i in 1:length(names_corr)
        i == idx_target && continue
        nm = names_corr[i]
        marg = corr_matrix[i, idx_target]
        part = _partial_spearman(ranked, i, idx_target, control_idx)
        in_controls = i in control_idx
        push!(rows, (name = nm, marginal = marg, partial = part, is_control = in_controls))
    end

    rows = sort(rows; by = r -> abs(r.partial), rev = true)
    return rows, names_corr
end

function _nmax_parameter_collisions(X, names; target_name = "nmax", rounded_digits::Union{Nothing, Int} = nothing)
    idx_nmax = findfirst(==(target_name), names)
    isnothing(idx_nmax) && error("Target variable $(target_name) not found.")

    feature_idx = [i for i in 1:length(names) if i != idx_nmax]
    groups = Dict{Tuple, Vector{Float64}}()

    for r in 1:size(X, 1)
        raw_vals = X[r, feature_idx]
        key_vals = if isnothing(rounded_digits)
            raw_vals
        else
            round.(raw_vals; digits = rounded_digits)
        end
        key = Tuple(key_vals)
        push!(get!(groups, key, Float64[]), X[r, idx_nmax])
    end

    collisions = NamedTuple[]
    for (key, nmax_vals) in groups
        length(nmax_vals) <= 1 && continue
        uniq = sort(unique(nmax_vals))
        if length(uniq) > 1
            push!(collisions, (
                group_size = length(nmax_vals),
                nmax_values = uniq,
                nmax_min = minimum(uniq),
                nmax_max = maximum(uniq),
                nmax_span = maximum(uniq) - minimum(uniq),
                feature_key = key,
            ))
        end
    end

    sort!(collisions; by = c -> (c.nmax_span, c.group_size), rev = true)

    total_groups = length(groups)
    repeated_groups = count(v -> length(v) > 1, values(groups))
    return (
        n_samples = size(X, 1),
        n_features = length(feature_idx),
        total_groups = total_groups,
        repeated_groups = repeated_groups,
        collision_groups = length(collisions),
        collisions = collisions,
        feature_names = names[feature_idx],
        rounded_digits = rounded_digits,
    )
end

function _print_collision_report(label::String, report; max_show::Int = 5)
    mode_txt = isnothing(report.rounded_digits) ? "exact" : "rounded($(report.rounded_digits) digits)"
    println("\n=== $(label): identical-parameter collision check [$(mode_txt)] ===")
    println("samples=", report.n_samples,
        ", groups=", report.total_groups,
        ", repeated_groups=", report.repeated_groups,
        ", collision_groups=", report.collision_groups)

    if isempty(report.collisions)
        println("No collisions found: identical parameter sets do not produce different nmax.")
        return
    end

    to_show = min(max_show, length(report.collisions))
    println("Top ", to_show, " collisions (same parameters, different nmax):")
    for (k, c) in enumerate(report.collisions[1:to_show])
        println("  #", k,
            " size=", c.group_size,
            ", nmax_values=", c.nmax_values,
            ", span=", c.nmax_span)
    end
end

function nmax_parameter_collision_analysis(X_quad, X_tri, names_quad, names_tri; rounded_digits::Int = 8, max_show::Int = 5)
    quad_exact = _nmax_parameter_collisions(X_quad, names_quad; rounded_digits = nothing)
    tri_exact = _nmax_parameter_collisions(X_tri, names_tri; rounded_digits = nothing)
    quad_rounded = _nmax_parameter_collisions(X_quad, names_quad; rounded_digits = rounded_digits)
    tri_rounded = _nmax_parameter_collisions(X_tri, names_tri; rounded_digits = rounded_digits)

    _print_collision_report("Quadrangles", quad_exact; max_show = max_show)
    _print_collision_report("Triangles", tri_exact; max_show = max_show)
    _print_collision_report("Quadrangles", quad_rounded; max_show = max_show)
    _print_collision_report("Triangles", tri_rounded; max_show = max_show)

    return (quad_exact = quad_exact, tri_exact = tri_exact, quad_rounded = quad_rounded, tri_rounded = tri_rounded)
end

function _nmax_parameter_collisions_by_epsilon(X, names; epsilon_name = "target_error", rounded_digits::Union{Nothing, Int} = nothing, max_show::Int = 5)
    idx_eps = findfirst(==(epsilon_name), names)
    isnothing(idx_eps) && error("Epsilon variable $(epsilon_name) not found.")

    eps_values = sort(unique(X[:, idx_eps]))
    reports = Dict{Float64, Any}()

    for eps in eps_values
        rows = findall(r -> r == eps, X[:, idx_eps])
        X_eps = X[rows, :]
        rep = _nmax_parameter_collisions(X_eps, names; rounded_digits = rounded_digits)
        reports[eps] = rep
    end

    return reports
end

function _print_collision_report_by_epsilon(label::String, reports::Dict{Float64, Any}; max_show::Int = 5)
    println("\n=== $(label): identical-parameter collision check [epsilon fixed] ===")
    for eps in sort(collect(keys(reports)))
        rep = reports[eps]
        println("epsilon=", eps,
            " | samples=", rep.n_samples,
            ", groups=", rep.total_groups,
            ", repeated_groups=", rep.repeated_groups,
            ", collision_groups=", rep.collision_groups)

        if isempty(rep.collisions)
            continue
        end

        to_show = min(max_show, length(rep.collisions))
        for (k, c) in enumerate(rep.collisions[1:to_show])
            println("  #", k,
                " size=", c.group_size,
                ", nmax_values=", c.nmax_values,
                ", span=", c.nmax_span)
        end
    end
end

function nmax_parameter_collision_analysis_by_epsilon(X_quad, X_tri, names_quad, names_tri; rounded_digits::Union{Nothing, Int} = nothing, max_show::Int = 5)
    quad_reports = _nmax_parameter_collisions_by_epsilon(X_quad, names_quad; rounded_digits = rounded_digits, max_show = max_show)
    tri_reports = _nmax_parameter_collisions_by_epsilon(X_tri, names_tri; rounded_digits = rounded_digits, max_show = max_show)

    _print_collision_report_by_epsilon("Quadrangles", quad_reports; max_show = max_show)
    _print_collision_report_by_epsilon("Triangles", tri_reports; max_show = max_show)

    return (quad_by_epsilon = quad_reports, tri_by_epsilon = tri_reports, rounded_digits = rounded_digits)
end

function _nmax_partial_correlation_figure(rows, figtitle)
    filtered = [r for r in rows if !r.is_control]
    labels = [r.name for r in filtered]
    marginal = [r.marginal for r in filtered]
    partial = [r.partial for r in filtered]

    fig = GLMakie.Figure(size = (1100, 460))
    ax = GLMakie.Axis(fig[1, 1], title = figtitle, ylabel = "Spearman correlation with nmax")

    n = length(labels)
    xpos = collect(1:n)
    δ = 0.18
    GLMakie.barplot!(ax, xpos .- δ, marginal; width = 0.33, color = (:steelblue, 0.85), label = "marginal")
    GLMakie.barplot!(ax, xpos .+ δ, partial; width = 0.33, color = (:darkorange, 0.85), label = "partial | target_error")

    ax.xticks = (xpos, string.(labels))
    ax.xticklabelrotation = π / 5
    GLMakie.hlines!(ax, [0.0], color = :black, linewidth = 1)
    GLMakie.ylims!(ax, -1.0, 1.0)
    GLMakie.axislegend(ax; position = :rb)

    return fig
end

function nmax_partial_correlation_analysis(X_quad, X_tri, names_quad, names_tri)
    rows_quad, _ = _nmax_correlation_diagnostics(X_quad, names_quad; control_names = ["target_error"])
    rows_tri, _ = _nmax_correlation_diagnostics(X_tri, names_tri; control_names = ["target_error"])

    fig_quad = _nmax_partial_correlation_figure(rows_quad, "Quadrangles - nmax correlation diagnostics")
    fig_tri = _nmax_partial_correlation_figure(rows_tri, "Triangles - nmax correlation diagnostics")

    println("\n=== Quadrangles: Spearman diagnostics for nmax ===")
    for r in rows_quad
        println("  ", r.name, ": marginal=", round(r.marginal, digits = 3), ", partial=", round(r.partial, digits = 3), r.is_control ? " (control)" : "")
    end

    println("\n=== Triangles: Spearman diagnostics for nmax ===")
    for r in rows_tri
        println("  ", r.name, ": marginal=", round(r.marginal, digits = 3), ", partial=", round(r.partial, digits = 3), r.is_control ? " (control)" : "")
    end

    return fig_quad, fig_tri
end

function _partial_spearman_matrix(X, names; control_names = ["target_error"], drop_controls = true)
    X_corr, names_corr = _nonconstant_columns(X, names)
    ranked = _rank_matrix(X_corr)

    control_idx = Int[]
    for cname in control_names
        idx = findfirst(==(cname), names_corr)
        if !isnothing(idx)
            push!(control_idx, idx)
        end
    end

    kept_idx = if drop_controls
        [i for i in 1:length(names_corr) if !(i in control_idx)]
    else
        collect(1:length(names_corr))
    end

    names_partial = names_corr[kept_idx]
    n_vars = length(kept_idx)
    corr_matrix = Matrix{Float64}(undef, n_vars, n_vars)

    for a in 1:n_vars
        corr_matrix[a, a] = 1.0
        for b in (a + 1):n_vars
            i = kept_idx[a]
            j = kept_idx[b]
            ρ = _partial_spearman(ranked, i, j, control_idx)
            corr_matrix[a, b] = ρ
            corr_matrix[b, a] = ρ
        end
    end

    return corr_matrix, names_partial
end

function _correlation_heatmap_from_matrix(corr_matrix, names_corr; figtitle = "")
    n_vars = length(names_corr)

    fig = GLMakie.Figure(size = (130 + 95 * n_vars, 130 + 95 * n_vars))
    ax = GLMakie.Axis(fig[1, 1], title = figtitle)

    hm = GLMakie.heatmap!(ax, 1:n_vars, 1:n_vars, corr_matrix; colormap = :balance, colorrange = (-1.0, 1.0))
    GLMakie.Colorbar(fig[1, 2], hm, label = "corr")

    ax.xticks = (1:n_vars, string.(names_corr))
    ax.yticks = (1:n_vars, string.(names_corr))
    ax.xticklabelrotation = π / 3
    ax.yreversed = true

    for i in 1:n_vars
        for j in 1:n_vars
            GLMakie.text!(ax, i, j, text = string(round(corr_matrix[j, i], digits = 2)), align = (:center, :center), fontsize = 10, color = :black)
        end
    end

    return fig
end

function _correlation_heatmap_figure(X, names; figtitle="")
    X_corr, names_corr = _nonconstant_columns(X, names)
    corr_matrix = _spearman_correlation_matrix(X_corr)
    fig = _correlation_heatmap_from_matrix(corr_matrix, names_corr; figtitle = figtitle)
    return fig, corr_matrix, names_corr
end

function _nmax_correlation_figure(corr_matrix, names_corr; figtitle="")
    idx_nmax = findfirst(==("nmax"), names_corr)
    isnothing(idx_nmax) && error("nmax was filtered out; cannot build nmax correlation plot.")

    other_idx = filter(i -> i != idx_nmax, 1:length(names_corr))
    corr_vals = corr_matrix[other_idx, idx_nmax]
    labels = names_corr[other_idx]

    order = sortperm(abs.(corr_vals), rev = true)
    corr_vals = corr_vals[order]
    labels = labels[order]

    fig = GLMakie.Figure(size = (1000, 420))
    ax = GLMakie.Axis(fig[1, 1], title = figtitle, ylabel = "Spearman corr with nmax")

    GLMakie.barplot!(ax, 1:length(labels), corr_vals)
    ax.xticks = (1:length(labels), string.(labels))
    ax.xticklabelrotation = π / 4
    GLMakie.hlines!(ax, [0.0], color = :black, linewidth = 1)
    GLMakie.ylims!(ax, -1.0, 1.0)

    return fig
end

function _shape_cloud_figure(elements, title_text; point_color = :steelblue)
    metrics = [compute_quality_metrics(el) for el in elements]
    compactness = [m.compactness for m in metrics]
    elongation = [m.elongation for m in metrics]

    fig = GLMakie.Figure(size = (650, 520))
    ax = GLMakie.Axis(
        fig[1, 1],
        title = title_text,
        xlabel = "elongation",
        ylabel = "compactness",
    )

    GLMakie.scatter!(ax, elongation, compactness; color = point_color, markersize = 10, strokewidth = 0.5, strokecolor = :white, transparency = true)

    return fig
end

function shape_clouds(seed::Int, nb_elements::Int)
    quads, _ = generate_random_quadrangles(nb_elements, seed = seed)
    tris, _ = generate_random_triangles(nb_elements, seed = seed)

    fig_quad = _shape_cloud_figure(quads, "Quadrangles - compactness vs elongation ($(length(quads)) elements)"; point_color = :teal)
    fig_tri = _shape_cloud_figure(tris, "Triangles - compactness vs elongation ($(length(tris)) elements)"; point_color = :darkorange)

    return fig_quad, fig_tri
end

function sampling(op, seed::Int, nb_elements::Int)
    epsilons = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
    total_steps = 2 * nb_elements * length(epsilons) * 3
    step = 0
    _print_progress(step, total_steps; prefix = "Global sampling")

    quads, quad_params = generate_random_quadrangles(nb_elements, seed=seed)
    summarize_quadrangle_set(quads, quad_params)

    # Generate triangles with parameters captured
    tris, tri_params = generate_random_triangles(nb_elements, seed=seed)
    summarize_triangle_set(tris, tri_params)

    cases_quad = Case[]

    for (i, el) in enumerate(quads)
        # Pass generation parameters to run_case_s
        gen_params = (aspect_ratio=quad_params[i].aspect_ratio, shear=quad_params[i].shear, trapezoid=quad_params[i].trapezoid)
        config1 = build_quad1n_config(el)
        config2 = build_quad4n_config(el)
        config3 = build_quad9n_config(el)
        for epsilon in epsilons
            cases_quad = union(cases_quad, run_case_s(config3, epsilon, op; generation_params=gen_params))
            step += 1
            _print_progress(step, total_steps; prefix = "Global sampling")
            cases_quad = union(cases_quad, run_case_s(config2, epsilon, op; generation_params=gen_params))
            step += 1
            _print_progress(step, total_steps; prefix = "Global sampling")
            cases_quad = union(cases_quad, run_case_s(config1, epsilon, op; generation_params=gen_params))
            step += 1
            _print_progress(step, total_steps; prefix = "Global sampling")
        end
    end

    cases_tri = Case[]

    for (i, el) in enumerate(tris)
        # Pass generation parameters to run_case_s
        gen_params = (aspect_ratio=tri_params[i].aspect_ratio, skewness=tri_params[i].skewness, elongation_gen=tri_params[i].elongation)
        config1 = build_tri1n_config(el)
        config2 = build_tri3n_config(el)
        config3 = build_tri6n_config(el)
        for epsilon in epsilons
            cases_tri = union(cases_tri, run_case_s(config3, epsilon, op; generation_params=gen_params))
            step += 1
            _print_progress(step, total_steps; prefix = "Global sampling")
            cases_tri = union(cases_tri, run_case_s(config2, epsilon, op; generation_params=gen_params))
            step += 1
            _print_progress(step, total_steps; prefix = "Global sampling")
            cases_tri = union(cases_tri, run_case_s(config1, epsilon, op; generation_params=gen_params))
            step += 1
            _print_progress(step, total_steps; prefix = "Global sampling")
        end
    end

    return cases_quad, cases_tri, quads, quad_params, tris, tri_params
end

function _nmax_dictionary_from_cases(cases::Vector{Case})
    nmax_by_element = Dict{Inti.LagrangeElement, Vector{Int}}()
    for c in cases
        push!(get!(nmax_by_element, c.el, Int[]), maximum(c.n_thetas))
    end
    return nmax_by_element
end

function _merge_nmax_dictionaries(dict_a::Dict{Inti.LagrangeElement, Vector{Int}}, dict_b::Dict{Inti.LagrangeElement, Vector{Int}})
    merged = Dict{Inti.LagrangeElement, Vector{Int}}()
    for (el, vals) in dict_a
        merged[el] = copy(vals)
    end
    for (el, vals) in dict_b
        if haskey(merged, el)
            append!(merged[el], vals)
        else
            merged[el] = copy(vals)
        end
    end
    return merged
end

function _element_polygon_xy(el::Inti.LagrangeElement)
    nodes = el.vals
    n = length(nodes)

    order = if n == 4
        [1, 2, 4, 3, 1]
    elseif n == 3
        [1, 2, 3, 1]
    else
        vcat(collect(1:n), 1)
    end

    x = [nodes[i][1] for i in order]
    y = [nodes[i][2] for i in order]
    return x, y
end

function _reference_point_to_physical(el::Inti.LagrangeSquare, ξ::SVector{2, Float64})
    u, v = ξ
    n1, n2, n3, n4 = el.vals
    return (1 - u) * (1 - v) * n1 + u * (1 - v) * n2 + (1 - u) * v * n3 + u * v * n4
end

function _reference_point_to_physical(el::Inti.LagrangeTriangle, ξ::SVector{2, Float64})
    u, v = ξ
    n1, n2, n3 = el.vals
    return n1 + u * (n2 - n1) + v * (n3 - n1)
end

function _get_polar_subtriangles_physical(el::Inti.LagrangeElement, ref_point::SVector{2, Float64})
    ref_shape = Inti.reference_domain(el)
    decompo = Inti.polar_decomposition(ref_shape, ref_point)

    source_physical = _reference_point_to_physical(el, ref_point)
    source_point = GLMakie.Point2f(source_physical[1], source_physical[2])

    triangles = []
    for (theta_min, theta_max, rho_func) in decompo
        rho_at_min = rho_func(theta_min)
        rho_at_max = rho_func(theta_max)

        x1_ref = ref_point[1] + rho_at_min * cos(theta_min)
        y1_ref = ref_point[2] + rho_at_min * sin(theta_min)
        v1_ref = SVector(x1_ref, y1_ref)

        x2_ref = ref_point[1] + rho_at_max * cos(theta_max)
        y2_ref = ref_point[2] + rho_at_max * sin(theta_max)
        v2_ref = SVector(x2_ref, y2_ref)

        v1_physical = _reference_point_to_physical(el, v1_ref)
        v2_physical = _reference_point_to_physical(el, v2_ref)

        v1 = GLMakie.Point2f(v1_physical[1], v1_physical[2])
        v2 = GLMakie.Point2f(v2_physical[1], v2_physical[2])

        push!(triangles, (source_point, v1, v2))
    end

    return triangles
end

function case_browser(cases::AbstractVector{<:Case}; sort_by::Symbol = :error, reverse_order::Bool = true)
    isempty(cases) && error("case_browser: empty case list")

    case_metric(c::Case) = sort_by == :nmax ? maximum(c.n_thetas) : sort_by == :mean_error ? mean(c.errors) : maximum(c.errors)
    entries = sort(copy(cases); by = case_metric, rev = reverse_order)

    n = length(entries)
    current_idx = GLMakie.Observable(1)

    fig = GLMakie.Figure(size = (980, 860))
    ax = GLMakie.Axis(
        fig[1, 1],
        xlabel = "x",
        ylabel = "y",
        title = "Case browser",
    )
    ax.aspect = GLMakie.DataAspect()

    polygon_obs = GLMakie.Observable(GLMakie.Point2f[])
    nodes_obs = GLMakie.Observable(GLMakie.Point2f[])
    source_obs = GLMakie.Observable(GLMakie.Point2f[])
    subtri_segments_obs = GLMakie.Observable(GLMakie.Point2f[])
    subtri_label_pos_obs = GLMakie.Observable(GLMakie.Point2f[])
    subtri_label_text_obs = GLMakie.Observable(String[])
    info_obs = GLMakie.Observable("")

    GLMakie.linesegments!(ax, subtri_segments_obs; color = (:seagreen, 0.55), linewidth = 2)
    GLMakie.lines!(ax, polygon_obs; color = :steelblue, linewidth = 3)
    GLMakie.scatter!(ax, nodes_obs; color = :darkorange, markersize = 12)
    GLMakie.scatter!(ax, source_obs; color = :crimson, marker = :star5, markersize = 18)
    GLMakie.text!(ax, subtri_label_pos_obs; text = subtri_label_text_obs, align = (:center, :center), fontsize = 14, color = :black)
    GLMakie.Label(fig[2, 1], info_obs; tellwidth = false, justification = :left)

    jump_grid = GLMakie.GridLayout(fig[3, 1])
    GLMakie.Label(jump_grid[1, 1], "Go to case #")
    jump_box = GLMakie.Textbox(jump_grid[1, 2], placeholder = "1..$(n)", stored_string = "1", width = 150)
    jump_button = GLMakie.Button(jump_grid[1, 3], label = "Go", width = 90)
    jump_feedback = GLMakie.Label(jump_grid[2, 1:3], "", tellwidth = false, justification = :left)

    all_x_coords = Float64[]
    all_y_coords = Float64[]
    for c in entries
        el = c.el
        x, y = _element_polygon_xy(el)
        append!(all_x_coords, x)
        append!(all_y_coords, y)
        source_physical = _reference_point_to_physical(el, c.point)
        push!(all_x_coords, source_physical[1])
        push!(all_y_coords, source_physical[2])
    end

    xmin_global = minimum(all_x_coords)
    xmax_global = maximum(all_x_coords)
    ymin_global = minimum(all_y_coords)
    ymax_global = maximum(all_y_coords)

    xpad = max(1e-8, 0.12 * max(xmax_global - xmin_global, 1.0))
    ypad = max(1e-8, 0.12 * max(ymax_global - ymin_global, 1.0))

    xmin_padded = xmin_global - xpad
    xmax_padded = xmax_global + xpad
    ymin_padded = ymin_global - ypad
    ymax_padded = ymax_global + ypad

    width_x = xmax_padded - xmin_padded
    width_y = ymax_padded - ymin_padded

    if width_x > width_y
        center_y = (ymin_padded + ymax_padded) / 2
        ymin_padded = center_y - width_x / 2
        ymax_padded = center_y + width_x / 2
    else
        center_x = (xmin_padded + xmax_padded) / 2
        xmin_padded = center_x - width_y / 2
        xmax_padded = center_x + width_y / 2
    end

    GLMakie.xlims!(ax, xmin_padded, xmax_padded)
    GLMakie.ylims!(ax, ymin_padded, ymax_padded)

    function refresh!(idx::Int)
        c = entries[idx]
        el = c.el
        x, y = _element_polygon_xy(el)

        polygon_obs[] = [GLMakie.Point2f(xi, yi) for (xi, yi) in zip(x, y)]
        vertex_points = [GLMakie.Point2f(xi, yi) for (xi, yi) in zip(x[1:end-1], y[1:end-1])]
        nodes_obs[] = vertex_points

        source_physical = _reference_point_to_physical(el, c.point)
        source_point = GLMakie.Point2f(source_physical[1], source_physical[2])
        source_obs[] = [source_point]

        polar_tris = _get_polar_subtriangles_physical(el, c.point)

        n_subtri = length(polar_tris)
        seg_points = GLMakie.Point2f[]
        label_pos = GLMakie.Point2f[]
        label_txt = String[]
        for i in 1:n_subtri
            src, v1, v2 = polar_tris[i]

            push!(seg_points, src, v1)
            push!(seg_points, v1, v2)
            push!(seg_points, v2, src)

            ctri = (src + v1 + v2) / 3
            push!(label_pos, ctri)
            nθ = i <= length(c.n_thetas) ? c.n_thetas[i] : missing
            push!(label_txt, i <= length(c.n_thetas) ? "nθ=$(nθ)" : "nθ=?")
        end
        subtri_segments_obs[] = seg_points
        subtri_label_pos_obs[] = label_pos
        subtri_label_text_obs[] = label_txt

        nmax_case = maximum(c.n_thetas)
        err_max = maximum(c.errors)
        err_mean = round(mean(c.errors), digits = 3)
        score = round(case_metric(c), digits = 3)
        compactness = round(get(c.quality_metrics, :compactness, NaN), digits = 4)
        kind = el isa Inti.LagrangeSquare ? "Quadrangle" : el isa Inti.LagrangeTriangle ? "Triangle" : string(typeof(el))

        info_obs[] = string(
            "Case ", idx, "/", n,
            " | type=", kind,
            " | epsilon=", c.epsilon,
            " | compactness=", compactness,
            " | nmax=", nmax_case,
            " | error_max=", err_max,
            " | error_mean=", err_mean,
            " | score(", sort_by, ")=", score,
            "\nSource point ref=", Tuple(round.(collect(c.point); digits = 4)),
            " | source point phys=", Tuple(round.([source_physical[1], source_physical[2], source_physical[3]]; digits = 4)),
            " | time=", round(c.time, digits = 4), " s",
            "\nSubtriangles=", n_subtri, " | n_thetas=", c.n_thetas,
            "\nControls: right arrow = next case, left arrow = previous case",
        )
    end

    on(current_idx) do idx
        refresh!(idx)
        jump_box.stored_string[] = string(idx)
        jump_feedback.text[] = ""
    end

    function jump_to_case!(input)
        idx = tryparse(Int, strip(String(input)))
        if isnothing(idx)
            jump_feedback.text[] = "Invalid input. Please enter an integer in [1, $(n)]."
            return
        end
        if idx < 1 || idx > n
            jump_feedback.text[] = "Out of range. Valid range is [1, $(n)]."
            return
        end
        current_idx[] = idx
    end

    on(jump_button.clicks) do _
        jump_to_case!(jump_box.displayed_string[])
    end

    on(GLMakie.events(fig).keyboardbutton) do ev
        if ev.action == GLMakie.Keyboard.press
            if ev.key == GLMakie.Keyboard.right
                current_idx[] = current_idx[] == n ? 1 : current_idx[] + 1
            elseif ev.key == GLMakie.Keyboard.left
                current_idx[] = current_idx[] == 1 ? n : current_idx[] - 1
            end
        end
    end

    refresh!(1)
    screen = display(GLMakie.Screen(), fig)
    return (screen = screen, figure = fig, index = current_idx, entries = entries, jump_box = jump_box)
end

function element_nmax_browser(nmax_by_element::Dict{Inti.LagrangeElement, Vector{Int}}; sort_by::Symbol = :max, reverse_order::Bool = true)
    isempty(nmax_by_element) && error("element_nmax_browser: empty dictionary")

    entries = collect(nmax_by_element)
    metric(vals) = sort_by == :mean ? mean(vals) : sort_by == :min ? minimum(vals) : sort_by == :p95 ? quantile(vals, 0.95) : maximum(vals)
    sort!(entries; by = kv -> metric(kv.second), rev = reverse_order)

    n = length(entries)
    current_idx = GLMakie.Observable(1)

    fig = GLMakie.Figure(size = (980, 780))
    ax = GLMakie.Axis(
        fig[1, 1],
        xlabel = "x",
        ylabel = "y",
        title = "Element browser",
    )
    ax.aspect = GLMakie.DataAspect()

    polygon_obs = GLMakie.Observable(GLMakie.Point2f[])
    nodes_obs = GLMakie.Observable(GLMakie.Point2f[])
    info_obs = GLMakie.Observable("")

    GLMakie.lines!(ax, polygon_obs; color = :steelblue, linewidth = 3)
    GLMakie.scatter!(ax, nodes_obs; color = :darkorange, markersize = 12)
    GLMakie.Label(fig[2, 1], info_obs; tellwidth = false, justification = :left)

    function refresh!(idx::Int)
        el, nmax_vals = entries[idx]
        x, y = _element_polygon_xy(el)

        polygon_obs[] = [GLMakie.Point2f(xi, yi) for (xi, yi) in zip(x, y)]
        nodes_obs[] = [GLMakie.Point2f(xi, yi) for (xi, yi) in zip(x[1:end-1], y[1:end-1])]

        xpad = max(1e-8, 0.1 * max(maximum(x) - minimum(x), 1.0))
        ypad = max(1e-8, 0.1 * max(maximum(y) - minimum(y), 1.0))
        GLMakie.xlims!(ax, minimum(x) - xpad, maximum(x) + xpad)
        GLMakie.ylims!(ax, minimum(y) - ypad, maximum(y) + ypad)

        m_min = minimum(nmax_vals)
        m_max = maximum(nmax_vals)
        m_mean = round(mean(nmax_vals), digits = 2)
        m_score = round(metric(nmax_vals), digits = 2)

        kind = el isa Inti.LagrangeSquare ? "Quadrangle" : el isa Inti.LagrangeTriangle ? "Triangle" : string(typeof(el))
        info_obs[] = string(
            "Element ", idx, "/", n,
            " | type=", kind,
            " | nmax count=", length(nmax_vals),
            " | min=", m_min,
            " | max=", m_max,
            " | mean=", m_mean,
            " | score(", sort_by, ")=", m_score,
            "\nControls: right arrow = next element, left arrow = previous element",
        )
    end

    on(current_idx) do idx
        refresh!(idx)
    end

    on(GLMakie.events(fig).keyboardbutton) do ev
        if ev.action == GLMakie.Keyboard.press
            if ev.key == GLMakie.Keyboard.right
                current_idx[] = current_idx[] == n ? 1 : current_idx[] + 1
            elseif ev.key == GLMakie.Keyboard.left
                current_idx[] = current_idx[] == 1 ? n : current_idx[] - 1
            end
        end
    end

    refresh!(1)
    screen = display(GLMakie.Screen(), fig)
    return (screen = screen, figure = fig, index = current_idx, entries = entries)
end

function correlation_matrices(op, seed::Int, nb_elements::Int)
    cases_quad, cases_tri, quads, quad_params, tris, tri_params = sampling(op, seed, nb_elements)
    X_quad = observation_matrix(cases_quad)
    X_tri = observation_matrix(cases_tri)
    nmax_quad = _nmax_dictionary_from_cases(cases_quad)
    nmax_tri = _nmax_dictionary_from_cases(cases_tri)
    nmax_by_element = _merge_nmax_dictionaries(nmax_quad, nmax_tri)
    
    names_quad = ["dist", "compactness", "min_angle", "side_ratio", "jacobian_ratio", "jacobian_at_point", "corner_dist", "target_error", "nmax"]
    names_tri = ["dist", "compactness", "min_angle", "side_ratio", "jacobian_ratio", "jacobian_at_point", "corner_dist", "target_error", "nmax"]
    all_cases = vcat(cases_quad, cases_tri)

    return X_quad, X_tri, names_quad, names_tri, quads, quad_params, tris, tri_params, nmax_by_element, all_cases
end

function plot_correlations(X_quad, X_tri, names_quad, names_tri)
    fig_quad, corr_quad, names_quad_corr = _correlation_heatmap_figure(X_quad, names_quad; figtitle = "Quadrangles - Spearman correlation matrix ($(size(X_quad, 1)) samples)")
    fig_tri, corr_tri, names_tri_corr = _correlation_heatmap_figure(X_tri, names_tri; figtitle = "Triangles - Spearman correlation matrix ($(size(X_tri, 1)) samples)")

    corr_quad_partial, names_quad_partial = _partial_spearman_matrix(X_quad, names_quad; control_names = ["target_error"], drop_controls = true)
    corr_tri_partial, names_tri_partial = _partial_spearman_matrix(X_tri, names_tri; control_names = ["target_error"], drop_controls = true)
    fig_quad_partial_matrix = _correlation_heatmap_from_matrix(corr_quad_partial, names_quad_partial; figtitle = "Quadrangles - Partial Spearman matrix | target_error ($(size(X_quad, 1)) samples)")
    fig_tri_partial_matrix = _correlation_heatmap_from_matrix(corr_tri_partial, names_tri_partial; figtitle = "Triangles - Partial Spearman matrix | target_error ($(size(X_tri, 1)) samples)")

    fig_quad_nmax = _nmax_correlation_figure(corr_quad, names_quad_corr; figtitle = "Quadrangles - Spearman correlation with nmax ($(size(X_quad, 1)) samples)")
    fig_tri_nmax = _nmax_correlation_figure(corr_tri, names_tri_corr; figtitle = "Triangles - Spearman correlation with nmax ($(size(X_tri, 1)) samples)")

    return fig_quad, fig_tri, fig_quad_partial_matrix, fig_tri_partial_matrix, fig_quad_nmax, fig_tri_nmax
end

function plot_correlations(op, seed::Int, nb_elements::Int)
    X_quad, X_tri, names_quad, names_tri, _, _, _, _, _, _ = correlation_matrices(op, seed, nb_elements)
    return plot_correlations(X_quad, X_tri, names_quad, names_tri)
end

function plot_nmax_partial_correlations(op, seed::Int, nb_elements::Int)
    X_quad, X_tri, names_quad, names_tri, _, _, _, _, _, _ = correlation_matrices(op, seed, nb_elements)
    return nmax_partial_correlation_analysis(X_quad, X_tri, names_quad, names_tri)
end

function plot_shape_clouds(seed::Int, nb_elements::Int)
    return shape_clouds(seed, nb_elements)
end
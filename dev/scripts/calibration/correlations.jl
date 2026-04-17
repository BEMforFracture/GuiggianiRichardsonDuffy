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

    return cases_quad, cases_tri
end

function correlation_matrices(op, seed::Int, nb_elements::Int)
    cases_quad, cases_tri = sampling(op, seed, nb_elements)
    X_quad = observation_matrix(cases_quad)
    X_tri = observation_matrix(cases_tri)
    
    names_quad = ["dist", "compactness", "min_angle", "side_ratio", "jacobian_ratio", "corner_dist", "target_error", "nmax"]
    names_tri = ["dist", "compactness", "min_angle", "side_ratio", "jacobian_ratio", "corner_dist", "target_error", "nmax"]

    return X_quad, X_tri, names_quad, names_tri
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
    X_quad, X_tri, names_quad, names_tri = correlation_matrices(op, seed, nb_elements)
    return plot_correlations(X_quad, X_tri, names_quad, names_tri)
end

function plot_nmax_partial_correlations(op, seed::Int, nb_elements::Int)
    X_quad, X_tri, names_quad, names_tri = correlation_matrices(op, seed, nb_elements)
    return nmax_partial_correlation_analysis(X_quad, X_tri, names_quad, names_tri)
end

function plot_shape_clouds(seed::Int, nb_elements::Int)
    return shape_clouds(seed, nb_elements)
end
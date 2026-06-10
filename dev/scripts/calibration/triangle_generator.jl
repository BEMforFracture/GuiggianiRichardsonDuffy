"""
Simplified Triangle Generator for Calibration

Generates triangles with controlled geometric parameters (no scale, no rotation).
Only shape parameters that affect quadrature difficulty are varied:
- aspect_ratio: base/height ratio
- skewness: horizontal offset of apex
- elongation: vertical distortion

Scale and rotation are irrelevant since:
- Integration is on reference element
- Source point is inside the element (relative position matters)

Author: Calibration study
Date: 2026-04-14
"""

using StaticArrays
using LinearAlgebra
using Random
using Distributions
using Inti

#===========================================================================================
CORE TRIANGLE CONSTRUCTION
===========================================================================================#

"""
    build_triangle(aspect_ratio, skewness, elongation; perturb=0.0, rng=Random.GLOBAL_RNG)

Build a triangle from 3 shape parameters (no scale, no rotation).

Arguments:
- `aspect_ratio`: Base/height ratio (> 1 = wider base, < 1 = narrower base)
- `skewness`: Horizontal offset of apex (0 = centered, ±1 = extreme offset)
- `elongation`: Vertical distortion factor (1.0 = equilateral-like, < 1 = flatter, > 1 = taller)
- `perturb`: Optional random perturbation scale for corners (default: 0)
- `rng`: Random number generator for perturbations

Construction:
1. Base vertices at y = -1 ± aspect_ratio
2. Apex at y = elongation * (1.0 + skewness_offset)
3. Optional corner perturbations
4. No rotation, no scaling (not needed for calibration)

Returns:
- Inti.LagrangeTriangle element
"""
function build_triangle(aspect_ratio, skewness, elongation; perturb=0.0, rng=Random.GLOBAL_RNG)
    a = aspect_ratio
    s = skewness
    e = elongation
    
    # Base vertices (in XY plane, Z=0)
    # Base at y = -1
    v1 = SVector(-a, -1.0, 0.0)           # Bottom-left
    v2 = SVector( a, -1.0, 0.0)           # Bottom-right
    
    # Apex at y = elongation
    # Horizontal position varies with skewness
    apex_x = s * 0.5 * a  # Skewness offset proportional to base width
    v3 = SVector(apex_x, e, 0.0)          # Top apex
    
    # Optional corner perturbations
    if perturb > 0
        δ1 = SVector(randn(rng)*perturb, randn(rng)*perturb, 0.0)
        δ2 = SVector(randn(rng)*perturb, randn(rng)*perturb, 0.0)
        δ3 = SVector(randn(rng)*perturb, randn(rng)*perturb, 0.0)
        
        v1 = v1 + δ1
        v2 = v2 + δ2
        v3 = v3 + δ3
    end
    
    return Inti.LagrangeTriangle((v1, v2, v3))
end

#===========================================================================================
RANDOM SAMPLING
===========================================================================================#

function _sample_with_extreme_bias_tri(rng, dist::Uniform; extreme_prob::Float64 = 0.45, tail_fraction::Float64 = 0.2)
    a = minimum(dist)
    b = maximum(dist)
    width = (b - a) * tail_fraction
    width <= 0 && return rand(rng, dist)

    if rand(rng) < extreme_prob
        if rand(rng) < 0.5
            return rand(rng, Uniform(a, a + width))
        else
            return rand(rng, Uniform(b - width, b))
        end
    end

    return rand(rng, dist)
end

function _sample_with_extreme_bias_tri(rng, dist::LogUniform; extreme_prob::Float64 = 0.45, tail_fraction::Float64 = 0.2)
    a = minimum(dist)
    b = maximum(dist)
    la = log(a)
    lb = log(b)
    width = (lb - la) * tail_fraction
    width <= 0 && return rand(rng, dist)

    if rand(rng) < extreme_prob
        if rand(rng) < 0.5
            return exp(rand(rng, Uniform(la, la + width)))
        else
            return exp(rand(rng, Uniform(lb - width, lb)))
        end
    end

    return rand(rng, dist)
end

"""
    generate_random_triangles(N::Int;
        aspect_ratio_range = (0.5, 2.0),
        skewness_range = (-1.0, 1.0),
        elongation_range = (0.5, 1.5),
        perturb = 0.0,
        log_uniform_aspect = true,
        seed = nothing
    )

Generate N triangles with random parameters.

Arguments:
- `N`: Number of triangles to generate
- `aspect_ratio_range`: (min, max) for aspect ratio
- `skewness_range`: (min, max) for skewness
- `elongation_range`: (min, max) for elongation
- `perturb`: Corner perturbation scale
- `log_uniform_aspect`: If true, use log-uniform distribution for aspect ratio
- `seed`: Random seed

Returns:
- `elements`: Vector of Inti.LagrangeTriangle elements
- `params`: Vector of NamedTuples with (aspect_ratio, skewness, elongation)
"""
function generate_random_triangles(N::Int;
    aspect_ratio_range = (0.35, 2.8),
    skewness_range = (-1.4, 1.4),
    elongation_range = (0.3, 1.9),
    perturb = 0.0,
    log_uniform_aspect = true,
    extreme_bias = true,
    extreme_prob = 0.45,
    tail_fraction = 0.2,
    seed = nothing
)
    rng = isnothing(seed) ? Random.GLOBAL_RNG : MersenneTwister(seed)
    
    # Distributions
    if log_uniform_aspect
        aspect_dist = LogUniform(aspect_ratio_range[1], aspect_ratio_range[2])
    else
        aspect_dist = Uniform(aspect_ratio_range[1], aspect_ratio_range[2])
    end
    skewness_dist = Uniform(skewness_range[1], skewness_range[2])
    elongation_dist = Uniform(elongation_range[1], elongation_range[2])
    
    elements = []
    params = []
    
    for i in 1:N
        if extreme_bias
            a = _sample_with_extreme_bias_tri(rng, aspect_dist; extreme_prob = extreme_prob, tail_fraction = tail_fraction)
            s = _sample_with_extreme_bias_tri(rng, skewness_dist; extreme_prob = extreme_prob, tail_fraction = tail_fraction)
            e = _sample_with_extreme_bias_tri(rng, elongation_dist; extreme_prob = extreme_prob, tail_fraction = tail_fraction)
        else
            a = rand(rng, aspect_dist)
            s = rand(rng, skewness_dist)
            e = rand(rng, elongation_dist)
        end
        
        el = build_triangle(a, s, e; perturb=perturb, rng=rng)
        push!(elements, el)
        push!(params, (aspect_ratio=a, skewness=s, elongation=e))
    end
    
    return elements, params
end

#===========================================================================================
QUALITY METRICS
===========================================================================================#

"""
    compute_quality_metrics(el::Inti.LagrangeTriangle)

Compute geometric quality metrics for a triangle.

Returns a NamedTuple with:
- `edge_lengths`: Tuple of 3 edge lengths
- `aspect_ratio_edges`: Max/min edge length ratio
- `compactness`: 4πA/P²
- `elongation`: sqrt(λmax/λmin) from vertex covariance
- `angles_deg`: Tuple of 3 interior angles (degrees)
- `min_angle`: Minimum interior angle (degrees)
- `max_angle`: Maximum interior angle (degrees)
- `area`: Element area
- `jacobian_ratio`: Max/min Jacobian determinant ratio (at reference corners)
- Generation parameters (if provided): `aspect_ratio`, `skewness`, `elongation`
"""
function compute_quality_metrics(el::Inti.LagrangeTriangle, generation_params::NamedTuple=NamedTuple())
    nodes = el.vals
    
    # Edges (vertices: v1 → v2 → v3 → v1)
    edges = [
        nodes[2] - nodes[1],  # Edge 1-2
        nodes[3] - nodes[2],  # Edge 2-3
        nodes[1] - nodes[3],  # Edge 3-1
    ]
    
    edge_lengths = tuple([norm(e) for e in edges]...)
    aspect_ratio_edges = maximum(edge_lengths) / minimum(edge_lengths)
    
    # Interior angles at each vertex
    # Vertex 1: angle between edge 3-1 (reversed) and edge 1-2
    # Vertex 2: angle between edge 1-2 (reversed) and edge 2-3
    # Vertex 3: angle between edge 2-3 (reversed) and edge 3-1
    
    angle_pairs = [
        (-edges[3], edges[1]),  # At vertex 1
        (-edges[1], edges[2]),  # At vertex 2
        (-edges[2], edges[3]),  # At vertex 3
    ]
    
    angles_rad = Float64[]
    for (v1, v2) in angle_pairs
        cos_angle = dot(v1, v2) / (norm(v1) * norm(v2))
        angle = acos(clamp(cos_angle, -1.0, 1.0))
        push!(angles_rad, angle)
    end
    
    angles_deg = tuple(rad2deg.(angles_rad)...)
    
    # Area (using cross product)
    area = 0.5 * norm(cross(nodes[2] - nodes[1], nodes[3] - nodes[1]))

    perimeter = sum(edge_lengths)
    compactness = perimeter > 0 ? (4 * π * area) / (perimeter^2) : NaN

    centroid = (nodes[1] + nodes[2] + nodes[3]) / 3
    m11 = 0.0
    m12 = 0.0
    m22 = 0.0
    for v in nodes
        dx = v[1] - centroid[1]
        dy = v[2] - centroid[2]
        m11 += dx * dx
        m12 += dx * dy
        m22 += dy * dy
    end
    M = Symmetric([m11 m12; m12 m22] ./ 3)
    λ = eigvals(M)
    λmin = max(minimum(λ), eps(Float64))
    λmax = maximum(λ)
    elongation = sqrt(λmax / λmin)
    
    # Jacobian measure at reference corners
    # For 2D elements in 3D, jacobian is 3×2, so we use the integration measure
    ref_corners = [
        SVector(0.0, 0.0),
        SVector(1.0, 0.0),
        SVector(0.0, 1.0),
    ]
    
    jac_measures = [Inti._integration_measure(Inti.jacobian(el, ξ)) for ξ in ref_corners]
    jacobian_ratio = maximum(jac_measures) / minimum(jac_measures)
    
    metrics = (
        edge_lengths = edge_lengths,
        aspect_ratio_edges = aspect_ratio_edges,
        compactness = compactness,
        elongation = elongation,
        angles_deg = angles_deg,
        min_angle = minimum(angles_deg),
        max_angle = maximum(angles_deg),
        area = area,
        jacobian_ratio = jacobian_ratio,
    )
    
    # Merge with generation parameters if provided
    if !isempty(generation_params)
        if haskey(generation_params, :aspect_ratio)
            generation_params = merge(generation_params, (aspect_ratio_gen = generation_params.aspect_ratio,))
        end
        if haskey(generation_params, :elongation)
            generation_params = merge(generation_params, (elongation_gen = generation_params.elongation,))
        end
        metrics = merge(metrics, generation_params)
    end
    
    return metrics
end

"""
    is_valid_triangle(el::Inti.LagrangeTriangle; min_angle=10.0, max_angle=170.0, min_area=1e-3)

Check if a triangle meets validity criteria.

Arguments:
- `el`: Triangle element
- `min_angle`: Minimum acceptable interior angle (degrees)
- `max_angle`: Maximum acceptable interior angle (degrees)
- `min_area`: Minimum acceptable area

Returns:
- `true` if valid, `false` otherwise
"""
function is_valid_triangle(el::Inti.LagrangeTriangle; min_angle=10.0, max_angle=170.0, min_area=1e-3)
    metrics = compute_quality_metrics(el)
    
    if metrics.area < min_area
        return false
    end
    
    if metrics.min_angle < min_angle || metrics.max_angle > max_angle
        return false
    end
    
    # Check for positive Jacobian measure everywhere (non-degenerate)
    ref_corners = [
        SVector(0.0, 0.0),
        SVector(1.0, 0.0),
        SVector(0.0, 1.0),
    ]
    
    for ξ in ref_corners
        jac_measure = Inti._integration_measure(Inti.jacobian(el, ξ))
        if jac_measure <= 0
            return false
        end
    end
    
    return true
end

#===========================================================================================
VISUALIZATION / SUMMARY
===========================================================================================#

"""
    summarize_triangle_set(elements, params)

Print summary statistics for a set of triangles.
"""
function summarize_triangle_set(elements, params)
    N = length(elements)
    println("="^70)
    println("TRIANGLE SET SUMMARY")
    println("="^70)
    println("Number of elements: $N")
    println()
    
    # Parameter statistics
    aspects = [get(p, :aspect_ratio, get(p, :aspect_ratio_edges, NaN)) for p in params]
    skews = [get(p, :skewness, NaN) for p in params]
    elongs = [get(p, :elongation, NaN) for p in params]
    
    println("Parameter Ranges:")
    if all(!isnan, aspects)
        println("  Aspect ratio: [$(minimum(aspects)), $(maximum(aspects))]")
    else
        println("  Aspect ratio: unavailable")
    end
    if all(!isnan, skews)
        println("  Skewness:    [$(minimum(skews)), $(maximum(skews))]")
    else
        println("  Skewness:    unavailable")
    end
    if all(!isnan, elongs)
        println("  Elongation:  [$(minimum(elongs)), $(maximum(elongs))]")
    else
        println("  Elongation:  unavailable")
    end
    println()
    
    # Quality metrics
    println("Computing quality metrics...")
    metrics = [compute_quality_metrics(el) for el in elements]
    
    ar_edges = [m.aspect_ratio_edges for m in metrics]
    min_angles = [m.min_angle for m in metrics]
    max_angles = [m.max_angle for m in metrics]
    areas = [m.area for m in metrics]
    jac_ratios = [m.jacobian_ratio for m in metrics]
    
    println("Quality Metrics:")
    println("  Edge aspect ratio: [$(minimum(ar_edges)), $(maximum(ar_edges))]")
    println("  Min angle (deg):   [$(minimum(min_angles)), $(maximum(min_angles))]")
    println("  Max angle (deg):   [$(minimum(max_angles)), $(maximum(max_angles))]")
    println("  Area:              [$(minimum(areas)), $(maximum(areas))]")
    println("  Jacobian ratio:    [$(minimum(jac_ratios)), $(maximum(jac_ratios))]")
    
    # Validity check
    n_valid = sum(is_valid_triangle(el) for el in elements)
    println()
    println("Validity: $n_valid / $N elements are valid")
    println("="^70)
end

function summarize_triangle_set(elements)
    metrics = [compute_quality_metrics(el) for el in elements]
    summarize_triangle_set(elements, metrics)
end

#===========================================================================================
EXAMPLE USAGE
===========================================================================================#

function demo()
    println("\n### Random Set (50 elements) ###")
    els_random, params_random = generate_random_triangles(50, seed=42)
    summarize_triangle_set(els_random, params_random)
end

# Uncomment to run demo
# demo()

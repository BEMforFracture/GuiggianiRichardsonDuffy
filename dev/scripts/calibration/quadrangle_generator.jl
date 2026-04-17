"""
Simplified Quadrangle Generator for Calibration

Generates quadrangles with controlled geometric parameters (no scale, no rotation).
Only shape parameters that affect quadrature difficulty are varied:
- aspect_ratio: width/height ratio
- shear: parallelogram distortion
- trapezoid: non-parallelism (trapezoidal effect)

Scale and rotation are irrelevant since:
- Integration is on reference element
- Source point is inside the element (relative position matters)
"""

using StaticArrays
using LinearAlgebra
using Random
using Distributions
using Inti

#===========================================================================================
CORE QUADRANGLE CONSTRUCTION
===========================================================================================#

"""
    build_quadrangle(aspect_ratio, shear, trapezoid; perturb=0.0, rng=Random.GLOBAL_RNG)

Build a quadrangle from 3 shape parameters (no scale, no rotation).

Arguments:
- `aspect_ratio`: Width/height ratio (> 1 = wider, < 1 = taller)
- `shear`: Horizontal shear parameter (parallelogram distortion)
- `trapezoid`: Trapezoidal effect (non-parallelism)
- `perturb`: Optional random perturbation scale for corners (default: 0)
- `rng`: Random number generator for perturbations

Construction:
1. Base corners on y = -1
2. Apply aspect ratio and shear (horizontal offsets)
3. Apply trapezoid as top-edge tilt (y3 != y4), so top/bottom are non-parallel
3. Optional corner perturbations
4. No rotation, no scaling (not needed for calibration)

Returns:
- Inti.LagrangeSquare element
"""
function build_quadrangle(aspect_ratio, shear, trapezoid; perturb=0.0, rng=Random.GLOBAL_RNG)
    a = aspect_ratio
    s = shear
    t = trapezoid
    
    # Base corners (in XY plane, Z=0)
    # Bottom edge fixed at y=-1.
    # Top edge is tilted using t so it is not parallel to the bottom edge when t != 0.
    y1 = SVector(-a, -1.0, 0.0)           # Bottom-left
    y2 = SVector( a, -1.0, 0.0)           # Bottom-right
    y3 = SVector(-a + s, 1.0 - t, 0.0)    # Top-left
    y4 = SVector( a + s, 1.0 + t, 0.0)    # Top-right
    
    # Optional corner perturbations
    if perturb > 0
        δ1 = SVector(randn(rng)*perturb, randn(rng)*perturb, 0.0)
        δ2 = SVector(randn(rng)*perturb, randn(rng)*perturb, 0.0)
        δ3 = SVector(randn(rng)*perturb, randn(rng)*perturb, 0.0)
        δ4 = SVector(randn(rng)*perturb, randn(rng)*perturb, 0.0)
        
        y1 = y1 + δ1
        y2 = y2 + δ2
        y3 = y3 + δ3
        y4 = y4 + δ4
    end
    
    return Inti.LagrangeSquare((y1, y2, y3, y4))
end

#===========================================================================================
RANDOM SAMPLING
===========================================================================================#

"""
    generate_random_quadrangles(N::Int;
        aspect_ratio_range = (0.5, 2.0),
        shear_range = (-0.5, 0.5),
        trapezoid_range = (-0.3, 0.3),
        perturb = 0.0,
        log_uniform_aspect = true,
        seed = nothing
    )

Generate N quadrangles with random parameters.

Arguments:
- `N`: Number of quadrangles to generate
- `aspect_ratio_range`: (min, max) for aspect ratio
- `shear_range`: (min, max) for shear
- `trapezoid_range`: (min, max) for trapezoid
- `perturb`: Corner perturbation scale
- `log_uniform_aspect`: If true, use log-uniform distribution for aspect ratio
- `seed`: Random seed

Returns:
- `elements`: Vector of Inti.LagrangeSquare elements
- `params`: Vector of NamedTuples with (aspect_ratio, shear, trapezoid)
"""
function generate_random_quadrangles(N::Int;
    aspect_ratio_range = (0.5, 2.0),
    shear_range = (-0.5, 0.5),
    trapezoid_range = (-0.3, 0.3),
    perturb = 0.0,
    log_uniform_aspect = true,
    seed = nothing
)
    rng = isnothing(seed) ? Random.GLOBAL_RNG : MersenneTwister(seed)
    
    # Distributions
    if log_uniform_aspect
        aspect_dist = LogUniform(aspect_ratio_range[1], aspect_ratio_range[2])
    else
        aspect_dist = Uniform(aspect_ratio_range[1], aspect_ratio_range[2])
    end
    shear_dist = Uniform(shear_range[1], shear_range[2])
    trap_dist = Uniform(trapezoid_range[1], trapezoid_range[2])
    
    elements = []
    params = []
    
    for i in 1:N
        a = rand(rng, aspect_dist)
        s = rand(rng, shear_dist)
        t = rand(rng, trap_dist)
        
        el = build_quadrangle(a, s, t; perturb=perturb, rng=rng)
        push!(elements, el)
        push!(params, (aspect_ratio=a, shear=s, trapezoid=t))
    end
    
    return elements, params
end

#===========================================================================================
QUALITY METRICS
===========================================================================================#

"""
    compute_quality_metrics(el::Inti.LagrangeSquare, generation_params::NamedTuple=NamedTuple())

Compute geometric quality metrics for a quadrangle.

Returns a NamedTuple with:
- `edge_lengths`: Tuple of 4 edge lengths
- `aspect_ratio_edges`: Max/min edge length ratio
- `compactness`: 4πA/P²
- `elongation`: sqrt(λmax/λmin) from vertex covariance
- `angles_deg`: Tuple of 4 interior angles (degrees)
- `min_angle`: Minimum interior angle (degrees)
- `max_angle`: Maximum interior angle (degrees)
- `area`: Element area
- `jacobian_ratio`: Max/min Jacobian determinant ratio (at corners)
- Generation parameters (if provided): `aspect_ratio`, `shear`, `trapezoid`
"""
function compute_quality_metrics(el::Inti.LagrangeSquare, generation_params::NamedTuple=NamedTuple())
    nodes = el.vals
    
    # Edges (assuming counterclockwise ordering: 1→2→4→3→1)
    edges = [
        nodes[2] - nodes[1],  # Edge 1-2
        nodes[4] - nodes[2],  # Edge 2-4
        nodes[3] - nodes[4],  # Edge 4-3
        nodes[1] - nodes[3],  # Edge 3-1
    ]
    
    edge_lengths = tuple([norm(e) for e in edges]...)
    aspect_ratio_edges = maximum(edge_lengths) / minimum(edge_lengths)
    
    # Interior angles
    # At each vertex, angle between incoming and outgoing edges
    # Vertex 1: between edges 3-1 and 1-2
    # Vertex 2: between edges 1-2 and 2-4
    # Vertex 3: between edges 4-3 and 3-1
    # Vertex 4: between edges 2-4 and 4-3
    
    angle_pairs = [
        (edges[4], edges[1]),  # At vertex 1
        (edges[1], edges[2]),  # At vertex 2
        (edges[3], edges[4]),  # At vertex 3
        (edges[2], edges[3]),  # At vertex 4
    ]
    
    angles_rad = Float64[]
    for (e_in, e_out) in angle_pairs
        # Angle between -e_in and e_out
        v1 = -e_in
        v2 = e_out
        cos_angle = dot(v1, v2) / (norm(v1) * norm(v2))
        angle = acos(clamp(cos_angle, -1.0, 1.0))
        push!(angles_rad, angle)
    end
    
    angles_deg = tuple(rad2deg.(angles_rad)...)
    
    # Area (using cross product, split into two triangles)
    area1 = 0.5 * norm(cross(nodes[2] - nodes[1], nodes[3] - nodes[1]))
    area2 = 0.5 * norm(cross(nodes[4] - nodes[2], nodes[3] - nodes[2]))
    area = area1 + area2

    perimeter = sum(edge_lengths)
    compactness = perimeter > 0 ? (4 * π * area) / (perimeter^2) : NaN

    centroid = (nodes[1] + nodes[2] + nodes[3] + nodes[4]) / 4
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
    M = Symmetric([m11 m12; m12 m22] ./ 4)
    λ = eigvals(M)
    λmin = max(minimum(λ), eps(Float64))
    λmax = maximum(λ)
    elongation = sqrt(λmax / λmin)
    
    # Jacobian measure at reference corners
    # For 2D elements in 3D, jacobian is 3×2, so we use the integration measure
    ref_corners = [
        SVector(-1.0, -1.0),
        SVector( 1.0, -1.0),
        SVector(-1.0,  1.0),
        SVector( 1.0,  1.0),
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
        metrics = merge(metrics, generation_params)
    end
    
    return metrics
end

"""
    is_valid_quadrangle(el::Inti.LagrangeSquare; min_angle=10.0, max_angle=170.0, min_area=1e-3)

Check if a quadrangle meets validity criteria.

Arguments:
- `el`: Quadrangle element
- `min_angle`: Minimum acceptable interior angle (degrees)
- `max_angle`: Maximum acceptable interior angle (degrees)
- `min_area`: Minimum acceptable area

Returns:
- `true` if valid, `false` otherwise
"""
function is_valid_quadrangle(el::Inti.LagrangeSquare; min_angle=10.0, max_angle=170.0, min_area=1e-3)
    metrics = compute_quality_metrics(el)
    
    if metrics.area < min_area
        return false
    end
    
    if metrics.min_angle < min_angle || metrics.max_angle > max_angle
        return false
    end
    
    # Check for positive Jacobian measure everywhere (non-crossing/non-degenerate)
    ref_corners = [
        SVector(-1.0, -1.0),
        SVector( 1.0, -1.0),
        SVector(-1.0,  1.0),
        SVector( 1.0,  1.0),
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
    summarize_quadrangle_set(elements, params)

Print summary statistics for a set of quadrangles.
"""
function summarize_quadrangle_set(elements, params)
    N = length(elements)
    println("="^70)
    println("QUADRANGLE SET SUMMARY")
    println("="^70)
    println("Number of elements: $N")
    println()
    
    # Parameter statistics
    aspects = [get(p, :aspect_ratio, get(p, :aspect_ratio_edges, NaN)) for p in params]
    shears = [get(p, :shear, NaN) for p in params]
    traps = [get(p, :trapezoid, NaN) for p in params]
    
    println("Parameter Ranges:")
    if all(!isnan, aspects)
        println("  Aspect ratio: [$(minimum(aspects)), $(maximum(aspects))]")
    else
        println("  Aspect ratio: unavailable")
    end
    if all(!isnan, shears)
        println("  Shear:        [$(minimum(shears)), $(maximum(shears))]")
    else
        println("  Shear:        unavailable")
    end
    if all(!isnan, traps)
        println("  Trapezoid:    [$(minimum(traps)), $(maximum(traps))]")
    else
        println("  Trapezoid:    unavailable")
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
    n_valid = sum(is_valid_quadrangle(el) for el in elements)
    println()
    println("Validity: $n_valid / $N elements are valid")
    println("="^70)
end

function summarize_quadrangle_set(elements)
    metrics = [compute_quality_metrics(el) for el in elements]
    summarize_quadrangle_set(elements, metrics)
end

#===========================================================================================
EXAMPLE USAGE
===========================================================================================#

function demo()
    println("\n### Random Set (50 elements) ###")
    els_random, params_random = generate_random_quadrangles(50, seed=42)
    summarize_quadrangle_set(els_random, params_random)
end

# Uncomment to run demo
# demo()

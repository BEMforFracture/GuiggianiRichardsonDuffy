import GuiggianiRichardsonDuffy as GRD
using Inti

const QUAD_RHO = Inti.GaussLegendre(5)

abstract type AbstractCase end

mutable struct Case{T} <: AbstractCase
    el::Inti.LagrangeElement
    point::SVector{2, Float64}
    quality_metrics::Dict{Symbol, Any} # aspect_ratio, skewness, jacobian_ratio
    epsilon::Float64 # tolerance for the calibration
    n_thetas::Vector{Int} # for each subtriangle, number of theta points needed for reaching desired accuracy
    closed_form::Union{Nothing, T} # closed-form value of the integral if known, otherwise nothing
    computed_integral::T # value of the integral computed with the quadrature method
    errors::Vector{Float64} # error for each subtriangle compared to the closed-form value (if known)
    time::Float64 # time taken to compute the integral
end

Case(el::Inti.LagrangeElement, point::SVector{2, Float64}, quality_metrics::Dict{Symbol, Any}, epsilon::Float64, n_thetas::Vector{Int}, closed_form::Union{Nothing, T}, computed_integral::T, errors::Vector{Float64}, time::Float64) where T = Case{T}(el, point, quality_metrics, epsilon, n_thetas, closed_form, computed_integral, errors, time)

function Base.show(io::IO, case::Case)
    println(io, "Case for element: $(typeof(case.el))")
    println(io, "Point: $(case.point)")
    println(io, "Quality metrics:")
    for (name, value) in pairs(case.quality_metrics)
        println(io, "\t  $name: $value")
    end
    println(io, "Epsilon: $(case.epsilon)")
    println(io, "Number of theta points per subtriangle: $(case.n_thetas)")
    println(io, "Computed integral: $(case.computed_integral)")
    println(io, "Relative errors per subtriangle: $(case.errors)")
    println(io, "Time taken: $(case.time) seconds")
end

function _compute_n_theta(K, û, x̂, el::Inti.LagrangeElement, ori, quad_rho, method::GRD.AbstractMethod, epsilon)
    T = Inti.return_type(K)
    ref_shape = Inti.reference_domain(el)
    decompo = Inti.polar_decomposition(ref_shape, x̂)
    n_triangles = length(decompo)
    errors = fill(Inf, n_triangles)
    ref_value = nothing
    n_thetas = ones(Int, n_triangles)
    quad_thetas = Vector{Inti.GaussLegendre}(undef, n_triangles)
    for i in 1:n_triangles
        quad_thetas[i] = Inti.GaussLegendre(n_thetas[i])
    end
    vals = Vector{T}(undef, n_triangles)
    for i in 1:n_triangles
        error = errors[i]
        quad_theta = quad_thetas[i]
        previous_integral = zero(T)
        n_theta = n_thetas[i]
        while error > epsilon && n_theta <= 100
            computed_integral = GRD.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_thetas, method)
            error = norm(computed_integral - previous_integral) / norm(computed_integral)
            n_theta += 1
            quad_theta = Inti.GaussLegendre(n_theta)
            quad_thetas[i] = quad_theta
            previous_integral = computed_integral
        end
        n_thetas[i] = n_theta
        errors[i] = error
        vals[i] = previous_integral
        quad_thetas[i] = quad_theta
        if n_theta > 100
            @warn "Maximum number of theta points reached for subtriangle $i without reaching desired accuracy. ϵ = $epsilon, final error = $error"
        end
    end
    final = @timed GRD.guiggiani_singular_integral(K, û, x̂, el, ori, quad_rho, quad_thetas, method)
    val = final.value
    time = final.time
    return n_thetas, val, ref_value, errors, time
end

function run_case_s(config::AbstractConfig, epsilon::Float64, op; generation_params::NamedTuple=NamedTuple())
    el = config.el
    metrics = compute_quality_metrics(config.el, generation_params)
    d = Dict{Symbol, Any}()
    for (name, value) in pairs(metrics)
        d[name] = value
    end
    û = ξ -> 1.0
    K = Inti.HyperSingularKernel(op)
    method = GRD.AutoDiffExpansion()

    n_points = hasfield(typeof(config), :point) ? 1 : length(config.points)
    case_s = Vector{Case}(undef, n_points)

    if n_points == 1
        x̂ = config.point
        n_thetas, computed_integral, return_value, errors, time = _compute_n_theta(K, û, x̂, el, 1, QUAD_RHO, method, epsilon)
        case = Case(el, x̂, d, epsilon, n_thetas, return_value, computed_integral, errors, time)
        case_s = [case]
    else
        for i in 1:n_points
            x̂ = config.points[i]
            n_thetas, final_value, ref_value, errors, time = _compute_n_theta(K, û, x̂, el, 1, QUAD_RHO, method, epsilon)
            case_s[i] = Case(el, x̂, d, epsilon, n_thetas, ref_value, final_value, errors, time)
        end
    end
    return case_s
end

function dist_to_border(::Inti.ReferenceHyperCube, point::SVector{2, Float64})
    return min(
        min(point[1], 1 - point[1]),
        min(point[2], 1 - point[2])
    )
end

function dist_to_border(::Inti.ReferenceTriangle, point::SVector{2, Float64})
    x, y = point
    return min(x, y, 1 - x - y)
end

function dist_to_nearest_corner(::Inti.ReferenceHyperCube, point::SVector{2, Float64})
    corners = [
        SVector(-1.0, -1.0),
        SVector( 1.0, -1.0),
        SVector(-1.0,  1.0),
        SVector( 1.0,  1.0),
    ]
    return minimum(norm(point - corner) for corner in corners)
end

function dist_to_nearest_corner(::Inti.ReferenceTriangle, point::SVector{2, Float64})
    corners = [
        SVector(0.0, 0.0),
        SVector(1.0, 0.0),
        SVector(0.0, 1.0),
    ]
    return minimum(norm(point - corner) for corner in corners)
end

function _extract_case_features(case::Case)
    """
    Extract compact shared features for both triangles and quadrangles.

    Returns a 9-element vector:
    1. dist: minimum distance from source point to element boundary
    2. compactness: 4πA/P²
    3. min_angle: minimum interior angle (degrees)
    4. side_ratio: largest edge length / smallest edge length
    5. jacobian_ratio: max/min Jacobian integration measure ratio
    6. jacobian_at_point: Jacobian integration measure at source point
    7. corner_dist: distance to nearest reference corner
    8. target_error: target accuracy epsilon
    9. nmax: maximum number of theta quadrature points needed
    """
    el = case.el
    ref_el = Inti.reference_domain(el)
    dist = dist_to_border(ref_el, case.point)

    qm = case.quality_metrics
    compactness = get(qm, :compactness, NaN)
    min_angle = get(qm, :min_angle, NaN)
    side_ratio = get(qm, :aspect_ratio_edges, NaN)
    jacobian_ratio = get(qm, :jacobian_ratio, NaN)
    jacobian_at_point = Inti._integration_measure(Inti.jacobian(el, case.point))
    corner_dist = dist_to_nearest_corner(ref_el, case.point)
    target_error = case.epsilon
    nmax = maximum(case.n_thetas)
    return [dist, compactness, min_angle, side_ratio, jacobian_ratio, jacobian_at_point, corner_dist, target_error, nmax]
end

function number_of_variables()
    return 9  # [dist, compactness, min_angle, side_ratio, jacobian_ratio, jacobian_at_point, corner_dist, target_error, nmax]
end

function observation_matrix(cases::Vector{Case})
    n_cases = length(cases)
    n_var = number_of_variables()
    M = zeros(n_cases, n_var)
    for i in 1:n_cases
        M[i, :] = _extract_case_features(cases[i])
    end
    return M
end

function demo()
    y1 = SVector(0.0, 0.0, 0.0)
    y2 = SVector(1.0, 0.0, 0.0)
    y3 = SVector(0.0, 1.0, 0.0)
    y4 = SVector(1.0, 1.0, 0.0)
    el = Inti.LagrangeSquare(y1, y2, y3, y4)
    config1 = build_quad1n_config(el)
    config2 = build_quad4n_config(el)
    config3 = build_quad9n_config(el)
    epsilons = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
    cases = run_case_s(config, epsilon, Inti.Laplace(; dim = 3))
    return cases
end

# cases = demo()
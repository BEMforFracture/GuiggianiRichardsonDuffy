import GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using LinearAlgebra
using GLMakie
using ForwardDiff

# ============================================================
# Laplace hypersingular: diagnostic Polar vs Duffy integrands
# Goal: explain why Duffy may require more points in b than Polar in theta
# ============================================================

# Inputs
xhat = SVector(0.1, 0.1)
ori = 1

# Method for Laurent coefficients
method = GRD.AnalyticalExpansion()

n_rho = 6
n_a = 6
nsamples = 1000

quad_rho = Inti.GaussLegendre(n_rho)
quad_a = Inti.GaussLegendre(n_a)

# Setup element
delta = 0.5
z = 0.0
y1 = SVector(-1.0, -1.0, z)
y2 = SVector(1.0 + delta, -1.0, z)
y3 = SVector(-1.0, 1.0, z)
y4 = SVector(1.0 - delta, 1.0, z)
nodes = (y1, y2, y3, y4)

el = Inti.LagrangeSquare(nodes)
ref_domain = Inti.reference_domain(el)
uhat = xi -> 1.0

# Kernel setup
K_base = Inti.HyperSingularKernel(Inti.Laplace(dim = 3))
SK = GRD.SplitKernel(K_base)
Kprod = (qx, qy) -> prod(SK(qx, qy))

x = el(xhat)
Dtau = Inti.jacobian(el, xhat)
nx = Inti._normal(Dtau, ori)
D2tau = Inti.hessian(el, xhat)
qx = (coords = x, normal = nx)
N = length(x)

# -------------------------
# Polar G1(theta), G2(theta)
# -------------------------
K_polar = GRD.polar_kernel_fun(Kprod, el, uhat, xhat, ori)
rho_max_fun = GRD.rho_fun(ref_domain, xhat)
L = GRD.laurents_coeffs(K_base, el, ori, uhat, xhat, method)

function G1_polar(theta)
    fm2, fm1 = L(theta)
    rho_max = rho_max_fun(theta)
    Irho = quad_rho() do (rho_ref,)
        rho = rho_max * rho_ref
        K_polar(rho, theta) - fm2 / rho^2 - fm1 / rho
    end
    return Irho * rho_max
end

function G2_polar(theta)
    fm2, fm1 = L(theta)
    rho_max = rho_max_fun(theta)
    return fm1 * log(rho_max) - fm2 / rho_max
end

# -------------------------
# Duffy G1(b), G2(b)
# -------------------------
function return_vertices(tau)
    if tau == 1
        return SVector(1.0, 0.0), SVector(1.0, 1.0)
    elseif tau == 2
        return SVector(1.0, 1.0), SVector(0.0, 1.0)
    elseif tau == 3
        return SVector(0.0, 1.0), SVector(0.0, 0.0)
    else
        return SVector(0.0, 0.0), SVector(1.0, 0.0)
    end
end

function duffy_decomposition(::Inti.ReferenceSquare)
    xiI, xiII = return_vertices(1)
    _, xiIII = return_vertices(2)
    _, xiIV = return_vertices(3)
    return (
        (xiI, xiII, 1),
        (xiII, xiIII, 2),
        (xiIII, xiIV, 3),
        (xiIV, xiI, 4),
    )
end

function local_geometry_data(xiI, xiII, b)
    c = b * (xiII - xiI) + xiI - xhat
    A = Dtau * c
    nA = norm(A)
    B = ntuple(i -> transpose(c) * D2tau[i, :, :] * c, N) |> SVector
    beta = 1 / nA
    gamma_over_beta_squared = -(A ⋅ B) / nA^2
    return c, A, B, beta, gamma_over_beta_squared
end

function fcore(a, c, A, B, surface)
    yhat = xhat + a * c
    jac_y = Inti.jacobian(el, yhat)
    ny = Inti._normal(jac_y, ori)
    y = el(yhat)
    qy = (coords = y, normal = ny)
    muy = Inti._integration_measure(jac_y)

    AB = A + a / 2 * B
    Ahat = AB / norm(AB)
    _, Khat = SK(qx, qy, Ahat)
    v = uhat(yhat)

    return Khat * v * surface * muy / norm(AB)^3
end

function singular_coeffs(f)
    f_m2 = f(0.0)
    f_m1 = ForwardDiff.derivative(f, 0.0)
    return f_m1, f_m2
end

function G1_duffy(xiI, xiII, b)
    surface = (xiI[1] - xhat[1]) * (xiII[2] - xiI[2]) - (xiII[1] - xiI[1]) * (xiI[2] - xhat[2])
    c, A, B, _, _ = local_geometry_data(xiI, xiII, b)

    f = a -> fcore(a, c, A, B, surface)
    f_m1, f_m2 = singular_coeffs(f)

    return quad_a() do (a,)
        (f(a) - f_m2 - a * f_m1) / a^2
    end
end

function G2_duffy(xiI, xiII, b)
    surface = (xiI[1] - xhat[1]) * (xiII[2] - xiI[2]) - (xiII[1] - xiI[1]) * (xiI[2] - xhat[2])
    c, A, B, beta, gamma_over_beta_squared = local_geometry_data(xiI, xiII, b)

    f = a -> fcore(a, c, A, B, surface)
    f_m1, f_m2 = singular_coeffs(f)

    return -f_m1 * log(abs(beta)) - f_m2 * (gamma_over_beta_squared + 1)
end

# -------------------------
# Diagnostics helpers
# -------------------------
angle01(theta) = theta < 0 ? theta + 2pi : theta

function unwrap_angles(thetas)
    out = copy(thetas)
    for i in 2:length(out)
        d = out[i] - out[i - 1]
        if d > pi
            out[i:end] .-= 2pi
        elseif d < -pi
            out[i:end] .+= 2pi
        end
    end
    return out
end

function curve_metrics(vals, xs)
    y = abs.(vals)
    dy = diff(y)
    dx = diff(xs)
    tv = sum(abs.(dy))
    maxslope = maximum(abs.(dy) ./ dx)
    osc = maximum(y) - minimum(y)
    return (tv = tv, maxslope = maxslope, osc = osc, ymin = minimum(y), ymax = maximum(y))
end

function dtheta_db(thetas, bs)
    dth = diff(thetas)
    db = diff(bs)
    mids = (bs[1:end-1] + bs[2:end]) / 2
    return mids, dth ./ db
end

# -------------------------
# Main diagnostic
# -------------------------
bs = collect(range(0.0, 1.0, length = nsamples))
sectors = duffy_decomposition(ref_domain)

fig_curves = Figure(size = (1800, 1300))
fig_maps = Figure(size = (1200, 1000))

for (k, (xiI, xiII, tau)) in enumerate(sectors)
    # b -> theta mapping induced by the edge segment in reference space
    cs = [b * (xiII - xiI) + xiI - xhat for b in bs]
    thetas_raw = [angle01(atan(c[2], c[1])) for c in cs]
    thetas = unwrap_angles(thetas_raw)

    # Native polar variable in this sector
    theta_native = collect(range(thetas[1], thetas[end], length = nsamples))
    t_native = collect(range(0.0, 1.0, length = nsamples))

    # Polar curves
    G1p_native = [G1_polar(mod2pi(th)) for th in theta_native]
    G2p_native = [G2_polar(mod2pi(th)) for th in theta_native]
    G1p_composed = [G1_polar(mod2pi(th)) for th in thetas]
    G2p_composed = [G2_polar(mod2pi(th)) for th in thetas]

    # Duffy curves
    G1d = [G1_duffy(xiI, xiII, b) for b in bs]
    G2d = [G2_duffy(xiI, xiII, b) for b in bs]

    # Metrics
    m_G1_p_native = curve_metrics(G1p_native, t_native)
    m_G1_p_comp = curve_metrics(G1p_composed, bs)
    m_G1_d = curve_metrics(G1d, bs)

    m_G2_p_native = curve_metrics(G2p_native, t_native)
    m_G2_p_comp = curve_metrics(G2p_composed, bs)
    m_G2_d = curve_metrics(G2d, bs)

    ratio_G1_tv = m_G1_d.tv / max(m_G1_p_native.tv, eps())
    ratio_G1_slope = m_G1_d.maxslope / max(m_G1_p_native.maxslope, eps())
    ratio_G2_tv = m_G2_d.tv / max(m_G2_p_native.tv, eps())
    ratio_G2_slope = m_G2_d.maxslope / max(m_G2_p_native.maxslope, eps())

    println("\n=== Secteur tau = $tau ===")
    println("G1 | TV polar(native)=$(m_G1_p_native.tv), TV polar(theta(b))=$(m_G1_p_comp.tv), TV duffy=$(m_G1_d.tv), ratio D/P(native)=$ratio_G1_tv")
    println("G1 | maxslope polar(native)=$(m_G1_p_native.maxslope), maxslope duffy=$(m_G1_d.maxslope), ratio D/P(native)=$ratio_G1_slope")
    println("G2 | TV polar(native)=$(m_G2_p_native.tv), TV polar(theta(b))=$(m_G2_p_comp.tv), TV duffy=$(m_G2_d.tv), ratio D/P(native)=$ratio_G2_tv")
    println("G2 | maxslope polar(native)=$(m_G2_p_native.maxslope), maxslope duffy=$(m_G2_d.maxslope), ratio D/P(native)=$ratio_G2_slope")

    # Curves figure
    ax1 = Axis(fig_curves[k, 1], xlabel = "parametre normalise", ylabel = "|G1|", title = "Secteur $tau : G1")
    lines!(ax1, t_native, abs.(G1p_native), color = :black, label = "Polar (theta natif)")
    lines!(ax1, bs, abs.(G1p_composed), color = :orange, linestyle = :dash, label = "Polar compose theta(b)")
    lines!(ax1, bs, abs.(G1d), color = :blue, label = "Duffy (b)")
    axislegend(ax1, position = :lt)

    ax2 = Axis(fig_curves[k, 2], xlabel = "parametre normalise", ylabel = "|G2|", title = "Secteur $tau : G2")
    lines!(ax2, t_native, abs.(G2p_native), color = :black, label = "Polar (theta natif)")
    lines!(ax2, bs, abs.(G2p_composed), color = :orange, linestyle = :dash, label = "Polar compose theta(b)")
    lines!(ax2, bs, abs.(G2d), color = :blue, label = "Duffy (b)")
    axislegend(ax2, position = :lt)

    mids, dthdb = dtheta_db(thetas, bs)
    ax3 = Axis(fig_curves[k, 3], xlabel = "b", ylabel = "dtheta/db", title = "Secteur $tau : distorsion de parametrisation")
    lines!(ax3, mids, abs.(dthdb), color = :red, label = "|dtheta/db|")
    axislegend(ax3, position = :lt)

    # Map figure: theta(b)
    row = div(k - 1, 2) + 1
    col = mod(k - 1, 2) + 1
    axm = Axis(fig_maps[row, col], xlabel = "b", ylabel = "theta(b)", title = "Secteur $tau")
    lines!(axm, bs, thetas, color = :purple)
end

display(GLMakie.Screen(), fig_curves)
display(GLMakie.Screen(), fig_maps)

# GLMakie.save("./dev/figures/laplace/laplace_duffy_polar_integrand_diagnostic_curves.png", fig_curves)
# GLMakie.save("./dev/figures/laplace/laplace_duffy_polar_integrand_diagnostic_theta_map.png", fig_maps)

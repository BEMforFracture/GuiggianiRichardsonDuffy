import GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using LinearAlgebra
using GLMakie

# Configuration
x̂ = SVector(0.3, 0.3)  # Source point in reference coordinates

# Method for Laurent coefficients
method = GRD.AnalyticalExpansion()

# Quadrature parameters
n_rho = 10
quad_rho = Inti.GaussLegendre(n_rho)
target_rel_tol = 1e-8
nmin_quad = 2
nmax_quad = 120

# Setup element
δ = 0.5
z = 0.0
y¹ = SVector(-1.0, -1.0, z)
y² = SVector(1.0 + δ, -1.0, z)
y³ = SVector(-1.0, 1.0, z)
y⁴ = SVector(1.0 - δ, 1.0, z)
nodes = (y¹, y², y³, y⁴)
el = Inti.LagrangeSquare(nodes)

ori = 1

# Density function
û = ξ -> 1.0

# Kernel setup
K_base = Inti.HyperSingularKernel(Inti.Laplace(dim = 3))
K = GRD.SplitKernel(K_base)
Kprod = (qx, qy) -> prod(K(qx, qy))

# Polar coordinates setup
K_polar = GRD.polar_kernel_fun(Kprod, el, û, x̂, ori)
ref_domain = Inti.reference_domain(el)
ρ_max_fun = GRD.rho_fun(ref_domain, x̂)

# Compute Laurent coefficients
ℒ = GRD.laurents_coeffs(K_base, el, ori, û, x̂, method)

decompo = Inti.polar_decomposition(ref_domain, x̂)

function G₁(θ)
	f₋₂, f₋₁ = ℒ(θ)
	ρ_max = ρ_max_fun(θ)
	I_rho = quad_rho() do (rho_ref,)
		ρ = ρ_max * rho_ref
		return K_polar(ρ, θ) - f₋₂ / ρ^2 - f₋₁ / ρ
	end
	return I_rho * ρ_max
end

function G₂(θ)
	f₋₂, f₋₁ = ℒ(θ)
	ρ_max = ρ_max_fun(θ)
	return f₋₁ * log(ρ_max) - f₋₂ / ρ_max
end

function G(θ)
	return G₁(θ) + G₂(θ)
end

function integrate_over_theta(decompo, G)
    I_G = zero(G(0.0))
    I_T = zero(G(0.0))

    for (theta_min, theta_max, _) in decompo
        Δθ = theta_max - theta_min
        n_theta = 10
        quad_theta_gauss = Inti.GaussLegendre(n_theta)
        quad_theta_trap = Inti.Trapezoid(n_theta)

        I_G += Δθ * quad_theta_gauss() do (θ_ref,)
            θ = theta_min + Δθ * θ_ref
            return G(θ)
        end
        I_T += Δθ * quad_theta_trap() do (θ_ref,)
            θ = theta_min + Δθ * θ_ref
            return G(θ)
        end
    end

    return I_G, I_T
end

I_G, I_T = integrate_over_theta(decompo, G)
I_true = GRD.hypersingular_laplace_integral_on_plane_element(el(x̂), el)

@info "Integral using Gaussian quadrature: $I_G, relative error: $(abs(I_G - I_true) / abs(I_true))"
@info "Integral using Trapezoidal quadrature: $I_T, relative error: $(abs(I_T - I_true) / abs(I_true))"
@info "True integral: $I_true"

fig = Figure()
ax = Axis(fig[1, 1], title="G(θ) over θ")
θs = range(0, 2π, length=10000)
G_values = [G(θ) for θ in θs]
lines!(ax, θs, G_values, label="G(θ)")
display(fig)
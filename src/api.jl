# Public API functions

"""
	polar_kernel_fun(K, el::Inti.ReferenceInterpolant, û, x̂)

Given a kernel `K`, a reference element `el`, a function `û` defined on the reference element, 
and a point `x̂` on the reference element, returns a function `F` that computes the complete 
kernel in polar coordinates centered at `x̂`:

```
F(ρ, θ) = K(x, y) * J(ŷ) * ρ * û(ŷ)
```

where:
- `x = el(x̂)` is the physical position of the source point
- `ŷ = x̂ + ρ * (cos(θ), sin(θ))` is the parametric position in polar coordinates
- `y = el(ŷ)` is the physical position
- `J(ŷ)` is the integration measure at `ŷ`

The kernel `K` is called as `K(qx, qy)` where `qx` and `qy` are named tuples with fields 
`coords` and `normal`.

# Arguments
- `K`: The kernel function (or `SplitKernel`)
- `el::Inti.ReferenceInterpolant`: The reference element
- `û`: Function defined on the reference element
- `x̂`: Point on the reference element (singularity location)

# Returns
- `F(ρ, θ)`: A function that evaluates the kernel in polar coordinates
"""
function polar_kernel_fun(K, el::Inti.ReferenceInterpolant, û, x̂)
	x = el(x̂)
	ori = 1
	jac_x = Inti.jacobian(el, x̂)
	nx = Inti._normal(jac_x, ori)
	qx = (coords = x, normal = nx)
	function F(ρ, θ)
		s, c = sincos(θ)
		ŷ = x̂ + ρ * SVector(c, s)
		y = el(ŷ)
		jac_y = Inti.jacobian(el, ŷ)
		ny = Inti._normal(jac_y, ori)
		μ = Inti._integration_measure(jac_y)
		qy = (coords = y, normal = ny)
		return K(qx, qy) * μ * ρ * û(ŷ)
	end
	return F
end

"""
	rho_fun(ref_domain::Inti.ReferenceDomain, x̂)

Given a reference domain `ref_domain` and a point `x̂` in the reference domain, returns the 
function `ρ(θ)` that gives the distance from `x̂` to the boundary of the reference domain in 
the direction `θ`.

# Arguments
- `ref_domain::Inti.ReferenceDomain`: The reference domain (e.g., `ReferenceSquare()`, `ReferenceTriangle()`)
- `x̂`: Point in the reference domain

# Returns
- `ρ(θ)`: Function that computes the distance to the boundary at angle `θ`
"""
function rho_fun(ref_domain, x̂)
	decompo = Inti.polar_decomposition(ref_domain, x̂)
	function ρ(θ)
		if decompo[1][1] ≤ θ < decompo[1][2]
			return decompo[1][3](θ)
		elseif decompo[2][1] ≤ θ < decompo[2][2]
			return decompo[2][3](θ)
		elseif decompo[3][1] ≤ θ < decompo[3][2]
			return decompo[3][3](θ)
		else
			return decompo[4][3](θ)
		end
	end
	return ρ
end

"""
	laurents_coeffs(
		K::Inti.AbstractKernel, 
		el::Inti.ReferenceInterpolant, 
		û, 
		x̂,
		method::AbstractMethod = FullRichardsonExpansion(),
	)

Compute Laurent coefficients `(f₋₂, f₋₁)` for the kernel `K` in polar coordinates centered at `x̂`.

This function creates a [`LaurentExpander`](@ref) and returns a memoized function `ℒ(θ)` that 
computes the Laurent coefficients for any angle `θ`.

# Arguments
- `K::Inti.AbstractKernel`: The kernel to expand
- `el::Inti.ReferenceInterpolant`: The reference element
- `û`: Function defined on the reference element (density function)
- `x̂`: Point on the reference element (singularity location)
- `method::AbstractMethod`: Expansion method to use. Can be:
  - `AnalyticalExpansion()`: Uses closed-form analytical expressions (fastest, limited availability)
  - `AutoDiffExpansion()`: Uses automatic differentiation (fast, requires translation-invariant kernels)
  - `SemiRichardsonExpansion(params)`: Hybrid Richardson extrapolation (moderate speed)
  - `FullRichardsonExpansion(params)`: Full Richardson extrapolation (slowest, always available)

# Returns
- `ℒ(θ)`: A memoized function that returns `(f₋₂, f₋₁)` for a given angle `θ`

# Examples
```julia
# Using default method (FullRichardson)
ℒ = laurents_coeffs(K, el, û, x̂)
f₋₂, f₋₁ = ℒ(0.5)  # Evaluate at θ = 0.5

# Using AutoDiff method
ℒ = laurents_coeffs(K, el, û, x̂, AutoDiffExpansion())

# Using explicit method with custom parameters
params = RichardsonParams(atol=1e-10, rtol=1e-8, maxeval=10)
ℒ = laurents_coeffs(K, el, û, x̂, FullRichardsonExpansion(params))
```
"""
function laurents_coeffs(
	K::Inti.AbstractKernel,
	el::Inti.ReferenceInterpolant,
	û,
	x̂,
	method::AbstractMethod = FullRichardsonExpansion(),
)
	# Create the expander
	expander = LaurentExpander(method, K, el, x̂, û)

	# Return a memoized function
	return Memoization.@memoize θ -> compute_coefficients(expander, θ)
end

"""
	guiggiani_singular_integral(
		K::Inti.AbstractKernel,
		û,
		x̂,
		el::Inti.ReferenceInterpolant,
		quad_rho,
		quad_theta,
		method::AbstractMethod = FullRichardsonExpansion(),
	)

Compute the singular integral of kernel `K` over element `el` using Guiggiani's method.

This function evaluates:
```
∫_{el} K(x, y) * û(ŷ) dS(y)
```
where `x = el(x̂)` is the singular point on the element.

The method works by:
1. Transforming to polar coordinates centered at `x̂`
2. Computing Laurent coefficients `(f₋₂, f₋₁)` to extract the singularity
3. Integrating the regularized integrand: `K - f₋₂/ρ² - f₋₁/ρ`
4. Adding back the analytical integrals of the singular terms

# Arguments
- `K::Inti.AbstractKernel`: The kernel (must have a defined `singularity_order`)
- `û`: Density function defined on the reference element
- `x̂`: Singular point location on the reference element
- `el::Inti.ReferenceInterpolant`: The reference element
- `quad_rho`: Quadrature rule for the radial direction (e.g., `Inti.GaussLegendre(10)`)
- `quad_theta`: Quadrature rule for the angular direction (e.g., `Inti.GaussLegendre(20)`)
- `method::AbstractMethod`: Method for computing Laurent coefficients (see [`laurents_coeffs`](@ref))

# Returns
- The value of the singular integral

# Notes
- The singularity order is automatically detected from `Inti.singularity_order(K)`
- In polar coordinates, the order is adjusted by +1 (due to the Jacobian factor ρ)
- Currently supports singularity orders -1, -2, and -3 in Cartesian coordinates

# Examples
```julia
# Basic usage with default parameters
K = Inti.Laplace(dim=2) |> Inti.HyperSingularKernel()
el = # ... reference element ...
û = ŷ -> 1.0  # constant density
x̂ = SVector(0.5, 0.5)
quad_rho = Inti.GaussLegendre(10)
quad_theta = Inti.GaussLegendre(20)

I = guiggiani_singular_integral(K, û, x̂, el, quad_rho, quad_theta)

# Using automatic differentiation for speed
I = guiggiani_singular_integral(K, û, x̂, el, quad_rho, quad_theta, AutoDiffExpansion())

# With custom Richardson parameters
params = RichardsonParams(atol=1e-12, maxeval=10)
I = guiggiani_singular_integral(K, û, x̂, el, quad_rho, quad_theta, FullRichardsonExpansion(params))
```
"""
function guiggiani_singular_integral(
	K::Inti.AbstractKernel,
	û,
	x̂,
	el::Inti.ReferenceInterpolant,
	quad_rho,
	quad_theta,
	method::AbstractMethod = FullRichardsonExpansion(),
)
	# Determine singularity order
	s = Inti.singularity_order(K)
	if isnothing(s)
		@warn "Kernel does not have a defined singularity_order. Assuming -3."
		s = -3
	end
	# In polar coordinates, the order is adjusted by +1 due to the Jacobian ρ
	sorder_polar = Val(s + 1)

	# Get reference shape for polar decomposition
	ref_shape = Inti.reference_domain(el)

	# Wrap kernel to handle SplitKernel if needed
	SK = K isa SplitKernel ? K : SplitKernel(K)
	Kprod = (qx, qy) -> prod(SK(qx, qy))

	# Create polar kernel function
	K_polar = polar_kernel_fun(Kprod, el, û, x̂)

	# Compute Laurent coefficients
	ℒ = laurents_coeffs(K, el, û, x̂, method)

	# Initialize accumulator
	acc = zero(Inti.return_type(K_polar, Float64, Float64))

	# Integrate over each angular sector
	for (theta_min, theta_max, rho_func) in Inti.polar_decomposition(ref_shape, x̂)
		Δθ = theta_max - theta_min
		I_theta = quad_theta() do (theta_ref,)
			θ = theta_min + theta_ref * Δθ
			ρ_max = rho_func(θ)
			f₋₂, f₋₁ = ℒ(θ)

			# Integrate regularized integrand in radial direction
			I_rho = quad_rho() do (rho_ref,)
				ρ = ρ_max * rho_ref
				# Subtract singular terms to regularize
				return _regularized_integrand(K_polar, ρ, θ, f₋₂, f₋₁, sorder_polar)
			end

			# Add back analytical integrals of singular terms
			return _analytical_singular_contribution(I_rho, ρ_max, f₋₂, f₋₁, sorder_polar)
		end
		I_theta *= Δθ
		acc += I_theta
	end

	return acc
end

# Helper function: compute regularized integrand
function _regularized_integrand(K_polar, ρ, θ, f₋₂, f₋₁, ::Val{-2})
	return K_polar(ρ, θ) - f₋₂ / ρ^2 - f₋₁ / ρ
end

function _regularized_integrand(K_polar, ρ, θ, f₋₂, f₋₁, ::Val{-1})
	return K_polar(ρ, θ) - f₋₁ / ρ
end

function _regularized_integrand(K_polar, ρ, θ, f₋₂, f₋₁, ::Val{0})
	return K_polar(ρ, θ)
end

function _regularized_integrand(K_polar, ρ, θ, f₋₂, f₋₁, ::Val{N}) where {N}
	if N > 0
		return K_polar(ρ, θ)
	else
		throw(ArgumentError("Singularity order $(N) not supported. Must be >= -2."))
	end
end

# Helper function: analytical contribution from singular terms
function _analytical_singular_contribution(I_rho, ρ_max, f₋₂, f₋₁, ::Val{-2})
	return I_rho * ρ_max + f₋₁ * log(ρ_max) - f₋₂ / ρ_max
end

function _analytical_singular_contribution(I_rho, ρ_max, f₋₂, f₋₁, ::Val{-1})
	return I_rho * ρ_max + f₋₁ * log(ρ_max)
end

function _analytical_singular_contribution(I_rho, ρ_max, f₋₂, f₋₁, ::Val{0})
	return I_rho * ρ_max
end

function _analytical_singular_contribution(I_rho, ρ_max, f₋₂, f₋₁, ::Val{N}) where {N}
	if N > 0
		return I_rho * ρ_max
	else
		throw(ArgumentError("Singularity order $(N) not supported. Must be >= -2."))
	end
end

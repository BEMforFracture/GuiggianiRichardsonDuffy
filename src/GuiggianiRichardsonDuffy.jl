"""
GuiggianiRichardsonDuffy

Outils pour l’intégration de noyaux singuliers/hypersinguliers en BEM via l’algorithme de Guiggiani,
avec calcul des coefficients de Laurent par méthodes analytiques, différentiation automatique ou
extrapolation de Richardson, et quadrature en coordonnées de Duffy.

- Fonction principale: `guiggiani_singular_integral`.
- Méthodes d’expansion: `:analytical`, `:auto_diff`, `:semi_richardson`, `:full_richardson`.
- Noyaux fournis: Laplace et Élastostatique (simple/double/adjoint/hypersingulier).

Exemple minimal
```julia
using Inti, StaticArrays
using GuiggianiRichardsonDuffy

# Élément de référence et point singulier
el = Inti.LagrangeSquare((SVector(0.0,0.0,0.0), SVector(1.0,0.0,0.0),
						  SVector(0.0,1.0,0.0), SVector(1.0,1.0,0.0)))
x̂ = SVector(0.3, 0.4)

# Noyau et fonction d’essai sur l’élément de référence
K = GuiggianiRichardsonDuffy.SplitLaplaceHypersingular
û(ξ) = 1.0

I = guiggiani_singular_integral(K, û, x̂, el, 16, 32; expansion = :full_richardson)
```
"""

module GuiggianiRichardsonDuffy

using LinearAlgebra
using StaticArrays
using Inti
using Richardson
using Memoization
using ForwardDiff

include("utils.jl")
include("kernels.jl")
include("geometry_expansion.jl")
include("closed_forms.jl")
include("kernel_expansion.jl")

@info "Loading GuiggianiRichardsonDuffy.jl"

"""
	const EXPANSION_METHODS = [:analytical, :auto_diff, :semi_richardson, :full_richardson]

Available expansion methods for Laurent coefficients of singular kernels.
"""
const EXPANSION_METHODS = [:analytical, :auto_diff, :semi_richardson, :full_richardson]

"""
	const ANALYTICAL_KERNELS = [:LaplaceHypersingular, :ElastostaticHypersingular]

Available kernels with analytical Laurent coefficients.
"""
const ANALYTICAL_KERNELS = [:LaplaceHypersingular, :ElastostaticHypersingular]

"""
	polar_kernel_fun(K::Inti.AbstractKernel, el::Inti.ReferenceInterpolant, û, x̂)

Given a kernel `K`, a reference element `el`, a function `û` defined on the reference element, and a point `x̂` on the reference element, returns a function `F` that computes the complete kernel in polar coordinates centered at `x̂` : F(ρ, θ) = K(x, y) * J(ŷ) * ρ * û(ŷ) where `x = el(x̂)`, `ŷ = x̂ + ρ * (cos(θ), sin(θ))`, `y = el(ŷ)`, and `J(ŷ)` is the integration measure at `ŷ`. `F` will be called as `F(ρ, θ)`. `K` has to be called as `K(qx, qy)` where `qx = (coords = x, normal = nx)` and `qy = (coords = y, normal = ny)` are cartesian points with their normals. 

`K` must return a unique value.
"""
function polar_kernel_fun(K, el::Inti.ReferenceInterpolant, û, x̂)
	x = el(x̂)
	ori = 1
	jac_x = Inti.jacobian(el, x̂)
	nx = Inti._normal(jac_x, ori)
	qx = (coords = x, normal = nx)
	function F(ρ, θ)
		s, c = sincos(θ)
		ŷ = x̂ + ρ * SVector(c, s)
		y = el(ŷ)
		jac_y = Inti.jacobian(el, ŷ)
		ny = Inti._normal(jac_y, ori)
		μ = Inti._integration_measure(jac_y)
		qy = (coords = y, normal = ny)
		return K(qx, qy) * μ * ρ * û(ŷ)
	end
	return F
end

"""
	rho_fun(ref_domain::Inti.ReferenceDomain, x̂)

Given a reference domain `ref_domain` and a point `x̂` in the reference domain, returns the function `ρ(θ)` that gives the distance from `x̂` to the boundary of the reference domain in the direction `θ`. `ρ` will be called as `ρ(θ)`.
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
	laurents_coeffs(K, el::Inti.ReferenceInterpolant, û, x̂; expansion = (method = :full_richardson,), kwargs...)

Given a kernel `K`, a reference element `el`, a function `û` defined on the reference element, and a point `x̂` on the reference element, returns the laurent coefficients `F₋₂` and `F₋₁` for the kernel `K` in polar coordinates centered at `x̂`. The coefficients are computed using the method specified in the `expansion` argument, which can be one of the following:

- `:analytical`: uses analytical expressions for the coefficients (if available). `kernel_kwargs...` are passed to analytical functions.
- `:auto_diff`: uses semi-analytical expressions for the coefficients (if available i.e. when the property of the kernel being translation-invariant holds) based on automatic differentiation used in the `ForwardDiff.jl` package. `kernel_kwargs...` are passed to the kernel `K̂`.
- `:semi_richardson`: uses another semi-analytical method for the coefficients (if available i.e. when the property of the kernel being translation-invariant holds). `richardson_kwargs...` are passed to `Richardson.extrapolate` (see [Richardson.jl](https://github.com/JuliaMath/Richardson.jl)) and `kernel_kwargs...` are passed to the kernel `K̂`.
- `:full_richardson`: uses Richardson extrapolation to compute both coefficients, available by default for any kernel. `richardson_kwargs...` are passed to `Richardson.extrapolate` (see [Richardson.jl](https://github.com/JuliaMath/Richardson.jl)) and `kernel_kwargs...` are passed to the kernel `K̂`.

K has to be called as K(qx, qy, r̂; kernel_kwargs...) where r̂ is the normalized relative position vector, qx = (coords = x, normal = nx) and qy = (coords = y, normal = ny). K(qx, qy, r̂; kernel_kwargs...) is returning the tuple (1/rˢ, K̂(qx, qy, r̂; kernel_kwargs...)) where s is the order of the singularity.

You can also put all the keyword arguments in `kwargs...`, they will be automatically split between kernel and richardson extrapolation arguments, which are in general : first_contract, contract, breaktol, maxeval, atol, rtol, x0, described in the `Richardson.extrapolate` documentation (see [Richardson.jl](https://github.com/JuliaMath/Richardson.jl)).
"""
function laurents_coeffs(
	K, el::Inti.ReferenceInterpolant, û, x̂;
	expansion::Symbol = :full_richardson,
	kernel_kwargs::NamedTuple = NamedTuple(),
	richardson_kwargs::NamedTuple = NamedTuple(),
	name::Symbol = :LaplaceHypersingular,
	kwargs...,
)
	# 1) Répartition auto des kwargs libres (compatibilité et ergonomie)
	auto_kernel, auto_rich = split_kwargs(kwargs)
	kwargs_kernel = (; kernel_kwargs..., auto_kernel...)
	kwargs_rich = (; richardson_kwargs..., auto_rich...)

	# 2) Délégation par méthode
	if expansion == :analytical
		# Ne passer que les kwargs noyau
		return _laurents_coeff_analytical(el, û, x̂; name = name, kwargs_kernel...)
	elseif expansion == :auto_diff
		return _laurents_coeff_auto_diff(K, el, û, x̂; kwargs_kernel...)
	elseif expansion == :semi_richardson
		return _laurents_coeff_semi_richardson(K, el, û, x̂, kwargs_kernel, kwargs_rich)
	elseif expansion == :full_richardson
		return _laurents_coeff_full_richardson(K, el, û, x̂, kwargs_kernel, kwargs_rich)
	else
		error("Unknown expansion type: $(expansion). Available types are: $(EXPANSION_METHODS)")
	end
end

"""
	guiggiani_singular_integral(
		K,
		û,
		x̂,
		el::Inti.ReferenceInterpolant,
		n_rho,
		n_theta;
		sorder::Val{P} = Val(-2),
		expansion::Symbol = :full_richardson,
		kernel_kwargs::NamedTuple = NamedTuple(),
		richardson_kwargs::NamedTuple = NamedTuple(),
		kwargs...,
	) where {P}

Given a kernel `K`, a function `û` defined on the reference element `el`, a point `x̂` on the reference element where the singularity is located, the number of quadrature points in the radial direction `n_rho`, the number of quadrature points in the angular direction `n_theta`, and the order of the singularity `sorder` (which has to be -1, -2 or -3), computes the integral of the kernel over the reference element using the Guiggiani method using expansion of Laurent coefficients specified in the `expansion` argument.

See [`GuiggianiRichardsonDuffy.laurents_coeffs`](@ref) for the available expansion methods and their parameters.
"""
function guiggiani_singular_integral(
	K,
	û,
	x̂,
	el::Inti.ReferenceInterpolant,
	n_rho,
	n_theta;
	sorder::Val{P} = Val(-2),
	expansion::Symbol = :full_richardson,
	kernel_kwargs::NamedTuple = NamedTuple(),
	richardson_kwargs::NamedTuple = NamedTuple(),
	name::Symbol = :LaplaceHypersingular,
	kwargs...,
) where {P}
	s = P - 1
	ref_shape = Inti.reference_domain(el)
	auto_kernel, auto_rich = split_kwargs(kwargs)
	kwargs_kernel = (; kernel_kwargs..., auto_kernel...)
	kwargs_rich = (; richardson_kwargs..., auto_rich...)
	Kprod = (qx, qy) -> prod(K(qx, qy; kwargs_kernel...))
	K_polar = polar_kernel_fun(Kprod, el, û, x̂)
	# integrate
	quad_rho = Inti.GaussLegendre(n_rho)
	quad_theta = Inti.GaussLegendre(n_theta)
	# T = Inti.return_type(K_polar, Float64, Float64)
	acc = zero(K_polar(1.0, 0.0))
	F₋₂, F₋₁ = laurents_coeffs(K, el, û, x̂; expansion = expansion, kernel_kwargs = kwargs_kernel, richardson_kwargs = kwargs_rich, name = name)
	for (theta_min, theta_max, rho_func) in Inti.polar_decomposition(ref_shape, x̂)
		Δθ = theta_max - theta_min
		I_theta = quad_theta() do (theta_ref,)
			θ = theta_min + theta_ref * Δθ
			ρ_max = rho_func(θ)
			I_rho = quad_rho() do (rho_ref,)
				ρ = ρ_max * rho_ref
				if s == -3
					return K_polar(ρ, θ) - F₋₂(θ) / ρ^2 - F₋₁(θ) / ρ
				else
					notimplemented()
				end
			end
			if s == -3
				return I_rho * ρ_max + F₋₁(θ) * log(ρ_max) - F₋₂(θ) / ρ_max
			else
				notimplemented()
			end
		end
		I_theta *= Δθ
		acc += I_theta
	end
	return acc
end

export guiggiani_singular_integral

end # module GuiggianiRichardsonDuffy

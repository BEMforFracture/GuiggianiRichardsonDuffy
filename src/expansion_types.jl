abstract type AbstractMethod end
abstract type AbstractRichardsonParams end

struct RichardsonParams{T <: Real} <: AbstractRichardsonParams
	first_contract::T
	contract::T
	breaktol::T
	atol::T
	rtol::T
	maxeval::Int
end

RichardsonParams(;
	first_contract = 1e-2,
	contract = 0.5,
	breaktol = Inf,
	atol = 0.0,
	rtol = 0.0,
	maxeval = 5,
) = RichardsonParams(
	first_contract,
	contract,
	breaktol,
	atol,
	rtol,
	maxeval,
)

struct AnalyticalExpansion <: AbstractMethod end
struct AutoDiffExpansion <: AbstractMethod end

struct SemiRichardsonExpansion <: AbstractMethod
	richardson_params::RichardsonParams
end

# Default constructor with default parameters
SemiRichardsonExpansion() = SemiRichardsonExpansion(RichardsonParams())

struct FullRichardsonExpansion <: AbstractMethod
	richardson_params::RichardsonParams
end

# Default constructor with default parameters
FullRichardsonExpansion() = FullRichardsonExpansion(RichardsonParams())

struct LaurentExpander{M <: AbstractMethod, Tk, T <: Real, N, Nd, D, Tu, TKpolar, TRho}
	method::M
	kernel::Inti.AbstractKernel{Tk}

	reference_element::Inti.ReferenceInterpolant{D, SVector{N, T}}
	source_point::SVector{Nd, T}

	x::SVector{N, T}
	nx::SVector{N, T}
	Dτ::SMatrix{N, Nd, T}
	D²τ::SArray{Tuple{N, Nd, Nd}, T, 3}
	μ::T

	û::Tu
	
	# Pre-computed closures (avoid recreation)
	K_polar::TKpolar
	rho_max_fun::TRho
end

function LaurentExpander(
	method::M,
	K::Inti.AbstractKernel{Tk},
	el::Inti.ReferenceInterpolant{D, SVector{N, T}},
	x̂::SVector{Nd, T},
	û::Tu,
) where {M <: AbstractMethod, Tk, T <: Real, N, Nd, D, Tu}
	x = el(x̂)
	Dτ = Inti.jacobian(el, x̂)
	ori = 1
	nx = Inti._normal(Dτ, ori)
	D²τ = Inti.hessian(el, x̂)
	μ = Inti._integration_measure(Dτ)
	@assert Nd == Inti.geometric_dimension(el) "source_point dimension must match reference element dimension"
	
	# Pre-compute closures ONCE (major performance optimization)
	SK = K isa SplitKernel ? K : SplitKernel(K)
	Kprod = (qx, qy) -> prod(SK(qx, qy))
	K_polar = polar_kernel_fun(Kprod, el, û, x̂)
	
	ref_domain = Inti.reference_domain(el)
	rho_max_fun = rho_fun(ref_domain, x̂)
	
	return LaurentExpander(
		method,
		K,
		el,
		x̂,
		x,
		nx,
		Dτ,
		D²τ,
		μ,
		û,
		K_polar,
		rho_max_fun,
	)
end

function Base.show(io::IO, expander::LaurentExpander)
	println(io, "LaurentExpander with :")
	println(io, "  - Method: $(typeof(expander.method))")
	println(io, "  - Kernel: $(typeof(expander.kernel))")
	println(io, "  - Source point: $(expander.source_point)")
	println(io, "  - Reference element: $(typeof(expander.reference_element))")
end

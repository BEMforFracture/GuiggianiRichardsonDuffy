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

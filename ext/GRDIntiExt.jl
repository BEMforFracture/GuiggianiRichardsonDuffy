module GRDIntiExt

using Inti: Inti
using GuiggianiRichardsonDuffy: GuiggianiRichardsonDuffy

function adaptive_correction(
	iop::IntegralOperator;
	method::AbstractMethod = FullRichardsonExpansion(),
	maxdist = nothing,
	rtol = nothing,
	atol = nothing,
	threads = true,
	kwargs...,
)
	#TODO: implement adaptive_correction
end

end # module GRDIntiExt

module GuiggianiRichardsonDuffy

using LinearAlgebra
using StaticArrays
using Inti
using Richardson
using Memoization
using ForwardDiff
using SparseArrays

@info "Loading GuiggianiRichardsonDuffy.jl"

include("utils.jl")
include("kernels.jl")
include("geometry_expansion.jl")
include("closed_forms.jl")
include("expansion_types.jl")
include("expansion_methods.jl")
include("helpers.jl")
include("api.jl")
include("inti_integration.jl")

@info "GuiggianiRichardsonDuffy.jl successfully loaded"

"""
	const ANALYTICAL_EXPANSIONS = [:LaplaceHypersingular, :ElastostaticHypersingular]

Available kernels with analytical Laurent coefficients.
"""
const ANALYTICAL_EXPANSIONS = [
	Inti.HyperSingularKernel{T, <:Inti.Laplace{N}} where {T, N},
	Inti.HyperSingularKernel{T, <:Inti.Elastostatic{N}} where {T, N},
]

export AbstractMethod
export RichardsonParams

export FullRichardsonExpansion
export AutoDiffExpansion
export AnalyticalExpansion
export SemiRichardsonExpansion

export adaptive_correction

end # module GuiggianiRichardsonDuffy

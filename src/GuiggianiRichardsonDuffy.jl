module GuiggianiRichardsonDuffy

using LinearAlgebra
using StaticArrays
using Inti
using Richardson
using Memoization
using ForwardDiff

@info "Loading GuiggianiRichardsonDuffy.jl"

include("utils.jl")
include("kernels.jl")
include("geometry_expansion.jl")
include("closed_forms.jl")
include("expansion_types.jl")
include("expansion_methods.jl")
include("helpers.jl")
include("api.jl")

@info "GuiggianiRichardsonDuffy.jl successfully loaded"

"""
	const ANALYTICAL_EXPANSIONS = [:LaplaceHypersingular, :ElastostaticHypersingular]

Available kernels with analytical Laurent coefficients.
"""
const ANALYTICAL_EXPANSIONS = [
	Inti.HyperSingularKernel{T, <:Inti.Laplace{N}} where {T, N},
	Inti.HyperSingularKernel{T, <:Inti.Elastostatic{N}} where {T, N},
]

export RichardsonParams

export FullRichardsonExpansion
export AutoDiffExpansion
export AnalyticalExpansion
export SemiRichardsonExpansion

export guiggiani_singular_integral

end # module GuiggianiRichardsonDuffy

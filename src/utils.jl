#= Functions that are not an obvious place to go =#

function ⊗(u::AbstractVector, v::AbstractVector)
	return u * transpose(v)
end

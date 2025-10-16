# = Kernels and their split versions =#

################################################################################
################################# LAPLACE ######################################
################################################################################

function SplitLaplaceSingleLayer(qx, qy, r̂ = nothing)
	r = qy.coords - qx.coords
	d = norm(r)
	r̂ = isnothing(r̂) ? r / d : r̂
	return 1 / d, 1 / (4π)
end

LaplaceSingleLayer(args...) = prod(SplitLaplaceSingleLayer(args...))

function SplitLaplaceDoubleLayer(qx, qy, r̂ = nothing)
	r = qx.coords - qy.coords
	d = norm(r)
	r̂ = isnothing(r̂) ? r / d : r̂
	ny = qy.normal
	return 1 / d^2, -1 / (4π) * dot(r̂, ny)
end

LaplaceDoubleLayer(args...) = prod(SplitLaplaceDoubleLayer(args...))

function SplitLaplaceAdjointDoubleLayer(qx, qy, r̂ = nothing)
	r = qx.coords - qy.coords
	d = norm(r)
	r̂ = isnothing(r̂) ? r / d : r̂
	nx = qx.normal
	return -1 / d^2, 1 / (4π) * dot(r̂, nx)
end

LaplaceAdjointDoubleLayer(args...) = prod(SplitLaplaceAdjointDoubleLayer(args...))

function SplitLaplaceHypersingular(qx, qy, r̂ = nothing)
	r = qy.coords - qx.coords
	d = norm(r)
	r̂ = isnothing(r̂) ? r / d : r̂
	nx = qx.normal
	ny = qy.normal
	return 1 / d^3, 1 / (4π) * transpose(nx) * ((LinearAlgebra.I - 3 * r̂ * transpose(r̂)) * ny)
end

LaplaceHypersingular(args...) = prod(SplitLaplaceHypersingular(args...))

################################################################################
################################# Elastostatic #################################
################################################################################

function SplitElastostaticSingleLayer(qx, qy, r̂ = nothing; μ, λ)
	ν = λ / (2 * (μ + λ))
	r = qy.coords - qx.coords
	d = norm(r)
	r̂ = isnothing(r̂) ? r / d : r̂
	return 1 / d, 1 / (16π * μ * (1 - ν)) * ((3 - 4 * ν) * LinearAlgebra.I + r̂ * transpose(r̂))
end

ElastostaticSingleLayer(args...; kwargs...) = prod(SplitElastostaticSingleLayer(args...; kwargs...))

function SplitElastostaticDoubleLayer(qx, qy, r̂ = nothing; μ, λ)
	ν = λ / (2 * (μ + λ))
	r = qy.coords - qx.coords
	d = norm(r)
	r̂ = isnothing(r̂) ? r / d : r̂
	ny = qy.normal
	return 1 / d^2, -1 / (8π * (1 - ν)) * (dot(r̂, ny) * ((1 - 2 * ν) * LinearAlgebra.I + 3 * r̂ * transpose(r̂)) + (1 - 2 * ν) * (r̂ * transpose(ny) - ny * transpose(r̂)))
end

ElastostaticDoubleLayer(args...; kwargs...) = prod(SplitElastostaticDoubleLayer(args...; kwargs...))

function SplitElastostaticAdjointDoubleLayer(qx, qy, r̂ = nothing; μ, λ)
	ν = λ / (2 * (μ + λ))
	r = qy.coords - qx.coords
	d = norm(r)
	r̂ = isnothing(r̂) ? r / d : r̂
	nx = qx.normal
	return -1 / d^2, 1 / (8π * (1 - ν)) * (dot(r̂, nx) * ((1 - 2 * ν) * LinearAlgebra.I + 3 * r̂ * transpose(r̂)) + (1 - 2 * ν) * (r̂ * transpose(nx) - nx * transpose(r̂)))
end

ElastostaticAdjointDoubleLayer(args...; kwargs...) = prod(SplitElastostaticAdjointDoubleLayer(args...; kwargs...))

function SplitElastostaticHypersingular(qx, qy, r̂ = nothing; μ, λ)
	ν = λ / (2 * (μ + λ))
	r = qy.coords - qx.coords
	d = norm(r)
	r̂ = isnothing(r̂) ? r / d : r̂
	nx = qx.normal
	ny = qy.normal
	return 1 / d^3,
	μ / (4π * (1 - ν)) * (
		3 * dot(r̂, ny) * (
			(1 - 2ν) * nx * transpose(r̂) + ν * (dot(r̂, nx) * LinearAlgebra.I + r̂ * transpose(nx)) -
			5 * dot(r̂, nx) * r̂ * transpose(r̂)
		) +
		3 * ν * (dot(r̂, nx) * ny * transpose(r̂) + dot(nx, ny) * r̂ * transpose(r̂)) +
		(1 - 2 * ν) * (
			3 * dot(r̂, nx) * r̂ * transpose(ny) +
			dot(nx, ny) * LinearAlgebra.I +
			ny * transpose(nx)
		) - (1 - 4ν) * nx * transpose(ny)
	)
end

ElastostaticHypersingular(args...; kwargs...) = prod(SplitElastostaticHypersingular(args...; kwargs...))

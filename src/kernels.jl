# = Kernels and their split versions =#

struct SplitKernel{K <: Inti.AbstractKernel} <: Inti.AbstractKernel{Tuple}
	kernel::K
end

Inti.singularity_order(SK::SplitKernel) = Inti.singularity_order(SK.kernel)

function (SK::SplitKernel)(target, source, r̂ = nothing; kwargs...)
	K = SK.kernel
	s = Inti.singularity_order(K)

	# Calculer r et d
	r = Inti.coords(target) - Inti.coords(source)
	d = norm(r)

	T = Inti.return_type(K, typeof(target), typeof(source))

	if d == 0
		singular_part = zero(T)
		_, regular_part = _extract_split_parts(K, target, source, r̂, d, s; kwargs...)
	else
		r̂ = r̂ === nothing ? r / d : r̂
		singular_part, regular_part = _extract_split_parts(K, target, source, r̂, d, s; kwargs...)
	end

	return (singular_part, regular_part)
end

function combine_split(singular::Real, regular)
	return singular * regular
end

function Base.prod(t::Tuple{S, K}) where {S <: Real, K <: Union{Real, AbstractArray}}
	return t[1] * t[2]
end

split_kernel(K::Inti.AbstractKernel) = SplitKernel(K)

function merge_split(SK::SplitKernel)
	return SK.kernel
end

################################################################################
################################# LAPLACE ######################################
################################################################################

function _extract_split_parts(K::Inti.SingleLayerKernel{T, <:Inti.Laplace{N}},
	target, source, r̂, d, s; kwargs...) where {T, N}
	if N == 3
		K̂ = 1 / (4π)
		return (1 / d, K̂)
	elseif N == 2
		K̂ = -1 / (2π)
		return (log(d), K̂)
	else 
		notimplemented()
	end
end

function _extract_split_parts(K::Inti.DoubleLayerKernel{T, <:Inti.Laplace{N}},
	target, source, r̂, d, s; kwargs...) where {T, N}
	ny = Inti.normal(source)

	if N == 3
		K̂ = -1 / (4π) * dot(r̂, ny)
		return (1 / d^2, K̂)
	elseif N == 2
		K̂ = -1 / (2π) * dot(r̂, ny)
		return (1 / d, K̂)
	else 
		notimplemented()
	end
end

function _extract_split_parts(K::Inti.AdjointDoubleLayerKernel{T, <:Inti.Laplace{N}},
	target, source, r̂, d, s; kwargs...) where {T, N}
	nx = Inti.normal(target)

	if N == 3
		K̂ = 1 / (4π) * dot(r̂, nx)
		return (1 / d^2, K̂)
	elseif N == 2
		K̂ = 1 / (2π) * dot(r̂, nx)
		return (1 / d, K̂)
	else 
		notimplemented()
	end
end

function _extract_split_parts(K::Inti.HyperSingularKernel{T, <:Inti.Laplace{N}},
	target, source, r̂, d, s; kwargs...) where {T, N}
	nx = Inti.normal(target)
	ny = Inti.normal(source)

	if N == 3
		K̂ = 1 / (4π) * transpose(nx) * ((I - 3 * r̂ * transpose(r̂)) * ny)
		return (1 / d^3, K̂)
	elseif N == 2
		K̂ = 1 / (2π) * transpose(nx) * ((I - 2 * r̂ * transpose(r̂)) * ny)
		return (1 / d^2, K̂)
	else 
		notimplemented()
	end
end

################################################################################
################################# Elastostatic #################################
################################################################################

function _extract_split_parts(K::Inti.SingleLayerKernel{T, <:Inti.Elastostatic{N}},
	target, source, r̂, d, s; kwargs...) where {T, N}
	μ, λ = K.op.μ, K.op.λ
	ν = λ / (2 * (μ + λ))

	if N == 3
		K̂ = 1 / (8π * μ * (1 - ν)) * ((3 - 4 * ν) * LinearAlgebra.I + r̂ * transpose(r̂))
		return (1 / d, K̂)
	else
		notimplemented()
	end
end

function _extract_split_parts(K::Inti.DoubleLayerKernel{T, <:Inti.Elastostatic{N}},
	target, source, r̂, d, s; kwargs...) where {T, N}
	μ, λ = K.op.μ, K.op.λ
	ν = λ / (2 * (μ + λ))
	ny = Inti.normal(source)

	if N == 3
		K̂ = -1 / (4π * (1 - ν)) * (dot(r̂, ny) * ((1 - 2ν) * LinearAlgebra.I + 3 * r̂ * transpose(r̂)) + (1 - 2ν) * (r̂ * transpose(ny) - ny * transpose(r̂)))
		return (1 / d^2, K̂)
	else
		notimplemented()
	end
end

function _extract_split_parts(K::Inti.AdjointDoubleLayerKernel{T, <:Inti.Elastostatic{N}},
	target, source, r̂, d, s; kwargs...) where {T, N}
	μ, λ = K.op.μ, K.op.λ
	ν = λ / (2 * (μ + λ))
	nx = Inti.normal(target)

	if N == 3
		K̂ = 1 / (4π * (1 - ν)) * (dot(r̂, nx) * ((1 - 2ν) * LinearAlgebra.I + 3 * r̂ * transpose(r̂)) + (1 - 2ν) * (r̂ * transpose(nx) - nx * transpose(r̂)))
		return (-1 / d^2, K̂)
	else
		notimplemented()
	end
end

function _extract_split_parts(K::Inti.HyperSingularKernel{T, <:Inti.Elastostatic{N}},
	target, source, r̂, d, s; kwargs...) where {T, N}
	μ, λ = K.op.μ, K.op.λ
	ν = λ / (2 * (μ + λ))
	nx = Inti.normal(target)
	ny = Inti.normal(source)

	if N == 3
		K̂ =
			μ / (4π * (1 - ν)) * (
				3 * dot(r̂, ny) * (
					(1 - 2ν) * nx * transpose(r̂) + ν * (dot(r̂, nx) * I + r̂ * transpose(nx)) -
					5 * dot(r̂, nx) * r̂ * transpose(r̂)
				) +
				3 * ν * (dot(r̂, nx) * ny * transpose(r̂) + dot(nx, ny) * r̂ * transpose(r̂)) +
				(1 - 2 * ν) * (
					3 * dot(r̂, nx) * r̂ * transpose(ny) +
					dot(nx, ny) * I +
					ny * transpose(nx)
				) - (1 - 4ν) * nx * transpose(ny)
			)
			return (1 / d^3, K̂)
	else
		notimplemented()
	end
end

################################################################################
################################ GENERIC ERROR #################################
################################################################################

function _extract_split_parts(K::Inti.AbstractKernel, target, source, r̂, d, s; kwargs...)
	throw(ErrorException("No split kernel implementation for kernel type $(typeof(K))"))
end

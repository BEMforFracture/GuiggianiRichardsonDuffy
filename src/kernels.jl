# = Kernels and their split versions =#

struct SplitKernel{K <: Inti.AbstractKernel} <: Inti.AbstractKernel{Tuple}
	kernel::K
end

Inti.singularity_order(SK::SplitKernel) = Inti.singularity_order(SK.kernel)

function (SK::SplitKernel)(target, source, r̂ = nothing; kwargs...)
	K = SK.kernel
	s = Inti.singularity_order(K)
	N = Inti.ambient_dimension(Inti.operator(K))

	# Calculer r et d
	r = Inti.coords(target) - Inti.coords(source)
	d = norm(r)

	if d ≤ Inti.SAME_POINT_TOLERANCE
		# Cas singulier - traitement spécial
		if !isnothing(r̂)
			return _split_kernel_at_singularity(K, target, source, r̂; kwargs...)
		else
			error("Direction r̂ required at singular point")
		end
	end

	# Cas régulier : extraire la partie singulière et régulière
	singular_part, regular_part = _extract_split_parts(K, target, source, r, d, s; kwargs...)

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
	target, source, r, d, s; kwargs...) where {T, N}
	# La partie singulière est 1/d (ordre -1)
	singular_part = 1 / d
	# La partie régulière K̂ est tout le reste
	K̂ = 1 / (4π) * LinearAlgebra.I
	return (singular_part, K̂)
end

function _extract_split_parts(K::Inti.DoubleLayerKernel{T, <:Inti.Laplace{N}},
	target, source, r, d, s; kwargs...) where {T, N}
	nx = Inti.normal(target)
	ny = Inti.normal(source)
	r̂ = r / d

	# La partie singulière est 1/d^2 (ordre -2)
	singular_part = 1 / d^2
	# La partie régulière K̂ est tout le reste
	if N == 2
		K̂ = -1 / (2π) * dot(r̂, ny)
	elseif N == 3
		K̂ = -1 / (4π) * dot(r̂, ny)
	end

	return (singular_part, K̂)
end

function _extract_split_parts(K::Inti.AdjointDoubleLayerKernel{T, <:Inti.Laplace{N}},
	target, source, r, d, s; kwargs...) where {T, N}
	nx = Inti.normal(target)
	ny = Inti.normal(source)
	r̂ = r / d

	# La partie singulière est -1/d^2 (ordre -2)
	singular_part = -1 / d^2
	# La partie régulière K̂ est tout le reste
	if N == 2
		K̂ = 1 / (2π) * dot(r̂, nx)
	elseif N == 3
		K̂ = 1 / (4π) * dot(r̂, nx)
	end

	return (singular_part, K̂)
end

function _extract_split_parts(K::Inti.HyperSingularKernel{T, <:Inti.Laplace{N}},
	target, source, r, d, s; kwargs...) where {T, N}
	nx = Inti.normal(target)
	ny = Inti.normal(source)

	# La partie singulière est 1/d^N (ordre -N pour hypersingular)
	singular_part = 1 / d^N

	# La partie régulière K̂ est tout le reste
	if N == 2
		K̂ = 1 / (2π) * transpose(nx) * ((I - 2 * r * transpose(r) / d^2) * ny)
	elseif N == 3
		K̂ = 1 / (4π) * transpose(nx) * ((I - 3 * r * transpose(r) / d^2) * ny)
	end

	return (singular_part, K̂)
end

function _split_kernel_at_singularity(K::Inti.SingleLayerKernel{T, <:Inti.Laplace{N}},
	target, source, r̂; kwargs...) where {T, N}
	# Pour la limite quand ρ → 0
	# La partie singulière diverge, la partie régulière est constante
	return (zero(T), 1 / (4π) * LinearAlgebra.I)
end

function _split_kernel_at_singularity(K::Inti.DoubleLayerKernel{T, <:Inti.Laplace{N}},
	target, source, r̂; kwargs...) where {T, N}
	# Pour la limite quand ρ → 0
	# La partie singulière diverge, la partie régulière dépend de la direction
	ny = Inti.normal(source)

	if N == 2
		K̂ = -1 / (2π) * dot(r̂, ny)
	elseif N == 3
		K̂ = -1 / (4π) * dot(r̂, ny)
	end

	return (zero(T), K̂)
end

function _split_kernel_at_singularity(K::Inti.AdjointDoubleLayerKernel{T, <:Inti.Laplace{N}},
	target, source, r̂; kwargs...) where {T, N}
	# Pour la limite quand ρ → 0
	# La partie singulière diverge, la partie régulière dépend de la direction
	nx = Inti.normal(target)

	if N == 2
		K̂ = 1 / (2π) * dot(r̂, nx)
	elseif N == 3
		K̂ = 1 / (4π) * dot(r̂, nx)
	end

	return (zero(T), K̂)
end

# Cas au point singulier (quand target ≈ source)
function _split_kernel_at_singularity(K::Inti.HyperSingularKernel{T, <:Inti.Laplace{N}},
	target, source, r̂; kwargs...) where {T, N}
	# Pour la limite quand ρ → 0
	# La partie singulière diverge, la partie régulière dépend de la direction r̂
	nx = Inti.normal(target)
	ny = Inti.normal(source)

	if N == 2
		K̂ = 1 / (2π) * transpose(nx) * ((I - 2 * r̂ * transpose(r̂)) * ny)
	elseif N == 3
		K̂ = 1 / (4π) * transpose(nx) * ((I - 3 * r̂ * transpose(r̂)) * ny)
	end

	return (zero(T), K̂)
end

################################################################################
################################# Elastostatic #################################
################################################################################

function _extract_split_parts(K::Inti.SingleLayerKernel{T, <:Inti.Elastostatic{N}},
	target, source, r, d, s; kwargs...) where {T, N}
	μ, λ = K.op.μ, K.op.λ
	ν = λ / (2 * (μ + λ))
	r̂ = r / d

	# La partie singulière
	if N == 2
		singular_part = 1 / d
		# Partie régulière
		K̂ = 1 / (4π * μ * (1 - ν)) * ((3 - 4 * ν) * LinearAlgebra.I + r̂ * transpose(r̂))
	elseif N == 3
		singular_part = 1 / d
		# Partie régulière
		K̂ = 1 / (8π * μ * (1 - ν)) * ((3 - 4 * ν) * LinearAlgebra.I + r̂ * transpose(r̂))
	end

	return (singular_part, K̂)
end

function _extract_split_parts(K::Inti.DoubleLayerKernel{T, <:Inti.Elastostatic{N}},
	target, source, r, d, s; kwargs...) where {T, N}
	μ, λ = K.op.μ, K.op.λ
	ν = λ / (2 * (μ + λ))
	nx = Inti.normal(target)
	ny = Inti.normal(source)
	r̂ = r / d

	# La partie singulière
	if N == 2
		singular_part = 1 / d^2
		# Partie régulière
		K̂ = -1 / (2π * (1 - ν)) * (dot(r̂, ny) * ((1 - 2ν) * LinearAlgebra.I + 2 * r̂ * transpose(r̂)) + (1 - 2ν) * (r̂ * transpose(ny) - ny * transpose(r̂)))
	elseif N == 3
		singular_part = 1 / d^2
		# Partie régulière
		K̂ = -1 / (4π * (1 - ν)) * (dot(r̂, ny) * ((1 - 2ν) * LinearAlgebra.I + 3 * r̂ * transpose(r̂)) + (1 - 2ν) * (r̂ * transpose(ny) - ny * transpose(r̂)))
	end

	return (singular_part, K̂)
end

function _extract_split_parts(K::Inti.AdjointDoubleLayerKernel{T, <:Inti.Elastostatic{N}},
	target, source, r, d, s; kwargs...) where {T, N}
	μ, λ = K.op.μ, K.op.λ
	ν = λ / (2 * (μ + λ))
	nx = Inti.normal(target)
	ny = Inti.normal(source)
	r̂ = r / d

	# La partie singulière
	if N == 2
		singular_part = -1 / d^2
		# Partie régulière
		K̂ = 1 / (2π * (1 - ν)) * (dot(r̂, nx) * ((1 - 2ν) * LinearAlgebra.I + 2 * r̂ * transpose(r̂)) + (1 - 2ν) * (r̂ * transpose(nx) - nx * transpose(r̂)))
	elseif N == 3
		singular_part = -1 / d^2
		# Partie régulière
		K̂ = 1 / (4π * (1 - ν)) * (dot(r̂, nx) * ((1 - 2ν) * LinearAlgebra.I + 3 * r̂ * transpose(r̂)) + (1 - 2ν) * (r̂ * transpose(nx) - nx * transpose(r̂)))
	end

	return (singular_part, K̂)
end

function _extract_split_parts(K::Inti.HyperSingularKernel{T, <:Inti.Elastostatic{N}},
	target, source, r, d, s; kwargs...) where {T, N}
	μ, λ = K.op.μ, K.op.λ
	ν = λ / (2 * (μ + λ))
	nx = Inti.normal(target)
	ny = Inti.normal(source)
	RRT = r * transpose(r)
	drdn = dot(r, ny) / d

	# La partie singulière
	if N == 2
		singular_part = 1 / d^2
		# Partie régulière
		K̂ =
			μ / (2π * (1 - ν)) * (
				2 * drdn / d * (
					(1 - 2ν) * nx * transpose(r) + ν * (dot(r, nx) * I + r * transpose(nx)) -
					4 * dot(r, nx) * RRT / d^2
				) +
				2 * ν / d^2 * (dot(r, nx) * ny * transpose(r) + dot(nx, ny) * RRT) +
				(1 - 2 * ν) * (
					2 / d^2 * dot(r, nx) * r * transpose(ny) +
					dot(nx, ny) * I +
					ny * transpose(nx)
				) - (1 - 4ν) * nx * transpose(ny)
			)
	elseif N == 3
		singular_part = 1 / d^3
		K̂ =
			μ / (4π * (1 - ν)) * (
				3 * drdn / d * (
					(1 - 2ν) * nx * transpose(r) + ν * (dot(r, nx) * I + r * transpose(nx)) -
					5 * dot(r, nx) * RRT / d^2
				) +
				3 * ν / d^2 * (dot(r, nx) * ny * transpose(r) + dot(nx, ny) * RRT) +
				(1 - 2 * ν) * (
					3 / d^2 * dot(r, nx) * r * transpose(ny) +
					dot(nx, ny) * I +
					ny * transpose(nx)
				) - (1 - 4ν) * nx * transpose(ny)
			)
	end

	return (singular_part, K̂)
end

function _split_kernel_at_singularity(K::Inti.SingleLayerKernel{T, <:Inti.Elastostatic{N}},
	target, source, r̂; kwargs...) where {T, N}
	μ, λ = K.op.μ, K.op.λ
	ν = λ / (2 * (μ + λ))

	# Pour la limite quand ρ → 0
	# La partie singulière diverge, la partie régulière est constante
	if N == 2
		return (zero(T), 1 / (4π * μ * (1 - ν)) * ((3 - 4 * ν) * LinearAlgebra.I + r̂ * transpose(r̂)))
	elseif N == 3
		return (zero(T), 1 / (8π * μ * (1 - ν)) * ((3 - 4 * ν) * LinearAlgebra.I + r̂ * transpose(r̂)))
	end
end

function _split_kernel_at_singularity(K::Inti.DoubleLayerKernel{T, <:Inti.Elastostatic{N}},
	target, source, r̂; kwargs...) where {T, N}
	μ, λ = K.op.μ, K.op.λ
	ν = λ / (2 * (μ + λ))
	ny = Inti.normal(source)

	# Pour la limite quand ρ → 0
	# La partie singulière diverge, la partie régulière dépend de la direction
	if N == 2
		K̂ = -1 / (2π * (1 - ν)) * (dot(r̂, ny) * ((1 - 2ν) * LinearAlgebra.I + 2 * r̂ * transpose(r̂)) + (1 - 2ν) * (r̂ * transpose(ny) - ny * transpose(r̂)))
	elseif N == 3
		K̂ = -1 / (4π * (1 - ν)) * (dot(r̂, ny) * ((1 - 2ν) * LinearAlgebra.I + 3 * r̂ * transpose(r̂)) + (1 - 2ν) * (r̂ * transpose(ny) - ny * transpose(r̂)))
	end

	return (zero(T), K̂)
end

function _split_kernel_at_singularity(K::Inti.AdjointDoubleLayerKernel{T, <:Inti.Elastostatic{N}},
	target, source, r̂; kwargs...) where {T, N}
	μ, λ = K.op.μ, K.op.λ
	ν = λ / (2 * (μ + λ))
	nx = Inti.normal(target)

	# Pour la limite quand ρ → 0
	# La partie singulière diverge, la partie régulière dépend de la direction
	if N == 2
		K̂ = 1 / (2π * (1 - ν)) * (dot(r̂, nx) * ((1 - 2ν) * LinearAlgebra.I + 2 * r̂ * transpose(r̂)) + (1 - 2ν) * (r̂ * transpose(nx) - nx * transpose(r̂)))
	elseif N == 3
		K̂ = 1 / (4π * (1 - ν)) * (dot(r̂, nx) * ((1 - 2ν) * LinearAlgebra.I + 3 * r̂ * transpose(r̂)) + (1 - 2ν) * (r̂ * transpose(nx) - nx * transpose(r̂)))
	end

	return (zero(T), K̂)
end

function _split_kernel_at_singularity(K::Inti.HyperSingularKernel{T, <:Inti.Elastostatic{N}},
	target, source, r̂; kwargs...) where {T, N}
	μ, λ = K.op.μ, K.op.λ
	ν = λ / (2 * (μ + λ))
	nx = Inti.normal(target)
	ny = Inti.normal(source)

	# Pour la limite quand ρ → 0
	# La partie singulière diverge, la partie régulière dépend de la direction
	if N == 2
		K̂ =
			μ / (2π * (1 - ν)) * (
				2 * dot(r̂, ny) * (
					(1 - 2ν) * nx * transpose(r̂) + ν * (dot(r̂, nx) * LinearAlgebra.I + r̂ * transpose(nx)) -
					4 * dot(r̂, nx) * r̂ * transpose(r̂)
				) +
				2 * ν * (dot(r̂, nx) * ny * transpose(r̂) + dot(nx, ny) * r̂ * transpose(r̂)) +
				(1 - 2 * ν) * (
					2 * dot(r̂, nx) * r̂ * transpose(ny) +
					dot(nx, ny) * LinearAlgebra.I +
					ny * transpose(nx)
				) - (1 - 4ν) * nx * transpose(ny)
			)
	elseif N == 3
		K̂ =
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
	return (zero(T), K̂)
end

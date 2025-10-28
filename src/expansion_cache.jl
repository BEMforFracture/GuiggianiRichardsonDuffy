struct LaurentCache{T}
	n_theta::Int
	coeffs_minus2::Vector{T}
	coeffs_minus1::Vector{T}
	computed::BitVector
end

LaurentCache{T}(n::Int) where T = LaurentCache{T}(
	n,
	Vector{T}(undef, n),
	Vector{T}(undef, n),
	falses(n),
)

@inline function get_or_compute!(
	cache::LaurentCache,
	expander::LaurentExpander,
	idx::Int,
	θ::Real,
)
	if !cache.computed[idx]
		f₋₂, f₋₁ = compute_coefficients(expander, θ)  # Dispatch sur M
		cache.coeffs_minus2[idx] = f₋₂
		cache.coeffs_minus1[idx] = f₋₁
		cache.computed[idx] = true
	end
	return cache.coeffs_minus2[idx], cache.coeffs_minus1[idx]
end

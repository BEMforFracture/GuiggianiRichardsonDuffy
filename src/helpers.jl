# Helper function: compute regularized integrand
function _regularized_integrand(K_polar, ρ, θ, f₋₂, f₋₁, ::Val{-2})
	return K_polar(ρ, θ) - f₋₂ / ρ^2 - f₋₁ / ρ
end

function _regularized_integrand(K_polar, ρ, θ, f₋₂, f₋₁, ::Val{-1})
	return K_polar(ρ, θ) - f₋₁ / ρ
end

function _regularized_integrand(K_polar, ρ, θ, f₋₂, f₋₁, ::Val{0})
	return K_polar(ρ, θ)
end

function _regularized_integrand(K_polar, ρ, θ, f₋₂, f₋₁, ::Val{N}) where {N}
	if N > 0
		return K_polar(ρ, θ)
	else
		throw(ArgumentError("Singularity order $(N) not supported. Must be >= -2."))
	end
end

# Helper function: analytical contribution from singular terms
function _analytical_singular_contribution(I_rho, ρ_max, f₋₂, f₋₁, ::Val{-2})
	return I_rho * ρ_max + f₋₁ * log(ρ_max) - f₋₂ / ρ_max
end

function _analytical_singular_contribution(I_rho, ρ_max, f₋₂, f₋₁, ::Val{-1})
	return I_rho * ρ_max + f₋₁ * log(ρ_max)
end

function _analytical_singular_contribution(I_rho, ρ_max, f₋₂, f₋₁, ::Val{0})
	return I_rho * ρ_max
end

function _analytical_singular_contribution(I_rho, ρ_max, f₋₂, f₋₁, ::Val{N}) where {N}
	if N > 0
		return I_rho * ρ_max
	else
		throw(ArgumentError("Singularity order $(N) not supported. Must be >= -2."))
	end
end

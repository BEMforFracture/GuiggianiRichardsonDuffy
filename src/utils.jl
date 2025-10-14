function custom_contraction(T::SArray{Tuple{3, 2, 2}, Float64, 3, 12}, M::SMatrix{2, 2, Float64, 4})
	res = MVector{3, Float64}(0.0, 0.0, 0.0)
	for i in 1:3
		res[i] = [T[i, j, k] * M[j, k] for j in 1:2, k in 1:2] |> sum
	end
	return res |> SVector
end

function notimplemented()
	throw(ErrorException("This function is not yet implemented."))
end

function hooke_tensor_iso(i, j, k, ℓ; λ, μ)
	return λ * (i == j) * (k == ℓ) + μ * ((i == k) * (j == ℓ) + (i == ℓ) * (j == k))
end

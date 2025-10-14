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

# Utilitaire pour séparer les kwargs en deux NamedTuple
split_kwargs(kwargs; rich_keys = (:first_contract, :contract, :breaktol, :maxeval, :atol, :rtol, :x0)) = begin
	kernel_pairs = Pair{Symbol, Any}[]
	rich_pairs   = Pair{Symbol, Any}[]
	for (k, v) in kwargs
		if k in rich_keys
			push!(rich_pairs, k => v)
		else
			push!(kernel_pairs, k => v)
		end
	end
	kernel_nt = isempty(kernel_pairs) ? NamedTuple() : NamedTuple{Tuple(first.(kernel_pairs))}(Tuple(last.(kernel_pairs)))
	rich_nt   = isempty(rich_pairs) ? NamedTuple() : NamedTuple{Tuple(first.(rich_pairs))}(Tuple(last.(rich_pairs)))
	return kernel_nt, rich_nt
end

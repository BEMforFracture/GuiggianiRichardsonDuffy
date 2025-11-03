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

function is_plane(el::Inti.LagrangeElement)
	tol = eps()
	nodes = el.vals
	npts = length(nodes)
	npts ≤ 3 && return true  # toujours plan avec 3 points ou moins
	P1, P2, P3 = nodes[1], nodes[2], nodes[3]
	n = cross(P2 - P1, P3 - P1)
	# Vérifie la coplanarité de tous les autres points
	for i in 4:npts
		if abs(dot(n, nodes[i] - P1)) > tol
			return false
		end
	end
	return true
end

macro suppress_output(expr)
	quote
		redirect_stdout(devnull) do
			redirect_stderr(devnull) do
				$(esc(expr))
			end
		end
	end
end

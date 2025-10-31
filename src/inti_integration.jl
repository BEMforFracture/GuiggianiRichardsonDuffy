function adaptive_correction(
	iop::Inti.IntegralOperator;
	method::AbstractMethod = FullRichardsonExpansion(),
	maxdist = nothing,
	rtol = nothing,
	atol = nothing,
	threads = true,
	kwargs...,
)
	# check if we need to compute a tolerance and/or a maxdist
	hastol = ((rtol !== nothing) || (atol !== nothing))
	if isnothing(maxdist) || !hastol
		maxdist_, rtol_, atol_ = Inti.local_correction_dist_and_tol(iop)
	end
	# normalize inputs
	maxdist = isnothing(maxdist) ? maxdist_ : maxdist
	rtol = isnothing(rtol) ? (hastol ? 0.0 : rtol_) : rtol
	atol = isnothing(atol) ? (hastol ? 0.0 : atol_) : atol
	# go on and compute the correction
	msh = Inti.mesh(Inti.source(iop))
	quads_dict = Dict()
	for E in Inti.element_types(msh)
		ref_domain = Inti.reference_domain(E)
		quads = (
			nearfield_quad = Inti.adaptive_quadrature(ref_domain; rtol, atol, kwargs...),
			radial_quad = Inti.adaptive_quadrature(Inti.ReferenceLine(); rtol, atol, kwargs...),
			angular_quad = Inti.adaptive_quadrature(Inti.ReferenceLine(); rtol, atol, kwargs...),
		)
		quads_dict[E] = quads
	end
	return adaptive_correction(iop, maxdist, quads_dict, threads, method)
end

function adaptive_correction(iop, maxdist, quads_dict::Dict, threads = true, method = FullRichardsonExpansion())
	# unpack type-unstable fields in iop, allocate output, and dispatch
	X, Y, K = Inti.target(iop), Inti.source(iop), Inti.kernel(iop)
	dict_near = Inti.near_interaction_list([Inti.coords(x) for x in X], Inti.mesh(Y); tol = maxdist)
	T = Inti.eltype(iop)
	msh = Inti.mesh(Y)
	# use the singularity order of the kernel and the geometric dimension to compute the
	# singularity order of the kernel in polar/spherical coordinates
	geo_dim = Inti.geometric_dimension(msh)
	p = Inti.singularity_order(K) # K(x,y) ~ |x-y|^{-p} as y -> 0
	sing_order = if isnothing(p)
		@warn "missing method `singularity_order` for kernel. Assuming finite part integral."
		-2
	else
		p + (geo_dim - 1) # in polar coordinates you muliply by r^{geo_dim-1}
	end
	# allocate output in a sparse matrix style
	correction = (I = Int[], J = Int[], V = T[])
	# loop over element types in the source mesh, unpack, and dispatch to type-stable
	# function
	for E in Inti.element_types(msh)
		nearlist = dict_near[E]
		els = Inti.elements(msh, E)
		ori = Inti.orientation(msh, E)
		# append the regular quadrature rule to the list of quads for the element type E
		# radial singularity order
		quads = merge(quads_dict[E], (regular_quad = Inti.quadrature_rule(Y, E),))
		L = Inti.lagrange_basis(quads.regular_quad)
		_adaptive_correction_etype!(
			correction,
			els,
			ori,
			quads,
			L,
			nearlist,
			X,
			Y,
			K,
			method,
			maxdist,
			threads,
		)
	end
	m, n = size(iop)
	return sparse(correction.I, correction.J, correction.V, m, n)
end

@noinline function _adaptive_correction_etype!(
	correction,
	el_iter,
	orientation,
	quads,
	L,
	nearlist,
	X,
	Y,
	K,
	method,
	nearfield_distance,
	threads,
)
	E = Inti.eltype(el_iter)
	Xqnodes = collect(X)
	Yqnodes = collect(Y)
	# reference quadrature nodes and weights
	x̂ = Inti.qcoords(quads.regular_quad) |> collect
	el2qtags = Inti.etype2qtags(Y, E)
	nel = length(el_iter)
	lck = Threads.SpinLock()
	# lck = ReentrantLock()
	Inti.@maybe_threads threads for n in 1:nel
		el = el_iter[n]
		ori = orientation[n]
		jglob = view(el2qtags, :, n)
		# inear = union(nearlist[n], jglob) # make sure to include nearfield nodes AND the element nodes
		inear = nearlist[n]
		for i in inear
			xnode = Xqnodes[i]
			# closest quadrature node
			dmin, j = findmin(
				n -> norm(Inti.coords(xnode) - Inti.coords(Yqnodes[jglob[n]])),
				1:length(jglob),
			)
			x̂nearest = x̂[j]
			dmin > nearfield_distance && continue
			# If singular, use Guiggiani's method. Otherwise use an oversampled quadrature
			if iszero(dmin)
				W = guiggiani_singular_integral(
					K,
					L,
					x̂nearest,
					el,
					ori,
					quads.radial_quad,
					quads.angular_quad,
					method,
				)
			else
				integrand = (ŷ) -> begin
					y = el(ŷ)
					jac = Inti.jacobian(el, ŷ)
					ν = Inti._normal(jac, ori)
					τ′ = Inti._integration_measure(jac)
					M = K(xnode, (coords = y, normal = ν))
					v = L(ŷ)
					map(v -> M * v, v) * τ′
				end
				W = quads.nearfield_quad(integrand)
			end
			@lock lck for (k, j) in enumerate(jglob)
				qx, qy = Xqnodes[i], Yqnodes[j]
				push!(correction.I, i)
				push!(correction.J, j)
				push!(correction.V, W[k] - K(qx, qy) * Inti.weight(qy))
			end
		end
	end
	return correction
end

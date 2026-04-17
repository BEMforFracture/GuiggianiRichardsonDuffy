using GLMakie
using StaticArrays
using Inti

# ------------------------------------------------------------
# Jacobienne de la transformation Duffy -> polaire:
# on trace les 4 composantes (drho/da, drho/db, dtheta/da, dtheta/db)
# Entree utilisateur: xhat (point source dans le carre de reference)
# ------------------------------------------------------------

# Input utilisateur
quad1D = Inti.GaussLegendre(2)
x̂ = quad1D.nodes[1][1]
xhat = SVector(0.1, 0.1)
# xhat = SVector(0.5, 0.5)
a_eval = 1.0

function return_vertices(tau::Int)
	if tau == 1
		return SVector(1.0, 0.0), SVector(1.0, 1.0)
	elseif tau == 2
		return SVector(1.0, 1.0), SVector(0.0, 1.0)
	elseif tau == 3
		return SVector(0.0, 1.0), SVector(0.0, 0.0)
	elseif tau == 4
		return SVector(0.0, 0.0), SVector(1.0, 0.0)
	else
		error("tau doit etre dans 1:4")
	end
end

function duffy_sectors()
	xiI, xiII = return_vertices(1)
	_, xiIII = return_vertices(2)
	_, xiIV = return_vertices(3)
	return (
		(xiI, xiII, 1),
		(xiII, xiIII, 2),
		(xiIII, xiIV, 3),
		(xiIV, xiI, 4),
	)
end

function bstar_on_sector(xiI::SVector{2,Float64}, xiII::SVector{2,Float64}, xhat::SVector{2,Float64})
	delta = xiII - xiI
	alpha = abs(delta[1]) > abs(delta[2]) ? 1 : 2
	den = xiI[alpha] - xiII[alpha]
	abs(den) < eps(Float64) && return NaN
	return (xiI[alpha] - xhat[alpha]) / den
end

function jacobian_components(xiI::SVector{2,Float64}, xiII::SVector{2,Float64}, xhat::SVector{2,Float64}, a::Float64, b::Float64)
	delta = xiII - xiI
	c = xiI - xhat + b * delta
	cn = norm(c)
	den = max(cn, eps(Float64))

	drho_da = cn
	drho_db = a * (c ⋅ delta) / den
	dtheta_da = 0.0
	dtheta_db = (c[1] * delta[2] - c[2] * delta[1]) / (den^2)

	return drho_da, drho_db, dtheta_da, dtheta_db
end

function compute_component_curves(
	xiI::SVector{2,Float64},
	xiII::SVector{2,Float64},
	xhat::SVector{2,Float64},
	a_eval::Float64,
	bs::Vector{Float64},
)
	drho_da_vals = Float64[]
	drho_db_vals = Float64[]
	dtheta_da_vals = Float64[]
	dtheta_db_vals = Float64[]

	for b in bs
		drho_da, drho_db, dtheta_da, dtheta_db = jacobian_components(xiI, xiII, xhat, a_eval, b)
		push!(drho_da_vals, drho_da)
		push!(drho_db_vals, drho_db)
		push!(dtheta_da_vals, dtheta_da)
		push!(dtheta_db_vals, dtheta_db)
	end

	return drho_da_vals, drho_db_vals, dtheta_da_vals, dtheta_db_vals
end

function plot_duffy_jacobian(xhat::SVector{2,Float64}; a_eval::Float64 = 1.0, nsamples::Int = 1000)
	bs = collect(range(0.0, 1.0, length = nsamples))
	sectors = duffy_sectors()

	fig = Figure(size = (1800, 900))
	Label(fig[0, 1:4], "Jacobienne (a,b) -> (rho,theta), a=$(round(a_eval, digits = 3)), xhat=($(xhat[1]), $(xhat[2]))", fontsize = 22)

	for (k, (xiI, xiII, tau)) in enumerate(sectors)
		drho_da_vals, drho_db_vals, dtheta_da_vals, dtheta_db_vals =
			compute_component_curves(xiI, xiII, xhat, a_eval, bs)

		bstar = bstar_on_sector(xiI, xiII, xhat)

		row = div(k - 1, 2) + 1
		plot_col = 2 * mod(k - 1, 2) + 1
		legend_col = plot_col + 1
		ax = Axis(fig[row, plot_col], xlabel = "b", ylabel = "valeur", title = "Secteur $tau")

		lines!(ax, bs, drho_da_vals, color = :royalblue, linewidth = 2, label = "drho/da")
		lines!(ax, bs, drho_db_vals, color = :seagreen, linewidth = 2, label = "drho/db")
		lines!(ax, bs, dtheta_da_vals, color = :darkorange, linewidth = 2, label = "dtheta/da")
		lines!(ax, bs, dtheta_db_vals, color = :firebrick, linewidth = 2, label = "dtheta/db")
		Legend(fig[row, legend_col], ax, framevisible = true)

		if isfinite(bstar)
			xlabel_rel = clamp(bstar + 0.015, 0.0, 1.0)
			vlines!(ax, [bstar], color = :black, linestyle = :dash, linewidth = 2)
			text!(ax, xlabel_rel, 0.5, text = "b*=$(round(bstar, digits = 4))", rotation = pi / 2, align = (:center, :center), space = :relative)
		end
	end

	colsize!(fig.layout, 1, Relative(0.42))
	colsize!(fig.layout, 2, Relative(0.08))
	colsize!(fig.layout, 3, Relative(0.42))
	colsize!(fig.layout, 4, Relative(0.08))
	rowsize!(fig.layout, 1, Relative(0.5))
	rowsize!(fig.layout, 2, Relative(0.5))

	return fig
end

fig = plot_duffy_jacobian(xhat, a_eval = a_eval)
display(GLMakie.Screen(), fig)

# GLMakie.save("./dev/figures/navier/navier_hypersingular_distorsion.png", fig)

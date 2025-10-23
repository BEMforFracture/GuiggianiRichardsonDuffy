import GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using BenchmarkTools
using PrettyTables

# INPUTS

x̂ = SVector(0.5, 0.5) # source point in reference coordinates

### Richardson extrapolation parameters
maxeval = 5
rtol = 0.0
atol = 0.0
contract = 0.5
first_contract = 1e-2
breaktol = Inf

# quadrature parameters
n_rho = 10
n_theta = 40

# benchmark parameters
n_sample = 1000
seconds = 0.1
evals = 10

# END INPUTS

δ = 0.5
z = 0.0
y¹ = SVector(-1.0, -1.0, z)
y² = SVector(1.0 + δ, -1.0, z)
y³ = SVector(-1.0, 1.0, z)
y⁴ = SVector(1.0 - δ, 1.0, z)
nodes = (y¹, y², y³, y⁴)

el = Inti.LagrangeSquare(nodes)
x = el(x̂)
expected_I = GRD.hypersingular_laplace_integral_on_plane_element(x, el)
ref_domain = Inti.reference_domain(el)
û = ξ -> 1.0

K = GRD.SplitLaplaceHypersingular

b_dict_gui = Dict{Symbol, BenchmarkTools.Trial}()

errors = Dict{Symbol, Float64}()

for method in GRD.EXPANSION_METHODS
	# Calculate result for error computation
	res = GRD.guiggiani_singular_integral(
		K,
		û,
		x̂,
		el,
		n_rho,
		n_theta;
		sorder = Val(-2),
		expansion = method,
		rtol = rtol,
		maxeval = maxeval,
		first_contract = first_contract,
		breaktol = breaktol,
		contract = contract,
		atol = atol,
	)
	error = abs(res - expected_I) / abs(expected_I)
	errors[method] = error

	# Benchmark
	b = @benchmark begin
		GRD.guiggiani_singular_integral(
			$K,
			$û,
			$x̂,
			$el,
			$n_rho,
			$n_theta;
			sorder = Val(-2),
			expansion = $method,
			rtol = $rtol,
			maxeval = $maxeval,
			first_contract = $first_contract,
			breaktol = $breaktol,
			contract = $contract,
			atol = $atol,
		)
	end samples = n_sample seconds = seconds evals = evals
	b_dict_gui[method] = b
end

for (method, b) in b_dict_gui
	println("Singular integral, ", method, ":")
	display(b)
	println()
end

b_dict_laurent = Dict{Symbol, BenchmarkTools.Trial}()

for method in GRD.EXPANSION_METHODS
	b = @benchmark begin
		ℒ = GRD.laurents_coeffs(
			$K,
			$el,
			$û,
			$x̂;
			expansion = $method,
			maxeval = $maxeval,
			rtol = $rtol,
			atol = $atol,
			contract = $contract,
			first_contract = $first_contract,
			breaktol = $breaktol,
		)
	end samples = n_sample seconds = seconds evals = evals
	b_dict_laurent[method] = b
end

for (method, b) in b_dict_laurent
	println("Laurent coefficients, ", method, ":")
	display(b)
	println()
end

b_dict_eval = Dict{Symbol, BenchmarkTools.Trial}()

for method in GRD.EXPANSION_METHODS
	ℒ = GRD.laurents_coeffs(
		K,
		el,
		û,
		x̂;
		expansion = method,
		maxeval = maxeval,
		rtol = rtol,
		atol = atol,
		contract = contract,
		first_contract = first_contract,
		breaktol = breaktol,
	)
	b = @benchmark begin
		θ = rand() * 2π
		f₋₂, f₋₁ = $ℒ(θ)
	end samples = n_sample seconds = seconds evals = evals
	b_dict_eval[method] = b
end

for (method, b) in b_dict_eval
	println("Laurent coefficients evaluation, ", method, ":")
	display(b)
	println()
end

methods_names = [string(m) for m in GRD.EXPANSION_METHODS]

# execution times in microseconds
t_integrals = [median(b_dict_gui[m].times) / 1e3 for m in GRD.EXPANSION_METHODS]
t_laurents = [median(b_dict_laurent[m].times) / 1e3 for m in GRD.EXPANSION_METHODS]
t_evals = [median(b_dict_eval[m].times) / 1e3 for m in GRD.EXPANSION_METHODS]

# allocations in number of allocations
a_integrals = [median(b_dict_gui[m].allocs) for m in GRD.EXPANSION_METHODS]
a_laurents = [median(b_dict_laurent[m].allocs) for m in GRD.EXPANSION_METHODS]
a_evals = [median(b_dict_eval[m].allocs) for m in GRD.EXPANSION_METHODS]

# garbage collect in bytes
g_integrals = [median(b_dict_gui[m].gctimes) for m in GRD.EXPANSION_METHODS]
g_laurents = [median(b_dict_laurent[m].gctimes) for m in GRD.EXPANSION_METHODS]
g_evals = [median(b_dict_eval[m].gctimes) for m in GRD.EXPANSION_METHODS]

# memory estimated in kilobytes
m_integrals = [median(b_dict_gui[m].memory) / 1e3 for m in GRD.EXPANSION_METHODS]
m_laurents = [median(b_dict_laurent[m].memory) / 1e3 for m in GRD.EXPANSION_METHODS]
m_evals = [median(b_dict_eval[m].memory) / 1e3 for m in GRD.EXPANSION_METHODS]

# relative errors
errors_values = [errors[m] for m in GRD.EXPANSION_METHODS]

# make 4 tables, one for each metric

data_exec_time = hcat(methods_names, t_integrals, t_laurents, t_evals, errors_values)
data_allocs = hcat(methods_names, a_integrals, a_laurents, a_evals, errors_values)
data_gc_bytes = hcat(methods_names, g_integrals, g_laurents, g_evals, errors_values)
data_memory = hcat(methods_names, m_integrals, m_laurents, m_evals, errors_values)

open("dev/results/laplace/benchmark_laplace_hypersingular.html", "w") do io
	write(io, "<h1>Benchmark Laplace Hypersingular Kernel</h1>\n")
	write(io, "<h2>Parameters used:</h2>\n")
	write(io, "<h3>Source point</h3>\n")
	write(io, "<ul>\n")
	write(io, "<li>x̂ = ($(x̂[1]), $(x̂[2]))</li>\n")
	write(io, "</ul>\n")
	write(io, "<h3>Quadrature</h3>\n")
	write(io, "<ul>\n")
	write(io, "<li>n_rho = $n_rho</li>\n")
	write(io, "<li>n_theta = $n_theta</li>\n")
	write(io, "</ul>\n")
	write(io, "<h3>Richardson Extrapolation</h3>\n")
	write(io, "<ul>\n")
	write(io, "<li>maxeval = $maxeval</li>\n")
	write(io, "<li>rtol = $rtol</li>\n")
	write(io, "<li>atol = $atol</li>\n")
	write(io, "<li>first_contract = $first_contract</li>\n")
	write(io, "<li>contract = $contract</li>\n")
	write(io, "<li>breaktol = $breaktol</li>\n")
	write(io, "</ul>\n")
	write(io, "<h3>Benchmark</h3>\n")
	write(io, "<ul>\n")
	write(io, "<li>n_sample = $n_sample</li>\n")
	write(io, "<li>seconds = $seconds</li>\n")
	write(io, "<li>evals = $evals</li>\n")
	write(io, "</ul>\n")
	write(io, "<h2>Execution Time (μs)</h2>\n")
	pretty_table(
		io,
		data_exec_time;
		column_labels = ["Méthode", "Intégrale", "Coefficients", "Évaluation", "Relative Error"],
		backend = :html,
		alignment = [:l, :r, :r, :r, :r],
	)

	write(io, "<h2>Allocations (nombre)</h2>\n")
	pretty_table(
		io,
		data_allocs;
		column_labels = ["Méthode", "Intégrale", "Coefficients", "Évaluation", "Relative Error"],
		backend = :html,
		alignment = [:l, :r, :r, :r, :r],
	)

	write(io, "<h2>Garbage Collection Time (ns)</h2>\n")
	pretty_table(
		io,
		data_gc_bytes;
		column_labels = ["Méthode", "Intégrale", "Coefficients", "Évaluation", "Relative Error"],
		backend = :html,
		alignment = [:l, :r, :r, :r, :r],
	)

	write(io, "<h2>Memory (kB)</h2>\n")
	pretty_table(
		io,
		data_memory;
		column_labels = ["Méthode", "Intégrale", "Coefficients", "Évaluation", "Relative Error"],
		backend = :html,
		alignment = [:l, :r, :r, :r, :r],
	)
end


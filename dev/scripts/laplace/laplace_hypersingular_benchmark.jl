import GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using BenchmarkTools
using PrettyTables

# Configuration
x̂ = SVector(0.5, 0.5)  # Source point in reference coordinates

# Richardson extrapolation parameters
rich_params = GRD.RichardsonParams(
	first_contract = 1e-2,
	contract = 0.5,
	breaktol = Inf,
	atol = 0.0,
	rtol = 0.0,
	maxeval = 5,
)

# Quadrature parameters
n_rho = 10
n_theta = 40
quad_rho = Inti.GaussLegendre(n_rho)
quad_theta = Inti.GaussLegendre(n_theta)

# Benchmark parameters
n_sample = 1000
seconds = 0.1
evals = 10

# Setup element
δ = 0.5
z = 0.0
y¹ = SVector(-1.0, -1.0, z)
y² = SVector(1.0 + δ, -1.0, z)
y³ = SVector(-1.0, 1.0, z)
y⁴ = SVector(1.0 - δ, 1.0, z)
nodes = (y¹, y², y³, y⁴)
el = Inti.LagrangeSquare(nodes)

# Reference value
x = el(x̂)
expected_I = GRD.hypersingular_laplace_integral_on_plane_element(x, el)

# Density function
û = ξ -> 1.0

# Kernel setup
K_base = Inti.HyperSingularKernel(Inti.Laplace(dim=3))
K = GRD.SplitKernel(K_base)

# Methods to benchmark
methods = [
	("Analytical", GRD.AnalyticalExpansion()),
	("AutoDiff", GRD.AutoDiffExpansion()),
	("SemiRichardson", GRD.SemiRichardsonExpansion(rich_params)),
	("FullRichardson", GRD.FullRichardsonExpansion(rich_params)),
]

b_dict_gui = Dict{String, BenchmarkTools.Trial}()
errors = Dict{String, Float64}()

for (method_name, method) in methods
	@info "Benchmarking $method_name..."
	
	# Use appropriate kernel (base for analytical, split for others)
	K_to_use = (method isa GRD.AnalyticalExpansion) ? K_base : K
	
	# Calculate result for error computation
	res = GRD.guiggiani_singular_integral(
		K_to_use, û, x̂, el, quad_rho, quad_theta, method
	)
	error = abs(res - expected_I) / abs(expected_I)
	errors[method_name] = error

	# Benchmark
	b = @benchmark GRD.guiggiani_singular_integral(
		$K_to_use, $û, $x̂, $el, $quad_rho, $quad_theta, $method
	) samples = n_sample seconds = seconds evals = evals
	
	b_dict_gui[method_name] = b
end

for (method, b) in b_dict_gui
	println("Singular integral, ", method, ":")
	display(b)
	println()
end

b_dict_laurent = Dict{String, BenchmarkTools.Trial}()

for (method_name, method) in methods
	@info "Benchmarking Laurent coefficients for $method_name..."
	
	# Use appropriate kernel
	K_to_use = (method isa GRD.AnalyticalExpansion) ? K_base : K
	
	b = @benchmark GRD.laurents_coeffs(
		$K_to_use, $el, $û, $x̂, $method
	) samples = n_sample seconds = seconds evals = evals
	
	b_dict_laurent[method_name] = b
end

for (method, b) in b_dict_laurent
	println("Laurent coefficients, ", method, ":")
	display(b)
	println()
end

b_dict_eval = Dict{String, BenchmarkTools.Trial}()

for (method_name, method) in methods
	@info "Benchmarking Laurent evaluation for $method_name..."
	
	# Use appropriate kernel
	K_to_use = (method isa GRD.AnalyticalExpansion) ? K_base : K
	
	# Create Laurent expander
	ℒ = GRD.laurents_coeffs(K_to_use, el, û, x̂, method)
	
	b = @benchmark begin
		θ = rand() * 2π
		f₋₂, f₋₁ = $ℒ(θ)
	end samples = n_sample seconds = seconds evals = evals
	
	b_dict_eval[method_name] = b
end

for (method, b) in b_dict_eval
	println("Laurent coefficients evaluation, ", method, ":")
	display(b)
	println()
end

# Extract method names
methods_names = [m[1] for m in methods]

# Execution times in microseconds
t_integrals = [median(b_dict_gui[m].times) / 1e3 for m in methods_names]
t_laurents = [median(b_dict_laurent[m].times) / 1e3 for m in methods_names]
t_evals = [median(b_dict_eval[m].times) / 1e3 for m in methods_names]

# Allocations
a_integrals = [median(b_dict_gui[m].allocs) for m in methods_names]
a_laurents = [median(b_dict_laurent[m].allocs) for m in methods_names]
a_evals = [median(b_dict_eval[m].allocs) for m in methods_names]

# Garbage collection time
g_integrals = [median(b_dict_gui[m].gctimes) for m in methods_names]
g_laurents = [median(b_dict_laurent[m].gctimes) for m in methods_names]
g_evals = [median(b_dict_eval[m].gctimes) for m in methods_names]

# Memory in kilobytes
m_integrals = [median(b_dict_gui[m].memory) / 1e3 for m in methods_names]
m_laurents = [median(b_dict_laurent[m].memory) / 1e3 for m in methods_names]
m_evals = [median(b_dict_eval[m].memory) / 1e3 for m in methods_names]

# Relative errors
errors_values = [errors[m] for m in methods_names]

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
	write(io, "<li>maxeval = $(rich_params.maxeval)</li>\n")
	write(io, "<li>rtol = $(rich_params.rtol)</li>\n")
	write(io, "<li>atol = $(rich_params.atol)</li>\n")
	write(io, "<li>first_contract = $(rich_params.first_contract)</li>\n")
	write(io, "<li>contract = $(rich_params.contract)</li>\n")
	write(io, "<li>breaktol = $(rich_params.breaktol)</li>\n")
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


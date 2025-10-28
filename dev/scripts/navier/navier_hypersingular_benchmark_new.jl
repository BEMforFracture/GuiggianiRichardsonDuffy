import GuiggianiRichardsonDuffy as GRD

using Inti
using StaticArrays
using BenchmarkTools
using PrettyTables
using LinearAlgebra

# Configuration
x̂ = SVector(0.5, 0.5)

# Material properties
μ = 1.0
λ = 1.0

# Richardson parameters
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
x = el(x̂)
ref_domain = Inti.reference_domain(el)
û = ξ -> 1.0

# Kernel setup
K_base = Inti.ElastostaticHypersingular(μ, λ)
K = GRD.SplitKernel(K_base)

# Methods to test
methods = [
	("Analytical", GRD.AnalyticalExpansion(), K_base),
	("AutoDiff", GRD.AutoDiffExpansion(), K),
	("SemiRichardson", GRD.SemiRichardsonExpansion(rich_params), K),
	("FullRichardson", GRD.FullRichardsonExpansion(rich_params), K),
]

# Storage
b_dict_gui = Dict{String, BenchmarkTools.Trial}()
b_dict_laurent = Dict{String, BenchmarkTools.Trial}()
b_dict_eval = Dict{String, BenchmarkTools.Trial}()
errors = Dict{String, Float64}()

@info "Running benchmarks for Navier hypersingular kernel..."

# Benchmark integral computation
for (method_name, method, K_to_use) in methods
	@info "Benchmarking integral: $method_name"
	
	res = GRD.guiggiani_singular_integral(
		K_to_use, û, x̂, el, n_rho, n_theta, method
	)
	
	if method_name != "Analytical"
		ref_value = GRD.guiggiani_singular_integral(
			K_base, û, x̂, el, n_rho, n_theta, GRD.AnalyticalExpansion()
		)
		errors[method_name] = norm(res - ref_value) / norm(ref_value)
	else
		errors[method_name] = 0.0
	end
	
	b = @benchmark GRD.guiggiani_singular_integral(
		$K_to_use, $û, $x̂, $el, $n_rho, $n_theta, $method
	) samples = n_sample seconds = seconds evals = evals
	
	b_dict_gui[method_name] = b
end

# Benchmark Laurent coefficients computation
for (method_name, method, K_to_use) in methods
	@info "Benchmarking Laurent coefficients: $method_name"
	
	b = @benchmark GRD.laurents_coeffs(
		$K_to_use, $el, $û, $x̂, $method
	) samples = n_sample seconds = seconds evals = evals
	
	b_dict_laurent[method_name] = b
end

# Benchmark Laurent coefficients evaluation
for (method_name, method, K_to_use) in methods
	@info "Benchmarking Laurent evaluation: $method_name"
	
	ℒ = GRD.laurents_coeffs(K_to_use, el, û, x̂, method)
	
	b = @benchmark begin
		θ = rand() * 2π
		f₋₂, f₋₁ = $ℒ(θ)
	end samples = n_sample seconds = seconds evals = evals
	
	b_dict_eval[method_name] = b
end

@info "Benchmarks complete. Generating tables..."

# Print results
for (method_name, b) in b_dict_gui
	println("Singular integral, ", method_name, ":")
	display(b)
	println()
end

for (method_name, b) in b_dict_laurent
	println("Laurent coefficients, ", method_name, ":")
	display(b)
	println()
end

for (method_name, b) in b_dict_eval
	println("Laurent coefficients evaluation, ", method_name, ":")
	display(b)
	println()
end

# Extract data for tables
method_names = [m[1] for m in methods]

# Execution times (μs)
t_integrals = [median(b_dict_gui[m].times) / 1e3 for m in method_names]
t_laurents = [median(b_dict_laurent[m].times) / 1e3 for m in method_names]
t_evals = [median(b_dict_eval[m].times) / 1e3 for m in method_names]

# Allocations
a_integrals = [median(b_dict_gui[m].allocs) for m in method_names]
a_laurents = [median(b_dict_laurent[m].allocs) for m in method_names]
a_evals = [median(b_dict_eval[m].allocs) for m in method_names]

# GC times (ns)
g_integrals = [median(b_dict_gui[m].gctimes) for m in method_names]
g_laurents = [median(b_dict_laurent[m].gctimes) for m in method_names]
g_evals = [median(b_dict_eval[m].gctimes) for m in method_names]

# Memory (kB)
m_integrals = [median(b_dict_gui[m].memory) / 1e3 for m in method_names]
m_laurents = [median(b_dict_laurent[m].memory) / 1e3 for m in method_names]
m_evals = [median(b_dict_eval[m].memory) / 1e3 for m in method_names]

# Errors
errors_values = [errors[m] for m in method_names]

# Create tables
data_exec_time = hcat(method_names, t_integrals, t_laurents, t_evals, errors_values)
data_allocs = hcat(method_names, a_integrals, a_laurents, a_evals, errors_values)
data_gc_bytes = hcat(method_names, g_integrals, g_laurents, g_evals, errors_values)
data_memory = hcat(method_names, m_integrals, m_laurents, m_evals, errors_values)

# Write HTML report
open("dev/results/navier/benchmark_navier_hypersingular.html", "w") do io
	write(io, "<h1>Benchmark Navier Hypersingular Kernel</h1>\n")
	write(io, "<h2>Parameters used:</h2>\n")
	write(io, "<h3>Source point</h3>\n")
	write(io, "<ul>\n")
	write(io, "<li>x̂ = ($(x̂[1]), $(x̂[2]))</li>\n")
	write(io, "</ul>\n")
	write(io, "<h3>Material properties</h3>\n")
	write(io, "<ul>\n")
	write(io, "<li>μ = $μ</li>\n")
	write(io, "<li>λ = $λ</li>\n")
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

@info "Results saved to dev/results/navier/benchmark_navier_hypersingular.html"

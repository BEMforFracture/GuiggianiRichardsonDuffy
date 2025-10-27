# Activer le projet des docs et charger le package
push!(LOAD_PATH, "..")

using Documenter
using DocumenterCitations
using GuiggianiRichardsonDuffy

bib = CitationBibliography(
	joinpath(@__DIR__, "src", "refs.bib");
	style = :numeric,
)

# Options de build
makedocs(
	sitename = "GuiggianiRichardsonDuffy.jl",
	modules = [GuiggianiRichardsonDuffy],
	format = Documenter.HTML(; prettyurls = false),
	pages = [
		"Welcome" => "index.md",
		"Guide" => "guide.md",
		"API" => "api.md",
		"Docstrings" => "docstrings.md",
		"Appendix" => "appendix.md",
		"References" => "references.md",
	],
	doctest = true,
	warnonly = [:missing_docs, :docs_block],  # Permet de construire malgré les docstrings manquantes
	plugins = [bib],
)

# Déploiement sur GitHub Pages
deploydocs(
	repo = "github.com/Aguelord/GuiggianiRichardsonDuffy",
	devbranch = "main",
)

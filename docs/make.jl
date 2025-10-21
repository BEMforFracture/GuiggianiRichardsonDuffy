# Activer le projet des docs et charger le package
push!(LOAD_PATH, "..")

using Documenter
using GuiggianiRichardsonDuffy

# Options de build
makedocs(
	sitename = "GuiggianiRichardsonDuffy.jl",
	modules = [GuiggianiRichardsonDuffy],
	format = Documenter.HTML(; prettyurls = false),
	pages = [
		"Accueil" => "index.md",
		"Guide" => "guide.md",
		"API" => "api.md",
		"Docstrings" => "docstrings.md",
	],
	doctest = true,
	warnonly = [:missing_docs, :docs_block],  # Permet de construire malgré les docstrings manquantes
)

# Déploiement sur GitHub Pages
deploydocs(
	repo = "github.com/Aguelord/GuiggianiRichardsonDuffy",
	devbranch = "main",
)

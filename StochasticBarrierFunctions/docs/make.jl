using Revise
using StochasticBarrierFunctions
using Documenter

push!(LOAD_PATH, "../src/")
DocMeta.setdocmeta!(StochasticBarrierFunctions, :DocTestSetup, :(using StochasticBarrierFunctions); recursive = true)

makedocs(;
    modules = [StochasticBarrierFunctions],
    authors = "Rayan Mazouz <rama7481@colorado.edu>, Frederik Baymler Mathiesen <frederik@baymler.com>, and contributors",
    sitename = "StochasticBarrierFunctions.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://aria-systems-group.github.io/StochasticBarrierFunctions.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Reference" => Any[
            "Barrier algorithms" => "reference/algorithms.md",
        ]
    ],
    doctest = false,
    checkdocs = :exports,
)

deploydocs(; repo = "github.com/aria-systems-group/StochasticBarrierFunctions.jl", devbranch = "main")

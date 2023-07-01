using SpectralDensities
using Documenter

DocMeta.setdocmeta!(SpectralDensities, :DocTestSetup, :(using SpectralDensities); recursive=true)

makedocs(;
    modules=[SpectralDensities],
    authors="Federico Cerisola <federico@cerisola.net",
    repo="https://github.com/quantum-exeter/SpectralDensities.jl/blob/{commit}{path}#{line}",
    sitename="SpectralDensities.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://quantum-exeter.github.io/SpectralDensities.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/quantum-exeter/SpectralDensities.jl",
    devbranch="main",
)

using ReplicationLanteriRampini2023
using Documenter

DocMeta.setdocmeta!(ReplicationLanteriRampini2023, :DocTestSetup, :(using ReplicationLanteriRampini2023); recursive=true)

makedocs(;
    modules=[ReplicationLanteriRampini2023],
    authors="alexgrlt <98583667+alexgrlt@users.noreply.github.com> and contributors",
    repo="https://github.com/alexgrlt/ReplicationLanteriRampini2023.jl/blob/{commit}{path}#{line}",
    sitename="ReplicationLanteriRampini2023.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://alexgrlt.github.io/ReplicationLanteriRampini2023.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/alexgrlt/ReplicationLanteriRampini2023.jl",
    devbranch="master",
)

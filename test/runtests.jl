using Pkg: Pkg
using EnsembleKalmanFilters
using Test
using TestReports
using Aqua
using Documenter

ts = @testset ReportingTestSet "" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(EnsembleKalmanFilters; ambiguities=false)
        Aqua.test_ambiguities(EnsembleKalmanFilters)
    end

    @testset "Unit tests" begin
        include("test_assimilate_data.jl")
    end

    # Set metadata for doctests.
    DocMeta.setdocmeta!(
        EnsembleKalmanFilters,
        :DocTestSetup,
        :(using EnsembleKalmanFilters, Test);
        recursive=true,
    )

    # Run doctests.
    doctest(EnsembleKalmanFilters; manual=true)
end

outputfilename = joinpath(@__DIR__, "..", "report.xml")
open(outputfilename, "w") do fh
    print(fh, report(ts))
end
@test !any_problems(ts)

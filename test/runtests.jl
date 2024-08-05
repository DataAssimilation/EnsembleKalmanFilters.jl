using Pkg: Pkg
using EnsembleKalmanFilter
using Test
using TestReports
using Aqua
using Documenter

ts = @testset ReportingTestSet "" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(EnsembleKalmanFilter; ambiguities=false)
        Aqua.test_ambiguities(EnsembleKalmanFilter)
    end

    include("test_pkg_stuff.jl")

    # Set metadata for doctests.
    DocMeta.setdocmeta!(EnsembleKalmanFilter, :DocTestSetup, :(using EnsembleKalmanFilter, Test); recursive=true)
    if EnsembleKalmanFilter.HAS_NATIVE_EXTENSIONS
        using Random
        DocMeta.setdocmeta!(
            EnsembleKalmanFilter.get_extension(EnsembleKalmanFilter, :RandomExt),
            :DocTestSetup,
            :(using EnsembleKalmanFilter, Test);
            recursive=true,
        )
    end

    # Run doctests.
    doctest(EnsembleKalmanFilter; manual=true)
    if EnsembleKalmanFilter.HAS_NATIVE_EXTENSIONS
        doctest(EnsembleKalmanFilter.get_extension(EnsembleKalmanFilter, :RandomExt); manual=true)
    end

    # Run examples.
    examples_dir = joinpath(@__DIR__, "..", "examples")
    for example in readdir(examples_dir)
        example_path = joinpath(examples_dir, example)
        @show example_path
        orig_project = Base.active_project()
        @testset "Example: $(example)" begin
            if isdir(example_path)
                Pkg.activate(example_path)
                Pkg.develop(; path=joinpath(@__DIR__, ".."))
                Pkg.instantiate()
            end
            script_path = joinpath(example_path, "main.jl")
            try
                include(script_path)
                println("Included script_path")
            finally
                Pkg.activate(orig_project)
            end
        end
    end
end

outputfilename = joinpath(@__DIR__, "..", "report.xml")
open(outputfilename, "w") do fh
    print(fh, report(ts))
end
exit(any_problems(ts))

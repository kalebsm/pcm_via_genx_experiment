using Dates
using Printf

function run_CEM_and_LAC()
    println(">>> Starting CEM case run at $(Dates.format(now(), "HH:MM:SS"))")
    println(pwd())
    cd("GenX.jl")
    run(`julia -e '
        using Pkg;
        Pkg.activate(".");
        Pkg.instantiate();
        include("src/run_CEM_cases.jl");
        include("src/cem_to_lac_capacity_update.jl")'
    `)
    cd("..")

    println(">>> CEM run complete. Transferring data to LAC...")

    cd("SPCM")
    run(`julia -e '
        using Pkg;
        Pkg.activate(".");
        Pkg.develop(path="src/scenario_generation/sequential_norta");
        Pkg.instantiate();
        include("src/run_script.jl")'`)

    println(">>> LAC case run complete at $(Dates.format(now(), "HH:MM:SS"))")
end

run_CEM_and_LAC()

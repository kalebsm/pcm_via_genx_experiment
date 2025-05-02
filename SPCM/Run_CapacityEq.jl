using Revise
push!(LOAD_PATH, "src/SPCMviaGenX.jl")
ENV["GENX_PRECOMPILE"] = "false"
using SPCMviaGenX
using Gurobi

mainpath = pwd()
case_path = joinpath(mainpath,"SPCM", "Research_Systems", "2_Hr_BESS")
run_genx_case!(case_path, Gurobi.Optimizer)
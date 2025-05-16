using Revise
push!(LOAD_PATH, "src/SPCMviaGenX.jl")
ENV["GENX_PRECOMPILE"] = "false"
using SPCMviaGenX
using Gurobi
using CSV
using DataFrames
mainpath = pwd()
case_path = joinpath(mainpath,"SPCM", "Research_Systems", "2_Hr_BESS_QUAD")
# run_genx_case!(case_path, Gurobi.Optimizer)
context = initialize_policy_model(case_path)
cem_capacities = CSV.read(joinpath(case_path, "results", "capacity.csv"), DataFrame).EndCap[1:context["inputs"]["G"]]
# Start with initial capacities
model_type = "dlac-i"
current_capacities = cem_capacities*1e-3  # Convert to GW
results = run_policy_model_new(context, model_type, current_capacities,write_results=true)

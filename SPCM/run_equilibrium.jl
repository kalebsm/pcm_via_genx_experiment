using Revise
push!(LOAD_PATH, "src/SPCMviaGenX.jl")
ENV["GENX_PRECOMPILE"] = "false"
using SPCMviaGenX
includet("equilibrium/CapacityEquilibrium.jl")
using .CapacityEquilibrium
using Gurobi
using CSV
using DataFrames
mainpath = pwd()
case_path = joinpath(mainpath,"SPCM", "Research_Systems", "2_Hr_BESS_QUAD")
context = initialize_policy_model(case_path)
model_type = "pf"
cem_capacities = CSV.read(joinpath(case_path, "results", "capacity.csv"), DataFrame).EndCap[1:context["inputs"]["G"]]
current_capacities = cem_capacities*1e-3 
current_capacities = [34.801825952408926,12.723730164650556,50.43440908203988,25.044957836788758,14.142482399322795]
result = compute_equilibrium(context, model_type; initial_capacities=current_capacities)

# # Analyze results
# analyze_equilibrium(result)

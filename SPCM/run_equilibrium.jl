using Revise
push!(LOAD_PATH, "src/SPCMviaGenX.jl")
includet("equilibrium/CapacityEquilibrium.jl")
ENV["GENX_PRECOMPILE"] = "false"
using SPCMviaGenX
using .CapacityEquilibrium
using Gurobi

mainpath = pwd()
case_path = joinpath(mainpath, "Research_Systems", "2_Hr_BESS")
context = initialize_policy_model(case_path, offset=true)
model_type = "dlac-p"
result = compute_equilibrium(context, model_type)

# Analyze results
analyze_equilibrium(result)

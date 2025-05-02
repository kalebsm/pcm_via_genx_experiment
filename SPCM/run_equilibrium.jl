using Revise
push!(LOAD_PATH, "src/SPCMviaGenX.jl")
ENV["GENX_PRECOMPILE"] = "false"
using SPCMviaGenX
includet("equilibrium/CapacityEquilibrium.jl")
using .CapacityEquilibrium
using Gurobi

mainpath = pwd()
case_path = joinpath(mainpath,"SPCM", "Research_Systems", "2_Hr_BESS")
context = initialize_policy_model(case_path, offset=true)
model_type = "dlac-p"
result = compute_equilibrium(context, model_type; initial_capacities=[34.36606076280974,12.885568330276815,49.74372506493803,25.845168472951674,12.37495698409709])

# Analyze results
analyze_equilibrium(result)

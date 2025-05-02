using Revise
push!(LOAD_PATH, "src/SPCMviaGenX.jl")
ENV["GENX_PRECOMPILE"] = "false"
using SPCMviaGenX
using Gurobi
mainpath = pwd()
case_path = joinpath(mainpath,"SPCM", "Research_Systems", "2_Hr_BESS_QUAD")
run_genx_case!(case_path, Gurobi.Optimizer)
context = initialize_policy_model(case_path)
# Start with initial capacities
model_type = "pf"
# current_capacities =[34646.96955039550,12954.406420649300,49965.193438319700, 25930.402201271500, 13039.450125359200] #linear
current_capacities = [34214.9928562353, 13626.402415722800, 48454.75598424260, 26481.92638172560, 12081.018266599000]
current_capacities = current_capacities*1e-3  # Convert to GW # Empty dict means use default capacities
results = run_policy_model_new(context, model_type, current_capacities,write_results=true)

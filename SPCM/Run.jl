using Revise
push!(LOAD_PATH, "src/SPCMviaGenX.jl")
ENV["GENX_PRECOMPILE"] = "false"
using SPCMviaGenX
using Gurobi
subfolders = [    
            #   "Thermal_Base",
              "2_Hr_BESS", 
              "2_Hr_BESS_Fuelx2",
              "4_Hr_BESS",
              "4_Hr_BESS_Fuelx2",
              "4_Hr_BESS_Fuelx3",
              "4_Hr_BESS_Fuelx4",
            #   "6_Hr_BESS",
            #   "6_Hr_BESS_Fuelx2",
            #   "8_Hr_BESS",
            #   "8_Hr_BESS_Fuelx2",
            #   "10_Hr_BESS",
            #   "10_Hr_BESS_Fuelx2",
              ]

mainpath = pwd()
case_path = joinpath(mainpath,"SPCM", "Research_Systems", "2_Hr_BESS")
context = initialize_policy_model(case_path)
# Start with initial capacities
model_type = "pf"
current_capacities =[]
# current_capacities = [34.46234822698904,12.888982800636322,49.83420557022405,25.87740856354249,12.653832031075554]  # Empty dict means use default capacities
results = run_policy_model_new(context, model_type, current_capacities,write_results=true)

for folder in subfolders
    runpath = joinpath(mainpath,"SPCM", "Research_Systems", folder)

        
    println("Navigating to $folder")
    # run_genx_case!(runpath, Gurobi.Optimizer)
    run_policy_model(runpath, "pf")
end

println("All subprocesses complete.")

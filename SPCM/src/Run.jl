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


for folder in subfolders
    runpath = joinpath(mainpath,"SPCM", "Research_Systems", folder)

        
    println("Navigating to $folder")
    # run_genx_case!(runpath, Gurobi.Optimizer)
    run_policy_model(runpath, "pf")
end

println("All subprocesses complete.")

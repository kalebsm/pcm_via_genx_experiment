# Helper function to log results
function log_process(file, script_name, folder)
    try
        Base.include(Main, file)  # Use Base.include to avoid polluting global namespace
        println(log, "SUCCESS: $script_name ran successfully in $folder")
    catch e
        # println(log, "ERROR: $script_name failed in $folder with error: $e")
        # print an error message
        error("ERROR: $script_name failed in $folder with error: $e")
    end
    GC.gc()  # Trigger garbage collection to free memory
end

# function log_process(file, script_name, folder)
#     Base.include(Main, file)  # Use Base.include to avoid polluting global namespace

#     GC.gc()  # Trigger garbage collection to free memory
# end

subfolders = [    
            #   "Thermal_Base",
              "2_Hr_BESS", 
              "2_Hr_BESS_Fuelx2",
              "4_Hr_BESS",
              "4_Hr_BESS_Fuelx2",
              "4_Hr_BESS_Fuelx3",
              "4_Hr_BESS_Fuelx4",
              "6_Hr_BESS",
              "6_Hr_BESS_Fuelx2",
              "8_Hr_BESS",
              "8_Hr_BESS_Fuelx2",
              "10_Hr_BESS",
              "10_Hr_BESS_Fuelx2",
              ]

mainpath = pwd()

# Define the log file path
log_file = joinpath(mainpath, "process_log.txt")

# Open the log file for writing
open(log_file, "w") do log
    println(log, "Process Log")
    println(log, "===========")

    for folder in subfolders
        # Get runpath
        runpath = joinpath(mainpath, "Research_Systems", folder)

        # Log navigation to the folder
        println(log, "Navigating to $folder")
        println("Navigating to $folder")

        # Run the processes
        println(log, "Running processes in $folder")
        println("Running processes in $folder")



        # Run each script and log the results
        log_process(joinpath(runpath, "Run_spcm_pf.jl"), "Run_spcm_pf.jl", folder)
        log_process(joinpath(runpath, "Run_spcm_dlac-p.jl"), "Run_spcm_dlac-p.jl", folder)
        log_process(joinpath(runpath, "Run_spcm_dlac-i.jl"), "Run_spcm_dlac-i.jl", folder)
        log_process(joinpath(runpath, "Run_spcm_slac.jl"), "Run_spcm_slac.jl", folder)
    end

    println(log, "All subprocesses complete.")
    println("All subprocesses complete.")
end
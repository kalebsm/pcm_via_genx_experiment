using GenX
using Gurobi





### Run.jl
case = dirname(@__FILE__)
optimizer = Gurobi.Optimizer

# input scenario generation information


### case_runner.jl

function get_settings_path(case::AbstractString)
    return joinpath(case, "settings")
end

function get_settings_path(case::AbstractString, filename::AbstractString)
    return joinpath(get_settings_path(case), filename)
end

function get_default_output_folder(case::AbstractString)
    return joinpath(case, "results")
end

### print_genx_version()
v = pkgversion(GenX)
ascii_art = raw"""
  ______   _______     ______  ____    ____  
.' ____ \ |_   __ \  .' ___  ||_   \  /   _| 
| (___ \_|  | |__) |/ .'   \_|  |   \/   |   
 _.____`.   |  ___/ | |         | |\  /| |   
| \____) | _| |_    \ `.___.'\ _| |_\/_| |_  
 \______.'|_____|    `.____ .'|_____||_____| 
                                          
 _   __  __   ,--.                           
[ \ [  ][  | `'_\ :                          
 \ \/ /  | | // | |,                         
  \__/__[___]\'-;__/        ____  ____       
 .' ___  |                 |_  _||_  _|      
/ .'   \_|  .---.  _ .--.    \ \  / /        
| |   ____ / /__\\[ `.-. |    > `' <         
\ `.___]  || \__., | | | |  _/ /'`\ \_       
 `._____.'  '.__.'[___||__]|____||____|     
"""
ascii_art *= "GenX Version: $(v)"
println(ascii_art)

genx_settings = get_settings_path(case, "genx_settings.yml") # Settings YAML file path
writeoutput_settings = get_settings_path(case, "output_settings.yml") # Write-output settings YAML file path
mysetup = configure_settings(genx_settings, writeoutput_settings) # mysetup dictionary stores settings and GenX-specific parameters

### run_genx_case_simple
settings_path = get_settings_path(case)

### Cluster time series inputs if necessary and if specified by the user
if mysetup["TimeDomainReduction"] == 1
    TDRpath = joinpath(case, mysetup["TimeDomainReductionFolder"])
    system_path = joinpath(case, mysetup["SystemFolder"])
    prevent_doubled_timedomainreduction(system_path)
    if !time_domain_reduced_files_exist(TDRpath)
        println("Clustering Time Series Data (Grouped)...")
        cluster_inputs(case, settings_path, mysetup)
    else
        println("Time Series Data Already Clustered.")
    end
end

### Configure solver
println("Configuring Solver")
solver_name = lowercase(get(mysetup, "Solver", ""))
OPTIMIZER = configure_solver(settings_path, optimizer; solver_name=solver_name)

#### Running a case

### Load inputs
println("Loading Inputs")
myinputs = load_inputs(mysetup, case)

### Load in Scenario Generation


println("Generating the SPCM Optimization")
# GENERATE SPCM MODEL
# time_elapsed = @elapsed EP = generate_spcm_model(mysetup, myinputs, OPTIMIZER)
println("Time elapsed for model building is")
println(time_elapsed)





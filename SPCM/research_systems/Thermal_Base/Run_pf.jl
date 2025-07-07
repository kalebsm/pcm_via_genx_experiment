using SPCMviaGenX
using Gurobi
using YAML
using BenchmarkTools
using JuMP
using DataFrames
using CSV
using DelimitedFiles
using Distributions
using Random
using HDF5
using Plots # does not work for some reason
using TimeZones
using TSFrames
using Tables
using LinearSolve
using LinearAlgebra
using Dates
using DelimitedFiles
using StatsBase
using BenchmarkTools


# Check memory usage and error out if it exceeds 90%
function check_memory_usage()
    mem_info = Sys.free_memory()
    total_memory = Sys.total_memory()
    used_memory = total_memory - mem_info
    memory_usage_percentage = (used_memory / total_memory) * 100

    if memory_usage_percentage >= 90
        error("Memory usage has exceeded 90%. Exiting to prevent system instability.")
    end
end


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

function hoursbefore(p::Int, t::Int, b::UnitRange{Int})::Vector{Int}
	period = div(t - 1, p)
	return period * p .+ mod1.(t .- b, p)
end

function lkad_hoursbefore(p::Int, t::Int, b::UnitRange{Int})::Vector{Int}
    # if t > p, simply return the period leading up to T
    if t > p
        return t .- b
    elseif t <= p
        return mod1.(t .- b, p)
    end
end


function save_hdf5(savepath, Tend, data_str, data_array)
    println("Saving ", data_str, " to HDF5 file")
    # Create the HDF5 file
    h5 = HDF5.h5open(joinpath(savepath, data_str * ".h5"), "w")
    # Write the wind_scen_array to the HDF5 file
    for i in 1:Tend
        dsetname = data_str * "_$i"
        HDF5.write(h5, dsetname, data_array[i])
    end
    # Close the HDF5 file
    close(h5)
end


case = dirname(@__FILE__)
folder_name_parts = split(case, "\\")
case_name = folder_name_parts[end]

# get model type XXX make automatic, update for PF vs LAC
model_type = "pf"


### Run.jl
test_dictionary = Dict(
    "test_scenario_path" => 1,
    "test_scenario_lookahead_path" => "value2",
    "test_prices_scen_path" => "value3"
)

test_scenario_path = test_dictionary["test_scenario_path"]
test_scenario_lookahead_path = test_dictionary["test_scenario_lookahead_path"]


# case = dirname(@__FILE__)
optimizer = Gurobi.Optimizer

### print_genx_version()
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
println(ascii_art)

genx_settings = get_settings_path(case, "genx_settings.yml") # Settings YAML file path
writeoutput_settings = get_settings_path(case, "output_settings.yml") # Write-output settings YAML file path
setup = configure_settings(genx_settings, writeoutput_settings) # setup dictionary stores settings and GenX-specific parameters

### run_genx_case_simple
settings_path = get_settings_path(case)

### Cluster time series inputs if necessary and if specified by the user
if setup["TimeDomainReduction"] == 1
    TDRpath = joinpath(case, setup["TimeDomainReductionFolder"])
    system_path = joinpath(case, setup["SystemFolder"])
    prevent_doubled_timedomainreduction(system_path)
    if !time_domain_reduced_files_exist(TDRpath)
        println("Clustering Time Series Data (Grouped)...")
        cluster_inputs(case, settings_path, setup)
    else
        println("Time Series Data Already Clustered.")
    end
end

### Configure solver
println("Configuring Solver")
solver_name = lowercase(get(setup, "Solver", ""))
OPTIMIZER = configure_solver(settings_path, optimizer; solver_name=solver_name)

#### Running a case

### Load inputs
println("Loading Inputs")
inputs = load_inputs(setup, case)

### update omega and fuel_costs and c_start for lookahead length
# XXX I'm not sure there is a better way to do this other than ex-post loading the inputs using GenX
# add 48 rows to omega 
inputs["omega"] = vcat(inputs["omega"], ones(48))
# for each fuel in fuel_costs, add 48 rows of last value
for fuel in keys(inputs["fuel_costs"])
    inputs["fuel_costs"][fuel] = vcat(inputs["fuel_costs"][fuel], repeat([inputs["fuel_costs"][fuel][end]], 48))
end
# for each generator, add 48 rows of last value
inputs["C_Start"] = hcat(inputs["C_Start"], repeat(inputs["C_Start"][:, end:end], 1, 48))

# Should be defined in module, but doesn't get read in these run files???
ModelScalingFactor = 1e+3; 


#=======================================================================
Set Case Specific Parameters
=======================================================================#
folder_name_parts = split(case, "\\")
case_name = folder_name_parts[end]



### Load in Scenario Generation information
scen_generator = scenario_generator_init()
# Save the objects from scen_generator into individual variables
unique_forecast_times = scen_generator["unique_forecast_times"]
unique_issue_times = scen_generator["unique_issue_times"]
start_date = scen_generator["start_date"]
corr_forecast_issue_times = scen_generator["corr_forecast_issue_times"]
forecast_scenario_length = scen_generator["forecast_scenario_length"]
number_of_scenarios = scen_generator["number_of_scenarios"]
solar_model_data = scen_generator["solar_model_data"]
M_load = scen_generator["M_load"]
M_solar = scen_generator["M_solar"]
M_wind = scen_generator["M_wind"]
lp_solar = scen_generator["lp_solar"]
load_marginals_by_issue = scen_generator["load_marginals_by_issue"]
solar_marginals_by_issue = scen_generator["solar_marginals_by_issue"]
wind_marginals_by_issue = scen_generator["wind_marginals_by_issue"]
load_landing_probabilities = scen_generator["load_landing_probabilities"]
solar_landing_probabilities = scen_generator["solar_landing_probabilities"]
wind_landing_probabilities = scen_generator["wind_landing_probabilities"]
load_actual_avg = scen_generator["load_actual_avg"]
solar_actual_avg = scen_generator["solar_actual_avg"]
wind_actual_avg = scen_generator["wind_actual_avg"]
solar_well_defined_cols = scen_generator["solar_well_defined_cols"]
solar_issue_decn_time_matrix = scen_generator["solar_issue_decn_time_matrix"]
load_actual_avg_GW = scen_generator["load_actual_avg_GW"]
solar_actual_avg_cf = scen_generator["solar_actual_avg_cf"]
wind_actual_avg_cf = scen_generator["wind_actual_avg_cf"]
decision_mdl_lkd_length = scen_generator["decision_mdl_lkd_length"]
max_solar_actual = scen_generator["max_solar_actual"];
max_wind_actual = scen_generator["max_wind_actual"];
start_date = scen_generator["start_date"];
#=======================================================================
DEFINE INDICES, DATETIMES, ISSUE SETS FOR NORTA SCENARIOS AND STOCASTIC SIM
=======================================================================#
# Set the date and time for the forecasts
start_index = findfirst(isequal(start_date), unique_forecast_times)

Tstart = start_index;
Tend = decision_mdl_lkd_length - 50; # should be -48 probablys
# Tend = 10;

# Tstart = 1 - 24 # need data for the hours before 1 then!
# Tmax = 8760 + 48

rh_len = forecast_scenario_length # scenario_length

print("model type is ", model_type)
if model_type == "pf"
    R = 1
elseif model_type == "dlac-p" || model_type == "dlac-i" || model_type == "slac"
    R = range(Tstart, Tend, step=1) # K = 1:1:Tmax-48 # scenario_length
else
    error("Model type not recognized")
end

# Rh as a dictionary
Rh = Dict()


# INTERIOR_SUBPERIODS
for r in R
    Rh[r] = r:1:r+forecast_scenario_length # scenario_length
end

# # Rh as an array
# Rharray = []

# for r in R
#     push!(Rharray, r:1:r+48)
# end

# initialize array for saving history of time indexes at which up down decisions are still relevant
RhHistory = [] # XXX not necessary for first attempt ignoring changing the intertemporal constraints


#### define warm start parameters


#### initialize elements that are saved across loops



gen = inputs["RESOURCES"]
zones = zone_id.(gen)
regions = region.(gen)
clusters = cluster.(gen)
rid = resource_id.(gen)
resource_names = inputs["RESOURCE_NAMES"]

COMMIT = inputs["COMMIT"]
THERM_COMMIT = inputs["THERM_COMMIT"] # can be outside of loop, never changes
STOR_LIST = inputs["STOR_ALL"]
STOR_ALL = inputs["STOR_ALL"]
VRE_LIST = inputs["VRE"]
if setup["OperationalReserves"] >= 1
    RSV = inputs["RSV"]# Generators contributing to operating reserves
    REG = inputs["REG"]     # Generators contributing to regulation 
end

solar_ids = [gen[y].id for y in VRE_LIST if gen[y].solar == 1]
wind_ids = [gen[y].id for y in VRE_LIST if gen[y].wind == 1]

# WIND_LIST = inputs["WIND"]
# SOLAR_LIST = inputs["SOLAR"]
# ZONES = ???
num_gen = inputs["G"]
G = inputs["G"] 
Z = inputs["Z"]     # Number of zones

# initialize dictionaries for saved variables, sVariable, could either be nested dictionaries or matrix arrays. XXX

var_strings = ["P", "RSV", "REG", "NSE", "COMMIT", "START", "SHUT", "CHARGE", "S"]
pri_strings = ["PowerBalance", "Reg", "RsvReq"]

var_dict = Dict(var_strings[i] => Dict() for i in 1:length(var_strings))
pri_dict = Dict()

# sP = Dict()
pgen_dp = zeros(num_gen, Tend)
# sRSV = Dict()
rsv_dp = zeros(num_gen, Tend);
# sREG = Dict()
reg_dp = zeros(num_gen, Tend);

# sNSE = Dict()
nse_dp = zeros(Z, Tend);
unmet_rsv_dp = zeros(Z,Tend);

# sCOMMIT = Dict()
commit_dp = zeros(num_gen, Tend);
# sSTART = Dict()
start_dp = zeros(num_gen, Tend);
# sSHUT = Dict()
shut_dp = zeros(num_gen, Tend);


# sCHARGE = Dict()
charge_dp = zeros(num_gen, Tend);
# sS = Dict()
s_dp = zeros(num_gen, Tend);

# initialize object to save prices
elec_prices = zeros(Z, Tend)
reg_prices = zeros(Z, Tend)
rsv_prices  = zeros(Z, Tend)

for price_key in pri_strings
    pri_dict[price_key] = zeros(Z, Tend)
end

max_discharge_const_duals = Array{Any}(undef, Tend)
max_charge_const_duals = Array{Any}(undef, Tend)
soc_link_duals = Array{Any}(undef, Tend)
soc_int_duals = Array{Any}(undef, Tend)

# initialize constant costs
var_om_cost_per_gen = [var_om_cost_per_mwh(gen[y]) for y in 1:G]
var_om_cost_in_per_gen = [y in STOR_ALL ? var_om_cost_per_mwh_in(gen[y]) : 0 for y in 1:G]
fixed_om_cost_per_gen = [fixed_om_cost_per_mwyr(gen[y]) for y in 1:G]

existing_cap_mw_per_gen = [existing_cap_mw(gen[y]) for y in 1:G]
existing_cap_mwh_per_gen = [existing_cap_mwh(gen[y]) for y in 1:G]
generator_name_per_gen = [gen[y].resource for y in 1:G]

inv_cost_per_mwyr_per_gen = [inv_cost_per_mwyr(gen[y]) for y in 1:G]
inv_cost_per_mwhyr_per_gen = [inv_cost_per_mwhyr(gen[y]) for y in 1:G]

fuel_costs = inputs["fuel_costs"]

fuel_cost_per_mmbtu = [fuel_costs[fuel(gen[y])][:] for y in 1:G]
fuel_cost_per_mmbtu = transpose(hcat(fuel_cost_per_mmbtu...))



#=======================================================================
Define Revenues and Costs to save from Simulation
=======================================================================#

### hourly components

# per generator components
energy_revs_dp = zeros(num_gen, Tend)
reg_revs_dp = zeros(num_gen, Tend)
rsv_revs_dp = zeros(num_gen, Tend)
var_om_costs_dp = zeros(num_gen, Tend)
fuel_costs_dp = zeros(num_gen, Tend)
start_costs_dp = zeros(num_gen, Tend)
charge_costs_dp = zeros(num_gen, Tend)

# per zone components - WELFARE 
nse_cost = zeros(Z, Tend)
unmet_rsv_cost = zeros(Z, Tend)

# Create a NetRevenue dataframe
dfNetRevenue = DataFrame(region = regions,
Resource = inputs["RESOURCE_NAMES"],
zone = zones,
Cluster = clusters,
R_ID = rid)

#=======================================================================
Define CEM wrap around initial conditions
=======================================================================#
### set up processing information required to get correct wraparound info
# initialize number of units that are started / on
# gen_up_lengths = [y in THERM_COMMIT ? gen[y].existing_cap_mw / gen[y].cap_size * 0.75 : 0.0 for y in 1:num_gen]


# define CEM path
cem_path = joinpath(case, "..", "..", "..", "GenX.jl", "research_systems", case_name)
cem_results_path = joinpath(cem_path, "results")

# load commit, commit, commit_dp
cem_commit_raw = CSV.read(joinpath(cem_results_path, "commit.csv"), DataFrame)
# Remove the first two rows and reset the index for `cem_commit`
cem_commit = cem_commit_raw[3:end,:] 

# load startup, start, start_dp
cem_start_raw = CSV.read(joinpath(cem_results_path, "start.csv"), DataFrame)
# Remove the first two rows and reset the index for `cem_start`
cem_start = cem_start_raw[3:end,:]

# load shut down, shutdown, shut_dp
cem_shut_raw = CSV.read(joinpath(cem_results_path, "shutdown.csv"), DataFrame)
# Remove the first two rows and reset the index for `cem_shut`
cem_shut = cem_shut_raw[3:end,:]

# load shut down, shutdown, shut_dp
cem_shut_raw = CSV.read(joinpath(cem_results_path, "shutdown.csv"), DataFrame)
# Remove the first two rows and reset the index for `cem_shut`
cem_shut = cem_shut_raw[3:end,:]

# load state of charge, storage, s_dp
cem_soc_raw = CSV.read(joinpath(cem_results_path, "storage.csv"), DataFrame)
# Remove the first two rows and reset the index for `cem_soc`
cem_soc = cem_soc_raw[3:end, :]

for col in names(cem_soc)
    if eltype(cem_soc[!, col]) == Float64
        cem_soc[!, col] .= cem_soc[!, col] ./ ModelScalingFactor
    end
end

# load dispatch, power, pgen_dp
cem_dispatch_raw = CSV.read(joinpath(cem_results_path, "power.csv"), DataFrame)
# Remove the first two rows and reset the index for `cem_dispatch`
cem_dispatch = cem_dispatch_raw[3:end,:]

for col in names(cem_dispatch)
    if eltype(cem_dispatch[!, col]) == Float64
        cem_dispatch[!, col] .= cem_dispatch[!, col] ./ ModelScalingFactor
    end
end


#=======================================================================
Define Inputs for Rolling Horizon Simulation
=======================================================================#
rhinputs = deepcopy(inputs) # this could be outside of the loop


# Initialize Locked Scenario Path Information
scen_path = Array{Any}(undef, Tend)

Random.seed!(12345);
# intialize normal distribution
normal_dist = Normal(0,1);

# intialize the start date
date = deepcopy(start_date)


println("Generating the SPCM Optimization")
# GENERATE SPCM MODEL
# time_elapsed = @elapsed EP = generate_spcm_model(setup, inputs, OPTIMIZER)
println("Time elapsed for model building is")
# println(time_elapsed)

# #=======================================================================
# Debug at specified time in rolling horizon model
# =======================================================================#

# pgen_dp = Matrix(DataFrame(CSV.File(savepath * "unit_pgen.csv", header = false))) / ModelScalingFactor
# commit_dp = Matrix(DataFrame(CSV.File(savepath * "unit_commit.csv", header = false)))
# start_dp = Matrix(DataFrame(CSV.File(savepath * "unit_start.csv", header = false)))
# shut_dp = Matrix(DataFrame(CSV.File(savepath * "unit_shut.csv", header = false)))
# s_dp = Matrix(DataFrame(CSV.File(savepath * "unit_state_of_charge.csv", header = false))) / ModelScalingFactor
# charge_dp = Matrix(DataFrame(CSV.File(savepath * "unit_charge.csv", header = false))) / ModelScalingFactor

#=======================================================================
Set 
=======================================================================#
if test_scenario_path == 1
    load_scen_path = zeros(Tend)
    solar_scen_path = zeros(Tend) 
    wind_scen_path = zeros(Tend) 
end

if test_scenario_lookahead_path == 1
    # load_scen_array = Array{Any}(undef, Tend) # [Matrix{Any}(undef, number_of_scenarios, scenario_length) for _ in 1:Tend] # 
    # solar_scen_array = Array{Any}(undef, Tend) # [Matrix{Any}(undef, number_of_scenarios, scenario_length) for _ in 1:Tend] #
    # wind_scen_array = Array{Any}(undef, Tend) # [Matrix{Any}(undef, number_of_scenarios, scenario_length) for _ in 1:Tend] # 
end

### Printing Testing data 
###========================================================================
# create testing  folder in results path if it doesn't exist
testing_results_folder = joinpath(results_folder, "testing")
if !isdir(testing_results_folder)
    println("Creating testing results folder at: ", testing_results_folder)
    mkpath(testing_results_folder)
end

solar_capacity_gw = [gen[y].existing_cap_mw for y in VRE_LIST if gen[y].solar == 1]
wind_capacity_gw = [gen[y].existing_cap_mw for y in VRE_LIST if gen[y].wind == 1]

# convert the solar and wind paths to capacity factors based on max actuals
solar_scen_path_cf = solar_scen_path ./ solar_capacity_gw
wind_scen_path_cf = wind_scen_path ./ wind_capacity_gw

solar_actual_avg_cf_dec_ln = solar_actual_avg_cf[1:Tend]
wind_actual_avg_cf_dec_ln = wind_actual_avg_cf[1:Tend]
load_actual_avg_GW_dec_ln = load_actual_avg_GW[1:Tend]

# append solar, wind cf and load actuals to dataframe
solar_data = DataFrame("scenario path [cf]" => solar_scen_path_cf, "solar actuals [cf]" => solar_actual_avg_cf_dec_ln)
wind_data = DataFrame("scenario path [cf]" => wind_scen_path_cf, "wind actuals [cf]" => wind_actual_avg_cf_dec_ln)
load_data = DataFrame("scenario path [GW]" => load_scen_path, "load actuals [GW]" => load_actual_avg_GW_dec_ln)

if test_scenario_path == 1
    CSV.write(joinpath(testing_results_folder, "solar_scen_path.csv"), solar_data)
    CSV.write(joinpath(testing_results_folder, "wind_scen_path.csv"), wind_data)
    CSV.write(joinpath(testing_results_folder, "load_scen_path.csv"), load_data)
end
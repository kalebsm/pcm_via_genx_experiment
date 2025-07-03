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

function get_settings_path(case::AbstractString)
    return joinpath(case, "settings")
end

function get_settings_path(case::AbstractString, filename::AbstractString)
    return joinpath(get_settings_path(case), filename)
end

function get_default_output_folder(case::AbstractString)
    return joinpath(case, "results")
end

case = dirname(@__FILE__)
folder_name_parts = split(case, "\\")
case_name = folder_name_parts[end]


# case = dirname(@__FILE__)
optimizer = Gurobi.Optimizer

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

# Should be defined in module, but doesn't get read in these run files???
ModelScalingFactor = 1e+3; 


#=======================================================================
Set Case Specific Parameters
=======================================================================#
folder_name_parts = split(case, "\\")
case_name = folder_name_parts[end]
model_type = "pf"


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

println("model type is", model_type)
if model_type == "pf"
    R = 1
elseif model_type == "dlac-p" || model_type == "dlac-i" || model_type == "slac"
    R = range(Tstart, Tend, step=1) # K = 1:1:Tmax-48 # scenario_length
else
    error("Model type not recognized")
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


# #### Lookahead Loop
for r in R
    # r = 1
    # r = 7655
    global decision_date = start_date + Dates.Hour(r - 1)

    #=======================================================================
    DEFINE DATETIMES AND FORECAST VS ACTUAL VS MODEL TIMES OR HOURS
    =======================================================================#

    horizon_start_index = findfirst(isequal(decision_date), unique_forecast_times);

    if model_type == "pf"
        # current_issue = corr_forecast_issue_times[horizon_start_index, :issue_time];
        current_issue = nothing

        issue_index = nothing
        # find next issue time and compare to start date
        # next_issue = unique_issue_times[issue_index + 1];
        # define active issues (XXX workshop name) as set of forecasts that are available for the hours to use for lookahead
        active_issues = nothing
        # get the indices of the forecasts of the active issue times
        current_forecast_indices = nothing

        # get the actual forecast times of the current forecast indices
        current_forecast_times = nothing
        # get the forecast times that are after the start_data
        forecast_times_start_incl = nothing
        # calculate the length of the forecast times after the start date
        policy_model_length = Tend
        policy_forecast_length = nothing

        # extract the actuals from the last 48 - model_forecast_length
        # actuals_length = 48 - policy_model_length + 1;
        
        policy_actuals_length = Tend

        lookahead_decision_hours = nothing
    elseif model_type == "dlac-p" || model_type == "dlac-i" || model_type == "slac"
        current_issue = corr_forecast_issue_times[horizon_start_index, :issue_time];

        issue_index = findall(x -> x == current_issue, unique_issue_times)[1];
        # find next issue time and compare to start date
        # next_issue = unique_issue_times[issue_index + 1];
        # define active issues (XXX workshop name) as set of forecasts that are available for the hours to use for lookahead
        active_issues = [current_issue];
        # get the indices of the forecasts of the active issue times
        current_forecast_indices = findall(x -> x in active_issues, corr_forecast_issue_times[!,:issue_time]);
    
        # get the actual forecast times of the current forecast indices
        current_forecast_times = corr_forecast_issue_times[current_forecast_indices, :forecast_time];
        # get the forecast times that are after the start_data
        forecast_times_start_incl = filter(x -> x >= decision_date, current_forecast_times);
        # calculate the length of the forecast times after the start date
        policy_model_length = length(forecast_times_start_incl);
        policy_lookahead_length = policy_model_length - 1; # minus one always for the existing lookahead...
    
        # extract the actuals from the last 48 - model_forecast_length
        # actuals_length = 48 - policy_model_length + 1;
        policy_actuals_length = forecast_scenario_length - policy_lookahead_length;

        lookahead_decision_hours = collect(policy_actuals_length:forecast_scenario_length);
    else
        error("Model type not recognized")
    end

    # define scenarios for pf and dlac-p vs dlac-i and slac
    if model_type == "pf" || model_type == "dlac-p"
        # #=======================================================================
        # GENERATE SCENARIOS FOR LOAD, SOLAR, AND WIND
        # =======================================================================#
        # Y_load =  generate_norta_scenarios(number_of_scenarios, scenario_length, issue_index, horizon_start_index, 
        #                                     active_issues, corr_forecast_issue_times, normal_dist,  
        #                                     M_load, load_marginals_by_issue, load_landing_probabilities, false)

        # Y_solar = generate_norta_scenarios(number_of_scenarios, scenario_length, issue_index, horizon_start_index, 
        #                                     active_issues, corr_forecast_issue_times, normal_dist,  
        #                                     M_solar, solar_marginals_by_issue, solar_landing_probabilities, true)

        # Y_wind = generate_norta_scenarios(number_of_scenarios, scenario_length, issue_index, horizon_start_index, 
        #                                     active_issues, corr_forecast_issue_times, normal_dist,  
        #                                     M_wind, wind_marginals_by_issue, wind_landing_probabilities, false)

        # # convert MWh to GWh for load
        # Y_load_GWh = Y_load ./ ModelScalingFactor;
        # # normalize the solar and wind scenarios to capacity factors based on the maximum actuals
        # Y_solar_cf = Y_solar ./ max_solar_actual;
        # Y_wind_cf = Y_wind ./ max_wind_actual;

        scen_data = DataFrame() # for insertcols each new scen in loop

        # Concatenate the columns of Y_load, Y_solar, and Y_wind
        # for i in 1:size(Y_load, 1)
        new_solar = solar_actual_avg_cf[r:r + policy_model_length - 1];
        new_wind = wind_actual_avg_cf[r:r + policy_model_length - 1];
        new_load = load_actual_avg_GW[r:r + policy_model_length - 1];
        new_scen = DataFrame(S = new_solar, W = new_wind, L = new_load)
        insertcols!(scen_data, :S => new_solar, :W => new_wind, :L => new_load, makeunique=true)
        # end
        # print(master_scen)
    elseif model_type == "dlac-i" || model_type == "slac"
        load_current_marginals = load_marginals_by_issue[issue_index];
        solar_current_marginals = solar_marginals_by_issue[issue_index];
        wind_current_marginals = wind_marginals_by_issue[issue_index];
        #=======================================================================
        GENERATE SCENARIOS FOR LOAD, SOLAR, AND WIND
        =======================================================================#
        Y_load =  generate_norta_scenarios(number_of_scenarios, forecast_scenario_length, decision_date, horizon_start_index, 
                                            active_issues, corr_forecast_issue_times, normal_dist,  
                                            M_load, load_current_marginals, load_landing_probabilities,
                                            solar_well_defined_cols, false)
    
        Y_solar = generate_norta_scenarios(number_of_scenarios, forecast_scenario_length, decision_date, horizon_start_index, 
                                            active_issues, corr_forecast_issue_times, normal_dist,  
                                            M_solar, solar_current_marginals, solar_landing_probabilities,
                                            solar_well_defined_cols, true)
    
        Y_wind = generate_norta_scenarios(number_of_scenarios, forecast_scenario_length, decision_date, horizon_start_index, 
                                            active_issues, corr_forecast_issue_times, normal_dist,  
                                            M_wind, wind_current_marginals, wind_landing_probabilities,
                                            solar_well_defined_cols, false)
    
        # array_Y_load[r] = Y_load
    
        # convert MWh to GWh for load
        Y_load_GWh = Y_load ./ ModelScalingFactor;
        # normalize the solar and wind scenarios to capacity factors based on the maximum actuals
        Y_solar_cf = Y_solar ./ max_solar_actual;
        Y_wind_cf = Y_wind ./ max_wind_actual;

        scen_data = DataFrame() # for insertcols each new scen in loop

        if model_type == "dlac-i"
            # average across all scenarios for each time step
            load_scenario_avg_GW = mean(Y_load_GWh, dims = 1);
            solar_scenario_avg_cf = mean(Y_solar_cf, dims = 1);
            wind_scenario_avg_cf = mean(Y_wind_cf, dims = 1);

            new_solar = vec(solar_scenario_avg_cf)
            new_wind = vec(wind_scenario_avg_cf)
            new_load = vec(load_scenario_avg_GW)
            new_scen = DataFrame(S = new_solar, W = new_wind, L = new_load)
            insertcols!(scen_data, :S => new_solar, :W => new_wind, :L => new_load, makeunique=true)

        elseif model_type == "slac"
            for i in 1:size(Y_load, 1)
                new_solar = Y_solar_cf[i,:];
                new_wind = Y_wind_cf[i,:];
                new_load = Y_load_GWh[i,:];
                new_scen = DataFrame(S = new_solar, W = new_wind, L = new_load)
                insertcols!(scen_data, :S => new_solar, :W => new_wind, :L => new_load, makeunique=true)
            end
        end

    end

    ### processing scenarios in columned sequential form
    # check that number of columns is divisible by 3
    no_col = size(scen_data)[2]
    if mod(no_col,3) == 0
        W = round(Int, size(scen_data)[2] / 3)
    else
        print("Scenario input does not have the required dimensions")
    end
end
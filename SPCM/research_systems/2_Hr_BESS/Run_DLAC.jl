"""
GenX: An Configurable Capacity Expansion Model
Copyright (C) 2021,  Massachusetts Institute of Technology
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
A complete copy of the GNU General Public License v2 (GPLv2) is available
in LICENSE.txt.  Users uncompressing this from an archive may not have
received this license file.  If not, see <http://www.gnu.org/licenses/>.

XXX Insert description of Rolling Horizon simulation model

"""

using GenX
using Gurobi
# using YAML
# using BenchmarkTools
# using JuMP
# using DataFrames
# using CSV
# using DelimitedFiles
# using Distributions
# using Random
# # using HDF5
# using Plots # does not work for some reason
# using TimeZones
# using TSFrames
# using TableinearSolve
# using LinearAlgebra
# using Dates
# using DelimitedFiles
# using StatsBase
# using BenchmarkTools


cd(dirname(@__FILE__))
settings_path = joinpath(pwd(), "Settings")
println("settings_path", settings_path)
src_path = "../../../src/"

inpath = pwd()
# using L

function hoursbefore(p::Int, t::Int, b::UnitRange{Int})::Vector{Int}
	period = div(t - 1, p)
	return period * p .+ mod1.(t .- b, p)
end


function rh_hoursbefore(p::Int, t::Int, b::UnitRange{Int})::Vector{Int}

    return mod1.(t .- b, p)
end

# function save_hdf5(savepath, Tend, data_str, data_array)
#     println("Saving ", data_str, " to HDF5 file")
#     # Create the HDF5 file
#     h5 = HDF5.h5open(joinpath(savepath, data_str * ".h5"), "w")
#     # Write the wind_scen_array to the HDF5 file
#     for i in 1:Tend
#         dsetname = data_str * "_$i"
#         HDF5.write(h5, dsetname, data_array[i])
#     end
#     # Close the HDF5 file
#     close(h5)
# end





# include("Hugos_function.jl")
# include("scenario_generator.jl")
# include(joinpath("GenX_SLM", src_path, "norta_scenarios", "fct_bind_historical_forecast.jl"));
# include(joinpath("GenX_SLM", src_path, "norta_scenarios", "fct_compute_hourly_average_actuals.jl"));
# include(joinpath("GenX_SLM", src_path, "norta_scenarios", "fct_compute_landing_probability.jl"));
# include(joinpath("GenX_SLM", src_path, "norta_scenarios", "fct_convert_hours_2018.jl"));
# include(joinpath("GenX_SLM", src_path, "norta_scenarios", "fct_convert_ISO_standard.jl"));
# include(joinpath("GenX_SLM", src_path, "norta_scenarios", "fct_convert_land_prob_to_data.jl"));
# include(joinpath("GenX_SLM", src_path, "norta_scenarios", "fct_generate_probability_scenarios.jl"));
# # include(joinpath("GenX_SLM", src_path, "norta_scenarios", "fct_getplots.jl"));
# # include(joinpath("GenX_SLM", src_path, "norta_scenarios", "fct_plot_correlation_heatmap.jl"));
# # include(joinpath("GenX_SLM", src_path, "norta_scenarios", "fct_plot_historical_landing.jl"));
# include(joinpath("GenX_SLM", src_path, "norta_scenarios", "fct_plot_historical_synthetic_autocorrelation.jl"));
# include(joinpath("GenX_SLM", src_path, "norta_scenarios", "fct_plot_correlogram_landing_probability.jl"));
# include(joinpath("GenX_SLM", src_path, "norta_scenarios", "fct_plot_scenarios_and_actual.jl"));
# include(joinpath("GenX_SLM", src_path, "norta_scenarios", "fct_read_h5_file.jl"));
# include(joinpath("GenX_SLM", src_path, "norta_scenarios", "fct_read_input_file.jl"));
# include(joinpath("GenX_SLM", src_path, "norta_scenarios", "fct_transform_landing_probability.jl"));
# include(joinpath("GenX_SLM", src_path, "norta_scenarios", "fct_write_percentiles.jl"));
# include(joinpath("Genx_SLM", src_path, "norta_scenarios", "fct_generate_lower_triangular_correlation.jl"));
# include(joinpath("Genx_SLM", src_path, "norta_scenarios", "fct_generate_norta_scenarios.jl"));


genx_settings = joinpath(settings_path, "genx_settings.yml") #Settings YAML file path
mysetup = configure_settings(genx_settings,"") # mysetup dictionary stores settings and GenX-specific parameters

### Cluster time series inputs if necessary and if specified by the user
# TDRpath = joinpath(inpath, mysetup["TimeDomainReductionFolder"])
# if mysetup["TimeDomainReduction"] == 1
#     if (!isfile(TDRpath*"/Load_data.csv")) || (!isfile(TDRpath*"/Generators_variability.csv")) || (!isfile(TDRpath*"/Fuels_data.csv"))
#         println("Clustering Time Series Data...")
#         cluster_inputs(inpath, settings_path, mysetup)
#     else
#         println("Time Series Data Already Clustered.")
#     end
# end

# print("inpath: ", inpath)
### Configure solver
println("Configuring Solver")
OPTIMIZER = configure_solver(settings_path, Gurobi.Optimizer)

#### Running a case

### Load inputs
println("Loading Inputs")
# myinputs dictionary will store read-in data and computed parameters
myinputs = load_inputs(mysetup, inpath)

const ModelScalingFactor = 1e+3;



#=======================================================================
READ INPUT FILE
=======================================================================#
input_file_path = joinpath("GenX_SLM", src_path, "copulas.txt")

data_type,
scenario_length,
number_of_scenarios,
scenario_hour,
scenario_day,
scenario_month,
scenario_year,
read_locally,
historical_load,
forecast_load,
historical_solar,
forecast_da_solar,
forecast_2da_solar,
historical_wind,
forecastd_da_wind,
forecast_2da_wind,
write_percentile = read_input_file(input_file_path);

forecast_scenario_length = 48;

number_of_scenarios = 20;
scenario_hour = 0;
scenario_day = 1;
scenario_month = 1;
scenario_year = 2018;

# initialize the start date
start_date = DateTime(string(scenario_year) * "-" * string(scenario_month) * "-" * string(scenario_day) * "T" * string(scenario_hour));


#=======================================================================
READ INPUT DATA: ARPA-E PERFORM PROJECT H5 FILES
=======================================================================#
genxpath = "../../../GenX_SLM/"

# Function that reads the .h5 file and binds the time index and the actuals/fore-
# cast values into a single dataframe.

# Load data
load_actuals_raw = read_h5_file(joinpath(genxpath,"data", historical_load), "load");
load_forecast_raw = read_h5_file(joinpath(genxpath,"data", "ercot_BA_load_forecast_day_ahead_2018.h5"), "load", false);

# Solar data
solar_actuals_raw = read_h5_file(joinpath(genxpath,"data", "ercot_BA_solar_actuals_Existing_2018.h5"), "solar");
solar_forecast_dayahead_raw = read_h5_file(joinpath(genxpath,"data", "ercot_BA_solar_forecast_day_ahead_existing_2018.h5"), "solar", false);
solar_forecast_2dayahead_raw = read_h5_file(joinpath(genxpath,"data", "ercot_BA_solar_forecast_2_day_ahead_existing_2018.h5"), "solar", false);

# Wind data
wind_actuals_raw = read_h5_file(joinpath(genxpath,"data", "ercot_BA_wind_actuals_Existing_2018.h5"), "wind");
wind_forecast_dayahead_raw = read_h5_file(joinpath(genxpath,"data", "ercot_BA_wind_forecast_day_ahead_existing_2018.h5"), "wind", false);
wind_forecast_2dayahead_raw = read_h5_file(joinpath(genxpath,"data", "ercot_BA_wind_forecast_2_day_ahead_existing_2018.h5"), "wind", false);

#=======================================================================
Compute the hourly average for the actuals data
=======================================================================#
# Load
aux = compute_hourly_average_actuals(load_actuals_raw);
load_actual_avg_raw = DataFrame();
time_index = aux[:, :Index];
avg_actual = aux[:, :values_mean];
load_actual_avg_raw[!, :time_index] = time_index;
load_actual_avg_raw[!, :avg_actual] = avg_actual;

# Solar
aux = compute_hourly_average_actuals(solar_actuals_raw);
time_index = aux[:, :Index];
avg_actual = aux[:, :values_mean];
solar_actual_avg_raw = DataFrame();
solar_actual_avg_raw[!, :time_index] = time_index;
solar_actual_avg_raw[!, :avg_actual] = avg_actual;

# Wind
aux = compute_hourly_average_actuals(wind_actuals_raw);
time_index = aux[:, :Index];
avg_actual = aux[:, :values_mean];
wind_actual_avg_raw = DataFrame();
wind_actual_avg_raw[!, :time_index] = time_index;
wind_actual_avg_raw[!, :avg_actual] = avg_actual;

#=======================================================================
ADJUST THE TIME 
=======================================================================#
#= For the year of 2018, adjust the time to Texas' UTC (UTC-6 or UTC-5)
depending on daylight saving time =#

# Load data
load_actuals = convert_hours_2018(load_actuals_raw);
load_actual_avg = convert_hours_2018(load_actual_avg_raw);
load_forecast = convert_hours_2018(load_forecast_raw, false);
ahead_factor = repeat(["two", "one"], size(load_forecast, 1) ÷ 2)
load_forecast[!, :ahead_factor] = ahead_factor
load_forecast_dayahead = filter(:ahead_factor => ==("one"), load_forecast)
load_forecast_2dayahead = filter(:ahead_factor => ==("two"), load_forecast);

# Solar data
solar_actuals = convert_hours_2018(solar_actuals_raw);
solar_actual_avg = convert_hours_2018(solar_actual_avg_raw);
solar_forecast_dayahead = convert_hours_2018(solar_forecast_dayahead_raw, false);
solar_forecast_2dayahead = convert_hours_2018(solar_forecast_2dayahead_raw, false);


# Wind data
wind_actuals = convert_hours_2018(wind_actuals_raw);
wind_actual_avg = convert_hours_2018(wind_actual_avg_raw);
wind_forecast_dayahead = convert_hours_2018(wind_forecast_dayahead_raw, false);
wind_forecast_2dayahead = convert_hours_2018(wind_forecast_2dayahead_raw, false);



#=======================================================================
BIND HOURLY HISTORICAL DATA WITH FORECAST DATA
========================================================================#

load_data = bind_historical_forecast(false,
    load_actual_avg,
    load_forecast_dayahead,
    load_forecast_2dayahead);

solar_data = bind_historical_forecast(false,
    solar_actual_avg,
    solar_forecast_dayahead,
    solar_forecast_2dayahead);

wind_data = bind_historical_forecast(false,
    wind_actual_avg,
    wind_forecast_dayahead,
    wind_forecast_2dayahead);

#=======================================================================
Landing probability
=======================================================================#
landing_probability_load = compute_landing_probability(load_data);
landing_probability_solar = compute_landing_probability(solar_data);
landing_probability_wind = compute_landing_probability(wind_data);

#=======================================================================
ADJUST LANDING PROBABILITY DATAFRAME
=======================================================================#
lp_load = transform_landing_probability(landing_probability_load);
lp_solar = transform_landing_probability(landing_probability_solar);
lp_wind = transform_landing_probability(landing_probability_wind);

#=======================================================================
Determine length of Decision Problem and additinal inputs
=======================================================================#
all_same = true;

if all_same
    x = copy(wind_data);
    # Sort data by issue time
    sort!(x, :issue_time);
    # Group data by issue time and count occurences in every group
    df = combine(groupby(x, [:issue_time]), DataFrames.nrow => :count);
    # Filter data by count. Only keep groups with 48 entries
    df_filtered = filter(:count => ==(48), df);
    issue_times_interest = df_filtered[!, :issue_time];
    # find all forecast times for these issue times of interest
    subset_wind_data = filter(row -> row[:issue_time] in issue_times_interest, wind_data);
    subset_forecast_times = subset_wind_data[!, :forecast_time];
    unique_forecast_times = unique(subset_forecast_times);
    decision_model_length = length(unique_forecast_times);

    unique_issue_times = unique(subset_wind_data[!, :issue_time]);

    #define the actual landing probabilities as a vector
    left_lp_solar = transpose(lp_solar[:, 1:size(lp_load, 2) ÷ 2]);
    solar_landing_probabilities = vec(left_lp_solar);

    #define the actual landing probabilities as a vector
    left_lp_wind = transpose(lp_wind[:, 1:size(lp_load, 2) ÷ 2]);
    wind_landing_probabilities = vec(left_lp_wind);

    left_lp_load = transpose(lp_load[:, 1:size(lp_load, 2) ÷ 2]);
    load_landing_probabilities = vec(left_lp_load);

    # define the issue time tracking objects
    num_issue_times = length(unique_issue_times)
    issue_idcs = 1:num_issue_times

    # initialize an array of of issue times for saving the marginal probabilities
    load_marginals_by_issue = Array{DataFrame}(undef, num_issue_times)
    solar_marginals_by_issue = Array{DataFrame}(undef, num_issue_times)
    wind_marginals_by_issue = Array{DataFrame}(undef, num_issue_times)

    # loop through the wind_data and extract the marginal probability dataframes
    for i in issue_idcs
        current_issue = unique_issue_times[i]
        load_marginals_by_issue[i] = filter(row -> row[:issue_time] == current_issue, load_data)
        solar_marginals_by_issue[i] = filter(row -> row[:issue_time] == current_issue, solar_data)
        wind_marginals_by_issue[i] = filter(row -> row[:issue_time] == current_issue, wind_data)
    end

    # filter the load_data, solar_data, and wind_data by unique_issue_times
    load_data_upd = filter(row -> row[:issue_time] in unique_issue_times, load_data);
    solar_data_upd = filter(row -> row[:issue_time] in unique_issue_times, solar_data);
    wind_data_upd = filter(row -> row[:issue_time] in unique_issue_times, wind_data);

    corr_forecast_issue_times = wind_data_upd[:, [:issue_time, :forecast_time]];
else
    error("The data issue and forecast times are not the same for all of load, solar, and wind");
end

# for the actuals and DLAC calculations, determine capacity factors at correct model times
forecast2model_indices = findall(in(unique_forecast_times), load_actual_avg[!, :time_index])

max_solar_actual = maximum(solar_actual_avg[!, :avg_actual]);
max_wind_actual = maximum(wind_actual_avg[!, :avg_actual]);

load_actual_avg_GW = load_actual_avg[forecast2model_indices, :avg_actual] ./ ModelScalingFactor;
solar_actual_avg_cf = solar_actual_avg[forecast2model_indices, :avg_actual] ./ max_solar_actual;
wind_actual_avg_cf = wind_actual_avg[forecast2model_indices, :avg_actual] ./ max_wind_actual;

# actuals_df = DataFrame(load = load_actual_avg_GW .*ModelScalingFactor, solar = solar_actual_avg_cf, 
#                 wind = wind_actual_avg_cf)

# # print to csv
# CSV.write("actuals.csv", actuals_df)

#=======================================================================
Perform Cholesky Decomposition to get Lower Triangular Correlation Matrix
=======================================================================#
M_load = generate_lower_triangular_correlation(lp_load, issue_idcs, false);
M_solar, sunny_decision_hours = generate_lower_triangular_correlation(lp_solar, issue_idcs, true);
M_wind = generate_lower_triangular_correlation(lp_wind, issue_idcs, false);


#=======================================================================
DEFINE INDICES, DATETIMES, ISSUE SETS FOR NORTA SCENARIOS AND STOCASTIC SIM
=======================================================================#
# Set the date and time for the forecasts
start_index = findfirst(isequal(start_date), unique_forecast_times)

Tstart = start_index;
Tend = decision_model_length - 50;
# Tend = 10;

# Tstart = 1 - 24 # need data for the hours before 1 then!
# Tmax = 8760 + 48

rh_len = forecast_scenario_length # scenario_length

R = range(Tstart, Tend, step = 1) # K = 1:1:Tmax-48 # scenario_length

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






# TESTING XXX

# dfGen = myinputs["dfGen"]


# GENERATORS = dfGen[!,:R_ID]
num_gen = myinputs["G"]
GENERATORS = 1:num_gen
gen = myinputs["RESOURCES"]
THERM_COMMIT = myinputs["THERM_COMMIT"] # can be outside of loop, never changes
STOR_LIST = myinputs["STOR_ALL"]
VRE_LIST = myinputs["VRE"]


# initialize dictionaries for saved variables, sVariable, could either be nested dictionaries or matrix arrays. XXX

var_strings = ["P", "RSV", "REG", "NSE", "COMMIT", "START", "SHUT", "CHARGE", "S"]
pri_strings = ["PowerBalance", "Reg", "RsvReq"]

var_dict = Dict(var_strings[i] => Dict() for i in 1:length(var_strings))
pri_dict = Dict()

# sP = Dict()
pgen_dp = zeros(num_gen, Tend);
# sRSV = Dict()
rsv_dp = zeros(length(GENERATORS), Tend);
# sREG = Dict()
reg_dp = zeros(num_gen, Tend);

# sNSE = Dict()
nse_dp = zeros(1, Tend);
unmet_rsv_dp = zeros(1, Tend);

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
elec_prices = zeros(Tend)
reg_prices = zeros(Tend)
rsv_prices  = zeros(Tend)

for price_key in pri_strings
    pri_dict[price_key] = zeros(Tend)
end

max_discharge_const_duals = Array{Any}(undef, Tend)
max_charge_const_duals = Array{Any}(undef, Tend)
soc_link_duals = Array{Any}(undef, Tend)
soc_int_duals = Array{Any}(undef, Tend)

# # initialize the dictionaries
# for gen in GENERATORS
#     sP[gen] = zeros(Tend)
#     ### include Res and Reg variables XXX
#     sRSV[gen] = zeros(Tend)
#     sREG[gen] = zeros(Tend)
# end

# for zone in 1:1
#     sNSE[zone] = zeros(Tend)
# end

# # initialize saved full model variables to zeros 
# for gen in THERM_COMMIT
#     # sCOMMIT[gen] = []
#     # sSTART[gen] = []
#     # sSHUT[gen] = []
#     sCOMMIT[gen] = zeros(Tend)
#     sSTART[gen] = zeros(Tend)
#     sSHUT[gen] = zeros(Tend)
#     # sP[gen] = zeros(Tend)
# end


# for bess in STOR_LIST
#     # sPstor[bess] = zeros(Tend)
#     sCHARGE[bess] = zeros(Tend)
#     sS[bess] = zeros(Tend)
# end







#=======================================================================
Define Revenues and Costs to save from Simulation
=======================================================================#


### hourly components

# per generator components
energy_revs = zeros(num_gen, Tend)
reg_revs = zeros(num_gen, Tend)
rsv_revs = zeros(num_gen, Tend)
var_om_costs = zeros(num_gen, Tend)
fuel_costs = zeros(num_gen, Tend)
start_costs = zeros(num_gen, Tend)
charge_costs = zeros(num_gen, Tend)

# per zone components - WELFARE 
nse_cost = zeros(1, Tend)
unmet_rsv_cost = zeros(1, Tend)


#=======================================================================
Define Inputs for Rolling Horizon Simulation
=======================================================================#
rhinputs = deepcopy(myinputs) # this could be outside of the loop


# Initialize Locked Scenario Path Information
scen_path = Array{Any}(undef, Tend)

Random.seed!(12345);
# intialize normal distribution
d = Normal(0,1);

# intialize the start date
date = deepcopy(start_date)


#=======================================================================
Set Case Specific Parameters
=======================================================================#
folder_name = splitdir(@__FILE__)[end-1]
folder_name_parts = split(folder_name, "\\")
case_name = folder_name_parts[end]

test_name = "DLAC_" * case_name # this would be set by a .bat file

casepath = "C:/Users/ks885/Documents/aa_research/Modeling/genx_documentation/results/" * case_name * "/"

savepath = casepath * test_name * "/"

# # # for debugging: intialize values for debugging from start of failed model
# # THERM_COMMIT = myinputs["THERM_COMMIT"]
# # STOR_LIST = myinputs["STOR_ALL"]



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
Debug specific objects
=======================================================================#
# load_scen_array = Array{Any}(undef, Tend) # [Matrix{Any}(undef, number_of_scenarios, scenario_length) for _ in 1:Tend] # 
# solar_scen_array = Array{Any}(undef, Tend) # [Matrix{Any}(undef, number_of_scenarios, scenario_length) for _ in 1:Tend] #
# wind_scen_array = Array{Any}(undef, Tend) # [Matrix{Any}(undef, number_of_scenarios, scenario_length) for _ in 1:Tend] # 
prices_scen_array = Array{Any}(undef, Tend) # [Matrix{Any}(undef, number_of_scenarios, scenario_length) for _ in 1:Tend] #
# step_models_array = Array{Any}(undef, Tend)

# #### Lookahead Loop
for r in R
    # r = 1
    # r = 7655
    global decision_date = start_date + Dates.Hour(r - 1)

    #=======================================================================
    DEFINE DATETIMES AND FORECAST VS ACTUAL VS MODEL TIMES OR HOURS
    =======================================================================#

    horizon_start_index = findfirst(isequal(decision_date), unique_forecast_times);

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
    policy_forecast_length = policy_model_length - 1; # minus one always for the existing lookahead...

    # extract the actuals from the last 48 - model_forecast_length
    # actuals_length = 48 - policy_model_length + 1;
    policy_actuals_length = scenario_length - policy_forecast_length;

    lookahead_decision_hours = collect(policy_actuals_length:scenario_length);

    # #=======================================================================
    # GENERATE SCENARIOS FOR LOAD, SOLAR, AND WIND
    # =======================================================================#
    # Y_load =  generate_norta_scenarios(number_of_scenarios, scenario_length, issue_index, horizon_start_index, 
    #                                     active_issues, corr_forecast_issue_times, d,  
    #                                     M_load, load_marginals_by_issue, load_landing_probabilities, false)

    # Y_solar = generate_norta_scenarios(number_of_scenarios, scenario_length, issue_index, horizon_start_index, 
    #                                     active_issues, corr_forecast_issue_times, d,  
    #                                     M_solar, solar_marginals_by_issue, solar_landing_probabilities, true)

    # Y_wind = generate_norta_scenarios(number_of_scenarios, scenario_length, issue_index, horizon_start_index, 
    #                                     active_issues, corr_forecast_issue_times, d,  
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

    

    ### processing scenarios in columned sequential form
    # check that number of columns is divisible by 3
    no_col = size(scen_data)[2]
    if mod(no_col,3) == 0
        W = round(Int, size(scen_data)[2] / 3)
    else
        print("Scenario input does not have the required dimensions")
    end

    # determine length of lookahead available from scenario information
    # t_lookahead = policy_model_length
    # t_lookahead = 38 # XXX use this to check if the program runs with a different lookahead time

    st = r
    en = r + policy_model_length - 1 # for defining ranges from r to t_lookahead inclusive

    println()
    println("Begin Horizon ", st, " to ", en)

    # initialize and save dictionary for scenarios to look like pD and pP_Max of original GenX inputs
    pd_dict = Dict() # 1xx
    pp_max_dict = Dict() # 1xx

    for scen_idx in 1:(W) # should be W = 1 only
        # scen_idx = 1
        col_idx = 1 + (scen_idx - 1) * 3
        
        sol_column = col_idx #1xx
        wind_column = col_idx + 1 #1xx
        load_column = col_idx + 2 #1xx

        # reformat load data to match original pD data type
        load_df = scen_data[!, load_column] #1xx
        load_mt = reshape(load_df, (length(load_df), 1))

        # save into dictionaries
        pd_dict[scen_idx] = load_mt #1xx

        pp_max_dict[scen_idx] = zeros(num_gen, policy_model_length) 

        pp_max_dict[scen_idx][rhinputs["THERM_ALL"],:] .= ones(1, policy_model_length) 
        pp_max_dict[scen_idx][rhinputs["STOR_ALL"],:] .= ones(1, policy_model_length) 
        pp_max_dict[scen_idx][rhinputs["SOLAR"],:] .= transpose(scen_data[!, sol_column])
        pp_max_dict[scen_idx][rhinputs["WIND"],:] .= transpose(scen_data[!, wind_column]) 
    end
    # add dictionary of scenarios and number of scenarios to generate_model inputs
    rhinputs["W"] = W
    
    ### Scenario dependent information / begin scenario loop
    # rhinputs["pD"] = myinputs["pD"][st:en] # XXX THIS PROBABLY NEEDS TO BE CHANGED
    # rhinputs["pP_Max"] = myinputs["pP_Max"][:, st:en] # XXX THIS PROBABLY NEEDS TO BE CHANGED
    rhinputs["pD"] = pd_dict
    rhinputs["pP_Max"] = pp_max_dict

    ### Data inputs consistent across all scenarios
    rhinputs["omega"] = myinputs["omega"][st:en]

    rhinputs["C_Fuel_per_MWh"] = myinputs["C_Fuel_per_MWh"][:, st:en]

    rhinputs["C_Start"] = myinputs["C_Start"][:, st:en]

    for k in keys(myinputs["fuel_costs"])
        rhinputs["fuel_costs"][k] = myinputs["fuel_costs"][k][st:en]
    end

    ### Time components consistent across all scenarios
    # make distinction between first model time step and all other timesteps
    rhinputs["INTERIOR_SUBPERIODS"] = range(2, policy_model_length, step = 1) # always hard coded to be 2 to t_lookahead, and not r + 2 to r + t_lookahead
    INTERIOR_SUBPERIODS = rhinputs["INTERIOR_SUBPERIODS"]
    START_SUBPERIODS = 1:1 # always hardcoded to be 1

    # change the lookahead length based on avaialable capacity time series
    # rhinputs["T"] = 48
    rhinputs["T"] = policy_model_length # XXX update to select the current scenario length instead # XXX iterating per lookahead horizon

    rhinputs["hours_per_subperiod"] = policy_model_length # outside loop

    ### Generate model
    println("Generating the Optimization Model")
    # EP, STEP = generate_model(mysetup, rhinputs, OPTIMIZER)

	## Start pre-solve timer
	presolver_start_time = time()

    ### Stochastic Model ###
	# Generate Stochastic Energy Portfolio (STEP) Model
	STEP = Model(OPTIMIZER)

    ### Generate model without calling GenX
	G = rhinputs["G"]     # Number of resources (generators, storage, DR, and DERs)
	T = rhinputs["T"]     # Number of time steps
	Z = rhinputs["Z"]     # Number of zones
    W = rhinputs["W"]		#Number of scenarios
	SEG = rhinputs["SEG"] # Number of load curtailment segments

	# Introduce dummy variable fixed to zero to ensure that expressions like eTotalCap,
	# eTotalCapCharge, eTotalCapEnergy and eAvail_Trans_Cap all have a JuMP variable
	@variable(STEP, vZERO == 0);

	# # @expression(STEP, st_ePowerBalance[t=1:T, z=1:Z, w=1:W], zeros((T,1,W))) # pretty sure this is wrong
	@expression(STEP, st_ePowerBalance[t=1:T, z=1:Z, w=1:W], 0)

	# objective
	@expression(STEP, st_eObj, zeros((W,1)))

    ### is this necessary?
	@expression(STEP, st_eGenerationByZone[z=1:Z, t=1:T, w=1:W], 0) 

    ### add discharge.jl
	@variable(STEP, st_vP[y=1:G,t=1:T,w=1:W] >=0);

	# add the here and now decision variable
	@variable(STEP, st_hnP[y=1:G, t=[1]] >= 0);
	@constraint(STEP, DISC_nonantic[y=1:G, t=[1], w=1:W], st_vP[y,t,w] == st_hnP[y,t])

	# Variable costs of "generation" for resource "y" during hour "t" = variable O&M plus fuel cost
	@expression(STEP, st_eCVar_out[y=1:G,t=1:T, w=1:W], (rhinputs["omega"][t]*(gen[y].var_om_cost_per_mwh + rhinputs["C_Fuel_per_MWh"][y,t])*st_vP[y,t,w]))
	#@expression(EP, eCVar_out[y=1:G,t=1:T], (round(inputs["omega"][t]*(gen[y].var_om_cost_per_mwh+inputs["C_Fuel_per_MWh"][y,t]), digits=RD)*vP[y,t]))
	# Sum individual resource contributions to variable discharging costs to get total variable discharging costs
	@expression(STEP, st_eTotalCVarOutT[t=1:T, w=1:W], sum(st_eCVar_out[y,t,w] for y in 1:G))
	@expression(STEP, st_eTotalCVarOut[w=1:W], sum(st_eTotalCVarOutT[t,w] for t in 1:T))

	# # Add total variable discharging cost contribution to the objective function
	STEP[:st_eObj] += st_eTotalCVarOut

	# constraints
	@constraint(STEP, st_Max_vP[y=1:G, t=1:T, w=1:W], st_vP[y,t,w] <= rhinputs["RESOURCES"][y].existing_cap_mw)

    ### non_served_energy.jl
    # variables
	# Non-served energy/curtailed demand in the segment "s" at hour "t" in zone "z"
	@variable(STEP, st_vNSE[s=1:SEG,t=1:T,z=1:Z,w=1:W] >= 0);


	@variable(STEP, st_hnNSE[s=1:SEG,t=[1],z=1:Z] >= 0);
	@constraint(STEP, NSE_nonantic[s=1:SEG,t=[1],z=1:Z,w=1:W], st_vNSE[s,t,z,w] == st_hnNSE[s,t,z])

	#expressions

	# Objective Function Expressions

	# Cost of non-served energy/curtailed demand at hour "t" in zone "z"
	@expression(STEP, st_eCNSE[s=1:SEG,t=1:T,z=1:Z, w=1:W], (rhinputs["omega"][t]*rhinputs["pC_D_Curtail"][s]*st_vNSE[s,t,z,w]))

	# Sum individual demand segment contributions to non-served energy costs to get total non-served energy costs
	# Julia is fastest when summing over one row one column at a time
	@expression(STEP, st_eTotalCNSETS[t=1:T,z=1:Z, w=1:W], sum(st_eCNSE[s,t,z,w] for s in 1:SEG))
	@expression(STEP, st_eTotalCNSET[t=1:T, w=1:W], sum(st_eTotalCNSETS[t,z,w] for z in 1:Z))
	@expression(STEP, st_eTotalCNSE[w=1:W], sum(st_eTotalCNSET[t,w] for t in 1:T))

	# Add total cost contribution of non-served energy/curtailed demand to the objective function
	STEP[:st_eObj] += st_eTotalCNSE

	## Power Balance Expressions ##
	@expression(STEP, st_ePowerBalanceNse[t=1:T, z=1:Z, w=1:W],
	    sum(st_vNSE[s,t,z,w] for s=1:SEG))

	# Add non-served energy/curtailed demand contribution to power balance expression
	STEP[:st_ePowerBalance] += st_ePowerBalanceNse

	# Constratints

	@constraint(STEP, st_cNSEPerSeg[s=1:SEG, t=1:T, z=1:Z, w=1:W], st_vNSE[s,t,z,w] <= rhinputs["pMax_D_Curtail"][s]*rhinputs["pD"][w][t,z]) 

	@constraint(STEP, st_cMaxNSE[t=1:T, z=1:Z, w=1:W], sum(st_vNSE[s,t,z,w] for s=1:SEG) <= rhinputs["pD"][w][t,z])

    ### investment_discharge
    @expression(STEP, st_eExistingCap[y in 1:G], gen[y].existing_cap_mw)

    # is this needed?
    @expression(STEP, st_eTotalCap[y in 1:G], st_eExistingCap[y] + STEP[:vZERO])
	
    # individaul fixed costs
	@expression(STEP, st_eCFix[y in 1:G],
			dfGen[y,:Fixed_OM_Cost_per_MWyr]*st_eTotalCap[y]
	)
	# Sum individual resource contributions to fixed costs to get total fixed costs
	@expression(STEP, st_eTotalCFix, sum(STEP[:st_eCFix][y] for y in 1:G))
    # add fixed cost to objective
    STEP[:st_eObj] += st_eTotalCFix .* ones(W,1)

    ### Update Intertemporal Constraints commented out in generate_model files

    ### UCommit
    COMMIT = rhinputs["COMMIT"] # For not, thermal resources are the only ones eligible for Unit Committment

    ### Variables ###

	## Decision variables for unit commitment
	# commitment state variable
	@variable(STEP, st_vCOMMIT[y in COMMIT, t=1:T, w=1:W] >= 0)
	# startup event variable
	@variable(STEP, st_vSTART[y in COMMIT, t=1:T, w=1:W] >= 0)
	# shutdown event variable
	@variable(STEP, st_vSHUT[y in COMMIT, t=1:T, w=1:W] >= 0)

	### Here and Now Variables
	# commitment state variable
	@variable(STEP, st_hnCOMMIT[y in COMMIT, t=[1]] >= 0)
	# startup event variable
	@variable(STEP, st_hnSTART[y in COMMIT, t=[1]] >= 0)
	# shutdown event variable
	@variable(STEP, st_hnSHUT[y in COMMIT, t=[1]] >= 0)

	### Nonanticipativity constraints for each variable
	@constraint(STEP, COMMIT_nonantic[y in COMMIT, t=[1], w=1:W], st_vCOMMIT[y,t,w] == st_hnCOMMIT[y,t])
	@constraint(STEP, START_nonantic[y in COMMIT, t=[1], w=1:W], st_vSTART[y,t,w] == st_hnSTART[y,t])
	@constraint(STEP, SHUT_nonantic[y in COMMIT, t=[1], w=1:W], st_vSHUT[y,t,w] == st_hnSHUT[y,t])

	### Expressions ###

	## Objective Function Expressions ##

	# Startup costs of "generation" for resource "y" during hour "t"
	@expression(STEP, st_eCStart[y in COMMIT, t=1:T, w=1:W],(rhinputs["omega"][t]*rhinputs["C_Start"][y]*st_vSTART[y,t,w]))

	# Julia is fastest when summing over one row one column at a time
	@expression(STEP, st_eTotalCStartT[t=1:T, w=1:W], sum(st_eCStart[y,t,w] for y in COMMIT))
	@expression(STEP, st_eTotalCStart[w=1:W], sum(st_eTotalCStartT[t,w] for t=1:T))

	STEP[:st_eObj] += st_eTotalCStart

    ### reserves

    REG = rhinputs["REG"]
	RSV = rhinputs["RSV"]

	## Decision variables for reserves
	@variable(STEP, st_vREG[y in REG, t=1:T, w=1:W] >= 0) # Contribution to regulation (primary reserves), assumed to be symmetric (up & down directions equal)
	@variable(STEP, st_vRSV[y in RSV, t=1:T, w=1:W] >= 0) # Contribution to operating reserves (secondary reserves or contingency reserves); only model upward reserve requirements

	# Storage techs have two pairs of auxilary variables to reflect contributions to regulation and reserves
	# when charging and discharging (primary variable becomes equal to sum of these auxilary variables)
	@variable(STEP, st_vREG_discharge[y in intersect(rhinputs["STOR_ALL"], REG), t=1:T, w=1:W] >= 0) # Contribution to regulation (primary reserves) (mirrored variable used for storage devices)
	@variable(STEP, st_vRSV_discharge[y in intersect(rhinputs["STOR_ALL"], RSV), t=1:T, w=1:W] >= 0) # Contribution to operating reserves (secondary reserves) (mirrored variable used for storage devices)
	@variable(STEP, st_vREG_charge[y in intersect(rhinputs["STOR_ALL"], REG), t=1:T, w=1:W] >= 0) # Contribution to regulation (primary reserves) (mirrored variable used for storage devices)
	@variable(STEP, st_vRSV_charge[y in intersect(rhinputs["STOR_ALL"], RSV), t=1:T, w=1:W] >= 0) # Contribution to operating reserves (secondary reserves) (mirrored variable used for storage devices)

	@variable(STEP, st_vUNMET_RSV[t=1:T, w=1:W] >= 0) # Unmet operating reserves penalty/cost


	# ## Decision variables for reserves
	@variable(STEP, st_hnREG[y in REG, t=[1]] >= 0) # Contribution to regulation (primary reserves), assumed to be symmetric (up & down directions equal)
	@variable(STEP, st_hnRSV[y in RSV, t=[1]] >= 0) # Contribution to operating reserves (secondary reserves or contingency reserves); only model upward reserve requirements

	# # Storage techs have two pairs of auxilary variables to reflect contributions to regulation and reserves
	# # when charging and discharging (primary variable becomes equal to sum of these auxilary variables)
	@variable(STEP, st_hnREG_discharge[y in intersect(rhinputs["STOR_ALL"], REG), t=[1]] >= 0) # Contribution to regulation (primary reserves) (mirrored variable used for storage devices)
	@variable(STEP, st_hnRSV_discharge[y in intersect(rhinputs["STOR_ALL"], RSV), t=[1]] >= 0) # Contribution to operating reserves (secondary reserves) (mirrored variable used for storage devices)
	@variable(STEP, st_hnREG_charge[y in intersect(rhinputs["STOR_ALL"], REG), t=[1]] >= 0) # Contribution to regulation (primary reserves) (mirrored variable used for storage devices)
	@variable(STEP, st_hnRSV_charge[y in intersect(rhinputs["STOR_ALL"], RSV), t=[1]] >= 0) # Contribution to operating reserves (secondary reserves) (mirrored variable used for storage devices)

	@variable(STEP, st_hnUNMET_RSV[t=[1]] >= 0) # Unmet operating reserves penalty/cost

	# Nonanticipativity constraints for each variable type

	@constraint(STEP, REG_nonantic[y in REG, t=[1], w=1:W], st_vREG[y,t,w] == st_hnREG[y,t])
	@constraint(STEP, RSV_nonantic[y in REG, t=[1], w=1:W], st_vRSV[y,t,w] == st_hnRSV[y,t])

	@constraint(STEP, REG_discharge_nonantic[y in intersect(rhinputs["STOR_ALL"], REG), t=[1], w=1:W], st_vREG_discharge[y,t,w] == st_hnREG_discharge[y,t])
	@constraint(STEP, RSV_discharge_nonantic[y in intersect(rhinputs["STOR_ALL"], RSV), t=[1], w=1:W], st_vRSV_discharge[y,t,w] == st_hnRSV_discharge[y,t])
	@constraint(STEP, REG_charge_nonantic[y in intersect(rhinputs["STOR_ALL"], REG), t=[1], w=1:W], st_vREG_charge[y,t,w] == st_hnREG_charge[y,t])
	@constraint(STEP, RSV_charge_nonantic[y in intersect(rhinputs["STOR_ALL"], RSV), t=[1], w=1:W], st_vRSV_charge[y,t,w] == st_hnRSV_charge[y,t])

	@constraint(STEP, UNMET_RSV_nonantic[t=[1], w=1:W], st_vUNMET_RSV[t,w] == st_hnUNMET_RSV[t])

	### Expressions ###
	## Total system reserve expressions
	# Regulation requirements as a percentage of load and scheduled variable renewable energy production in each hour
	# Reg up and down requirements are symmetric
	@expression(STEP, st_eRegReq[t=1:T, w=1:W], rhinputs["pReg_Req_Load"]*sum(rhinputs["pD"][w][t,z] for z=1:Z) +
    rhinputs["pReg_Req_VRE"]*sum(rhinputs["pP_Max"][w][y,t]*STEP[:st_eTotalCap][y] for y in intersect(rhinputs["VRE"], rhinputs["MUST_RUN"])))
	# Operating reserve up / contingency reserve requirements as ˚a percentage of load and scheduled variable renewable energy production in each hour
	# and the largest single contingency (generator or transmission line outage)
	@expression(STEP, st_eRsvReq[t=1:T, w=1:W], rhinputs["pRsv_Req_Load"]*sum(rhinputs["pD"][w][t,z] for z=1:Z) +
    rhinputs["pRsv_Req_VRE"]*sum(rhinputs["pP_Max"][w][y,t]*STEP[:st_eTotalCap][y] for y in intersect(rhinputs["VRE"], rhinputs["MUST_RUN"])))


	## Objective Function Expressions ##

	# Penalty for unmet operating reserves
	@expression(STEP, st_eCRsvPen[t=1:T, w=1:W], rhinputs["omega"][t]*rhinputs["pC_Rsv_Penalty"]*st_vUNMET_RSV[t,w])
	@expression(STEP, st_eTotalCRsvPen[w=1:W], sum(st_eCRsvPen[t,w] for t=1:T) +
		sum(dfGen[y,:Reg_Cost]*st_vRSV[y,t,w] for y in RSV, t=1:T) +
		sum(dfGen[y,:Rsv_Cost]*st_vREG[y,t,w] for y in REG, t=1:T) )
		STEP[:st_eObj] += st_eTotalCRsvPen

	### Constraints ###

	## Total system reserve constraints
	# Regulation requirements as a percentage of load and scheduled variable renewable energy production in each hour
	# Note: frequencty regulation up and down requirements are symmetric and all resources contributing to regulation are assumed to contribute equal capacity to both up and down directions
	@constraint(STEP, st_cReg[t=1:T, w=1:W], sum(st_vREG[y,t,w] for y in REG) >= STEP[:st_eRegReq][t,w])

	@constraint(STEP, st_cRsvReq[t=1:T, w=1:W], sum(st_vRSV[y,t,w] for y in RSV) + st_vUNMET_RSV[t,w] >= STEP[:st_eRsvReq][t,w])

	@constraint(STEP, st_cUNMET_RSV_MAX[t=1:T,w=1:W], st_vUNMET_RSV[t,w]  <= STEP[:st_eRsvReq][t,w])

	@constraint(STEP, st_cREG_MAX[y in REG,t=1:T,w=1:W], st_vREG[y,t,w] <= STEP[:st_eRegReq][t,w])

	@constraint(STEP, st_cRSV_MAX[y in RSV,t=1:T,w=1:W], st_vRSV[y,t,w] <= STEP[:st_eRsvReq][t,w])

    ### curtailable_variable_renewable

    VRE = rhinputs["VRE"]

	VRE_POWER_OUT = intersect(dfGen[dfGen.Num_VRE_Bins.>=1,:R_ID], VRE)
	# VRE_NO_POWER_OUT = setdiff(VRE, VRE_POWER_OUT) # XXX not used

    ## Power Balance Expressions ##

    @expression(STEP, st_ePowerBalanceDisp[t=1:T, z=1:Z, w=1:W],
	sum(STEP[:st_vP][y,t,w] for y in intersect(VRE, dfGen[dfGen[!,:Zone].==z,:R_ID])))

	STEP[:st_ePowerBalance] += st_ePowerBalanceDisp

    # Constraints
	# For resource for which we are modeling hourly power output
	for y in VRE_POWER_OUT
		# Define the set of generator indices corresponding to the different sites (or bins) of a particular VRE technology (E.g. wind or solar) in a particular zone.
		# For example the wind resource in a particular region could be include three types of bins corresponding to different sites with unique interconnection, hourly capacity factor and maximim available capacity limits.
		VRE_BINS = intersect(dfGen[dfGen[!,:R_ID].>=y,:R_ID], dfGen[dfGen[!,:R_ID].<=y+dfGen[y,:Num_VRE_Bins]-1,:R_ID])


        # Maximum power generated per hour by renewable generators must be less than
        # sum of product of hourly capacity factor for each bin times its the bin installed capacity
        # Note: inequality constraint allows curtailment of output below maximum level.
        @constraint(STEP, [t=1:T,w=1:W], STEP[:st_vP][y,t,w] <= sum(rhinputs["pP_Max"][w][yy,t]*STEP[:st_eTotalCap][yy] for yy in VRE_BINS))


    end


    ### storage.jl
    Reserves = mysetup["Reserves"]
    OperationWrapping = mysetup["OperationWrapping"]
    STOR_SHORT_DURATION = rhinputs["STOR_SHORT_DURATION"]

    STOR_ALL = rhinputs["STOR_ALL"]

    ### investment_energy
    @expression(STEP, st_eExistingCapEnergy[y in STOR_ALL], dfGen[y,:Existing_Cap_MWh])

    @expression(STEP, st_eTotalCapEnergy[y in STOR_ALL], st_eExistingCapEnergy[y] + STEP[:vZERO])


    ### storage_all.jl


	# Storage level of resource "y" at hour "t" [MWh] on zone "z" - unbounded
	@variable(STEP, st_vS[y in STOR_ALL, t=1:T, w=1:W] >= 0);

	# Energy withdrawn from grid by resource "y" at hour "t" [MWh] on zone "z"
	@variable(STEP, st_vCHARGE[y in STOR_ALL, t=1:T, w=1:W] >= 0);

	## Here and Now Variables
	@variable(STEP, st_hnS[y in STOR_ALL, t=[1]] >= 0);
	@variable(STEP, st_hnCHARGE[y in STOR_ALL, t=[1]] >= 0);

	## Nonanticipativity Constraints
	@constraint(STEP, vS_nonantic[y in STOR_ALL, t=[1], w=1:W], st_vS[y,t,w] == st_hnS[y,t])
	@constraint(STEP, vCHARGE_nonantic[y in STOR_ALL, t=[1], w=1:W], st_vCHARGE[y,t,w] == st_hnCHARGE[y,t])

	### Expressions ###

	# Energy losses related to technologies (increase in effective demand)
	@expression(STEP, st_eELOSS[y in STOR_ALL, w=1:W], sum(rhinputs["omega"][t]*STEP[:st_vCHARGE][y,t,w] for t in 1:T) - sum(rhinputs["omega"][t]*STEP[:st_vP][y,t,w] for t in 1:T))

	## Objective Function Expressions ##

	#Variable costs of "charging" for technologies "y" during hour "t" in zone "z"
	@expression(STEP, st_eCVar_in[y in STOR_ALL,t=1:T, w=1:W], rhinputs["omega"][t]*dfGen[y,:Var_OM_Cost_per_MWh_In]*st_vCHARGE[y,t,w])

	# Sum individual resource contributions to variable charging costs to get total variable charging costs
	@expression(STEP, st_eTotalCVarInT[t=1:T, w=1:W], sum(st_eCVar_in[y,t,w] for y in STOR_ALL))
	@expression(STEP, st_eTotalCVarIn[w=1:W], sum(st_eTotalCVarInT[t,w] for t in 1:T))
	STEP[:st_eObj] += st_eTotalCVarIn

	## Power Balance Expressions ##

	# Term to represent net dispatch from storage in any period
	@expression(STEP, st_ePowerBalanceStor[t=1:T, z=1:Z, w=1:W],
		sum(STEP[:st_vP][y,t,w]-STEP[:st_vCHARGE][y,t,w] for y in intersect(dfGen[dfGen.Zone.==z,:R_ID],STOR_ALL)))

	STEP[:st_ePowerBalance] += st_ePowerBalanceStor

	### Constraints ###

	# @constraint(STEP, st_Max_vCHARGE[y in STOR_ALL, t=1:T, w=1:W], st_vCHARGE[y,t,w] <= rhinputs["dfGen"].Existing_Cap_MW[y])

	@constraints(STEP, begin

		# # Max and min constraints on energy storage capacity built (as proportion to discharge power capacity)
		st_cSTOR_MinStorCap[y in STOR_ALL, w=1:W], STEP[:st_eTotalCapEnergy][y] >= dfGen[y,:Min_Duration] * STEP[:st_eTotalCap][y]
		st_cSTOR_MaxStorCap[y in STOR_ALL, w=1:W], STEP[:st_eTotalCapEnergy][y] <= dfGen[y,:Max_Duration] * STEP[:st_eTotalCap][y]

		# Maximum energy stored must be less than installed energy capacity
		st_cSTOR_MaxEnergyVol[y in STOR_ALL, t in 1:T, w=1:W], STEP[:st_vS][y,t,w] <= STEP[:st_eTotalCapEnergy][y]

		# st_cSTOR_MaxCharge[y in STOR_ALL, t in 1:T, w=1:W], STEP[:st_vCHARGE][y,t,w] <= STEP[:st_eTotalCapEnergy][y]

		# # energy stored for the next hour
		# st_cSTOR_SOCInt[t in INTERIOR_SUBPERIODS, y in STOR_ALL, w=1:W], STEP[:st_vS][y,t,w] ==
		# 	STEP[:st_vS][y,t-1,w]-(1/dfGen[y,:Eff_Down]*STEP[:st_vP][y,t,w])+(dfGen[y,:Eff_Up]*STEP[:st_vCHARGE][y,t,w])-(dfGen[y,:Self_Disch]*STEP[:st_vS][y,t-1,w]) 
	end)


    ### storage_all_reserves
    # intialize storage state of charge
    initial_vS = dfGen[!,:Existing_Cap_MWh] * 0.5;

    # parameters
    STOR_REG_RSV = intersect(STOR_ALL, rhinputs["REG"], rhinputs["RSV"]) # Set of storage resources with both REG and RSV reserves

    STOR_REG = intersect(STOR_ALL, rhinputs["REG"]) # Set of storage resources with REG reserves
    STOR_RSV = intersect(STOR_ALL, rhinputs["RSV"]) # Set of storage resources with RSV reserves

    STOR_NO_RES = setdiff(STOR_ALL, STOR_REG, STOR_RSV) # Set of storage resources with no reserves

    STOR_REG_ONLY = setdiff(STOR_REG, STOR_RSV) # Set of storage resources only with REG reserves
    STOR_RSV_ONLY = setdiff(STOR_RSV, STOR_REG) # Set of storage resources only with RSV reserves

    @constraints(STEP, begin
        # Maximum storage contribution to reserves is a specified fraction of installed discharge power capacity
        cSTOR_MaxFreqReg[y in STOR_REG_RSV, t=1:T, w=1:W], STEP[:st_vREG][y,t,w] <= dfGen[y,:Reg_Max]*STEP[:st_eTotalCap][y] #cSTOR_MaxFreqReg
        cSTOR_MaxReserves[y in STOR_REG_RSV, t=1:T, w=1:W], STEP[:st_vRSV][y,t,w] <= dfGen[y,:Rsv_Max]*STEP[:st_eTotalCap][y] #cSTOR_MaxReserves

        # Actual contribution to regulation and reserves is sum of auxilary variables for portions contributed during charging and discharging
        cSTOR_TotRegContrution[y in STOR_REG_RSV, t=1:T, w=1:W], STEP[:st_vREG][y,t,w] == STEP[:st_vREG_charge][y,t,w]+STEP[:st_vREG_discharge][y,t,w] #cSTOR_TotRegContrution
        cSTOR_TotResContribution[y in STOR_REG_RSV, t=1:T, w=1:W], STEP[:st_vRSV][y,t,w] == STEP[:st_vRSV_charge][y,t,w]+STEP[:st_vRSV_discharge][y,t,w] #cSTOR_TotResContribution

        # Maximum charging rate plus contribution to reserves up must be greater than zero
        # Note: when charging, reducing charge rate is contributing to upwards reserve & regulation as it drops net demand
        cSTOR_NonnegNetCharge[y in STOR_REG_RSV, t=1:T, w=1:W], STEP[:st_vCHARGE][y,t,w]-STEP[:st_vREG_charge][y,t,w]-STEP[:st_vRSV_charge][y,t,w] >= 0 #cSTOR_NonnegNetCharge

        # Maximum discharging rate and contribution to reserves down must be greater than zero
        # Note: when discharging, reducing discharge rate is contributing to downwards regulation as it drops net supply
        cSTOR_NonnegNetDischarge[y in STOR_REG_RSV, t=1:T, w=1:W], STEP[:st_vP][y,t,w]-STEP[:st_vREG_discharge][y,t,w] >= 0 #cSTOR_NonnegNetDischarge

        # Maximum discharging rate and contribution to reserves up must be less than power rating OR available stored energy in prior period, whichever is less
        # wrapping from end of sample period to start of sample period for energy capacity constraint
        cSTOR_MaxRegRsvByCap[y in STOR_REG_RSV, t=1:T, w=1:W], STEP[:st_vP][y,t,w]+STEP[:st_vREG_discharge][y,t,w]+STEP[:st_vRSV_discharge][y,t,w] <= STEP[:st_eTotalCap][y] #cSTOR_MaxRegRsvByCap
    end)

    # Reg charge Linking Constraint
    if r == 1
        @constraint(STEP, st_cSTOR_MaxRegChargeLink[y in STOR_REG_RSV, t in START_SUBPERIODS, w=1:W], STEP[:st_vCHARGE][y,t,w]+STEP[:st_vREG_charge][y,t,w] <= 
            STEP[:st_eTotalCapEnergy][y]-initial_vS[y])
    else r > 1
        @constraint(STEP, st_cSTOR_MaxRegChargeLink[y in STOR_REG_RSV, t in START_SUBPERIODS, w=1:W], STEP[:st_vCHARGE][y,t,w]+STEP[:st_vREG_charge][y,t,w] <= 
            STEP[:st_eTotalCapEnergy][y]-STEP[:st_vS][y,t-1,w])
    end

    # Reg charge Interior Constraint
    @constraint(STEP, st_cSTOR_MaxRegChargeInt[y in STOR_REG_RSV, t in INTERIOR_SUBPERIODS, w=1:W], STEP[:st_vCHARGE][y,t,w]+STEP[:st_vREG_charge][y,t,w] <= 
        STEP[:st_eTotalCapEnergy][y]-STEP[:st_vS][y,t-1,w]) 

    # Reg Rsv Linking
    if r == 1
        @constraint(STEP, st_cSTOR_MaxRegRsvLink[y in STOR_REG_RSV, t in START_SUBPERIODS, w=1:W], STEP[:st_vP][y,t,w]+STEP[:st_vREG_discharge][y,t,w] +
            STEP[:st_vRSV_discharge][y,t,w] <= initial_vS[y])
    else r > 1
        @constraint(STEP, st_cSTOR_MaxRegRsvLink[y in STOR_REG_RSV, t in START_SUBPERIODS, w=1:W], STEP[:st_vP][y,t,w]+STEP[:st_vREG_discharge][y,t,w] +
            STEP[:st_vRSV_discharge][y,t,w] <= STEP[:st_vS][y,t-1,w])
    end

    # Reg Rsv Interior
    @constraint(STEP, st_cSTOR_MaxRegRsvInt[y in STOR_REG_RSV, t in INTERIOR_SUBPERIODS, w=1:W], STEP[:st_vP][y,t,w]+STEP[:st_vREG_discharge][y,t,w] + 
        STEP[:st_vRSV_discharge][y,t,w] <= STEP[:st_vS][y,t-1,w]) #cSTOR_MaxRegRsvLink


    if r == 1 ### XXX This is where the wrap-around or the warm-start needs to be implemented
        @constraint(STEP, st_cSTOR_SOCLink[y in STOR_ALL, t in START_SUBPERIODS, w = 1:W], STEP[:st_vS][y,t,w] ==
        initial_vS[y]-(1/dfGen[y,:Eff_Down]*STEP[:st_vP][y,t,w])
            +(dfGen[y,:Eff_Up]*STEP[:st_vCHARGE][y,t,w])-(dfGen[y,:Self_Disch]*initial_vS[y]))
    else r > 1
        @constraint(STEP, st_cSTOR_SOCLink[y in STOR_ALL, t in START_SUBPERIODS, w = 1:W], STEP[:st_vS][y,t,w] ==
            s_dp[y,r-1]-(1/dfGen[y,:Eff_Down]*STEP[:st_vP][y,t,w])
            +(dfGen[y,:Eff_Up]*STEP[:st_vCHARGE][y,t,w])-(dfGen[y,:Self_Disch]*s_dp[y,r-1]))
    end

    # energy stored for the next hour
    @constraint(STEP, st_cSTOR_SOCInt[y in STOR_ALL, t in INTERIOR_SUBPERIODS, w = 1:W], STEP[:st_vS][y,t,w] ==
        STEP[:st_vS][y,t-1,w]-(1/dfGen[y,:Eff_Down]*STEP[:st_vP][y,t,w])+(dfGen[y,:Eff_Up]*STEP[:st_vCHARGE][y,t,w])-(dfGen[y,:Self_Disch]*STEP[:st_vS][y,t-1,w]))


    # thermal.jl
    THERM_COMMIT = rhinputs["THERM_COMMIT"]
	THERM_NO_COMMIT = rhinputs["THERM_NO_COMMIT"]
	THERM_ALL = rhinputs["THERM_ALL"]

    # thermal_commit.jl
    ### Expressions ###

	## Power Balance Expressions ##
	@expression(STEP, st_ePowerBalanceThermCommit[t=1:T, z=1:Z, w=1:W],
    sum(STEP[:st_vP][y,t,w] for y in intersect(THERM_COMMIT, dfGen[dfGen[!,:Zone].==z,:R_ID])))

    STEP[:st_ePowerBalance] += st_ePowerBalanceThermCommit

    	### Capacitated limits on unit commitment decision variables (Constraints #1-3)
	@constraints(STEP, begin
        st_cTC_MaxCommitUnits[y in THERM_COMMIT, t=1:T, w=1:W], STEP[:st_vCOMMIT][y,t,w] <= STEP[:st_eTotalCap][y]/dfGen[y,:Cap_Size] #cTC-MaxCommitUnits
        st_cTC_MaxStartupUnits[y in THERM_COMMIT, t=1:T, w=1:W], STEP[:st_vSTART][y,t,w] <= STEP[:st_eTotalCap][y]/dfGen[y,:Cap_Size] #cTC-MaxStartupUnits
        st_cTC_MaxShutdownUnits[y in THERM_COMMIT, t=1:T, w=1:W], STEP[:st_vSHUT][y,t,w] <= STEP[:st_eTotalCap][y]/dfGen[y,:Cap_Size] #cTC-MaxShutdownUnits
    end)

    ### cTC
    # Define important parameters for new constraints
    Up_Time = zeros(Int, size(dfGen, 1))
    Up_Time[THERM_COMMIT] .= Int.(floor.(dfGen[THERM_COMMIT,:Up_Time]))

    Down_Time = zeros(Int, size(dfGen,1))
    Down_Time[THERM_COMMIT] .= Int.(floor.(dfGen[THERM_COMMIT,:Down_Time]))

    p = policy_model_length # XXX might need to be changed to include warm start # Maybe Tmax?
    T = policy_model_length  # rhindex["T"]

    # find max up or down time
    all_times = [Up_Time Down_Time]
    max_UDtime = maximum(all_times)

    @constraint(STEP, st_cTC_MinDownTime_Link[y in THERM_COMMIT, t in 1:(Down_Time[y]-1), w = 1:W], 
        STEP[:st_eTotalCap][y]/dfGen[y,:Cap_Size] - STEP[:st_vCOMMIT][y,t,w] >= 
        sum(shut_dp[y,hoursbefore(Tend,r,1:Down_Time[y])]) + sum(STEP[:st_vSHUT][y,rt,w] for rt in 1:t)
    )

    @constraint(STEP, st_cTC_MinUpTime_Link[y in THERM_COMMIT, t in 1:(Up_Time[y]-1), w = 1:W], 
        STEP[:st_vCOMMIT][y,t,w] >= sum(start_dp[y,hoursbefore(Tend,r,1:Up_Time[y])]) 
        + sum(STEP[:st_vSTART][y,rt,w] for rt in 1:t)
    )

    # add constraints beyond max_UDtime
    @constraint(STEP, st_cTC_MinDownTime[y in THERM_COMMIT, t in (max_UDtime):T, w = 1:W],
        STEP[:st_eTotalCap][y]/dfGen[y,:Cap_Size]-STEP[:st_vCOMMIT][y,t,w] >= sum(STEP[:st_vSHUT][y, hoursbefore(p, t, 0:(Down_Time[y] - 1)), w])
    )

    @constraint(STEP, st_cTC_MinUpTime[y in THERM_COMMIT, t in (max_UDtime):T, w = 1:W],
        STEP[:st_vCOMMIT][y,t,w] >= sum(STEP[:st_vSTART][y, hoursbefore(p, t, 0:(Up_Time[y] - 1)), w])
    ) ### XXX could max_UDtime be replaced with Up_Time[y] ?

    num_starting_units = dfGen[!,:Existing_Cap_MW] ./ dfGen[!,:Cap_Size] * 0.5
    # for y in THERM_COMMIT
    #     num_starting_units[y] = gen[y].existing_cap_mw / dfGen[y,:Cap_Size] * 0.5
    # end

    ### Update Thermal Single Intertemp Constraints
    ## cTC_BalCommitUnits
    @constraint(STEP, st_cTC_CommitUnitsInt[y in THERM_COMMIT, t in INTERIOR_SUBPERIODS, w=1:W], STEP[:st_vCOMMIT][y,t,w] == STEP[:st_vCOMMIT][y,t-1,w] + STEP[:st_vSTART][y,t,w] - STEP[:st_vSHUT][y,t,w])
    ## cTC_LinkCommitUnits
    if r == 1 ### XXX This is where the wrap-around or the warm-start needs to be implemented
        @constraint(STEP, st_cTC_CommitUnitsLink[y in THERM_COMMIT, t in START_SUBPERIODS, w=1:W], STEP[:st_vCOMMIT][y,t,w] == num_starting_units[y] + STEP[:st_vSTART][y,t,w] - STEP[:st_vSHUT][y,t,w])
    else r > 1
        # @constraint(EP, cTC_CommitUnitsLink[y in THERM_COMMIT], EP[:vCOMMIT][y,1] == EP[:vSTART][y,1] - EP[:vSHUT][y,1])
        @constraint(STEP, st_cTC_CommitUnitsLink[y in THERM_COMMIT, t in START_SUBPERIODS, w=1:W], STEP[:st_vCOMMIT][y,t,w] == commit_dp[y,r-1] + STEP[:st_vSTART][y,t,w] - STEP[:st_vSHUT][y,t,w])
    end

    initial_vP = dfGen[!,:Existing_Cap_MW] * 0.5;

    ## cTC_MaxRampDownInt
    @constraint(STEP, st_cTC_MaxRampDownInt[y in THERM_COMMIT, t in INTERIOR_SUBPERIODS, w=1:W],
    STEP[:st_vP][y,t-1,w]-STEP[:st_vP][y,t,w] <= dfGen[y,:Ramp_Dn_Percentage]*dfGen[y,:Cap_Size]*(STEP[:st_vCOMMIT][y,t,w]-STEP[:st_vSTART][y,t,w])
        -dfGen[y,:Min_Power]*dfGen[y,:Cap_Size]*STEP[:st_vSTART][y,t,w]
        +min(rhinputs["pP_Max"][w][y,t],max(dfGen[y,:Min_Power],dfGen[y,:Ramp_Dn_Percentage]))*dfGen[y,:Cap_Size]*STEP[:st_vSHUT][y,t,w])
    ## cTC_MaxRampDownLink
    if r == 1 ### XXX This is where the wrap-around or the warm-start needs to be implemented
        @constraint(STEP, st_cTC_MaxRampDownLink[y in THERM_COMMIT, t in START_SUBPERIODS, w=1:W],
        initial_vP[y] - STEP[:st_vP][y,t,w] <= dfGen[y,:Ramp_Dn_Percentage]*dfGen[y,:Cap_Size]*(STEP[:st_vCOMMIT][y,t,w]-STEP[:st_vSTART][y,t,w])
            - dfGen[y,:Min_Power]*dfGen[y,:Cap_Size]*STEP[:st_vSTART][y,t,w]
            + min(rhinputs["pP_Max"][w][y,t],max(dfGen[y,:Min_Power],dfGen[y,:Ramp_Dn_Percentage]))*dfGen[y,:Cap_Size]*STEP[:st_vSHUT][y,t,w])
    else r > 1
        @constraint(STEP,st_cTC_MaxRampDownLink[y in THERM_COMMIT, t in START_SUBPERIODS,w=1:W],
        pgen_dp[y,r-1] - STEP[:st_vP][y,t,w] <= dfGen[y,:Ramp_Dn_Percentage]*dfGen[y,:Cap_Size]*(STEP[:st_vCOMMIT][y,t,w]-STEP[:st_vSTART][y,t,w])
            - dfGen[y,:Min_Power]*dfGen[y,:Cap_Size]*STEP[:st_vSTART][y,t,w]
            + min(rhinputs["pP_Max"][w][y,t],max(dfGen[y,:Min_Power],dfGen[y,:Ramp_Dn_Percentage]))*dfGen[y,:Cap_Size]*STEP[:st_vSHUT][y,t,w])
    end

    ## cTC_MaxRampUpInt
    @constraint(STEP,st_cTC_MaxRampUpInt[y in THERM_COMMIT, t in INTERIOR_SUBPERIODS,w=1:W],
        STEP[:st_vP][y,t,w]-STEP[:st_vP][y,t-1,w] <= dfGen[y,:Ramp_Up_Percentage]*dfGen[y,:Cap_Size]*(STEP[:st_vCOMMIT][y,t,w]-STEP[:st_vSTART][y,t,w])
            + min(rhinputs["pP_Max"][w][y,t],max(dfGen[y,:Min_Power],dfGen[y,:Ramp_Up_Percentage]))*dfGen[y,:Cap_Size]*STEP[:st_vSTART][y,t,w]
            -dfGen[y,:Min_Power]*dfGen[y,:Cap_Size]*STEP[:st_vSHUT][y,t,w])
    ## cTC_MaxRampUpLink
    if r == 1 ### XXX This is where the wrap-around or the warm-start needs to be implemented
        @constraint(STEP,st_cTC_MaxRampUpLink[y in THERM_COMMIT, t in START_SUBPERIODS,w=1:W],
        STEP[:st_vP][y,t,w]- initial_vP[y] <= dfGen[y,:Ramp_Up_Percentage]*dfGen[y,:Cap_Size]*(STEP[:st_vCOMMIT][y,t,w] - STEP[:st_vSTART][y,t,w])
            + min(rhinputs["pP_Max"][w][y,t],max(dfGen[y,:Min_Power],dfGen[y,:Ramp_Up_Percentage]))*dfGen[y,:Cap_Size]*STEP[:st_vSTART][y,t,w]
            - dfGen[y,:Min_Power]*dfGen[y,:Cap_Size]*STEP[:st_vSHUT][y,t,w])
    else r > 1
        @constraint(STEP,st_cTC_MaxRampUpLink[y in THERM_COMMIT, t in START_SUBPERIODS,w=1:W],
        STEP[:st_vP][y,t,w]- pgen_dp[y,r-1] <= dfGen[y,:Ramp_Up_Percentage]*dfGen[y,:Cap_Size]*(STEP[:st_vCOMMIT][y,t,w] - STEP[:st_vSTART][y,t,w])
            + min(rhinputs["pP_Max"][w][y,t],max(dfGen[y,:Min_Power],dfGen[y,:Ramp_Up_Percentage]))*dfGen[y,:Cap_Size]*STEP[:st_vSTART][y,t,w]
            - dfGen[y,:Min_Power]*dfGen[y,:Cap_Size]*STEP[:st_vSHUT][y,t,w])
    end

    # stochastic_thermal_commit_reserves
    THERM_COMMIT_REG_RSV = intersect(THERM_COMMIT, rhinputs["REG"], rhinputs["RSV"])
    @constraints(STEP, begin
        # Maximum regulation and reserve contributions
        st_cMaxREGContrib[y in THERM_COMMIT_REG_RSV, t=1:T, w=1:W], STEP[:st_vREG][y,t,w] <= rhinputs["pP_Max"][w][y,t]*dfGen[y,:Reg_Max]*dfGen[y,:Cap_Size]*STEP[:st_vCOMMIT][y,t,w] 
        st_cMaxRSVContrib[y in THERM_COMMIT_REG_RSV, t=1:T, w=1:W], STEP[:st_vRSV][y,t,w] <= rhinputs["pP_Max"][w][y,t]*dfGen[y,:Rsv_Max]*dfGen[y,:Cap_Size]*STEP[:st_vCOMMIT][y,t,w] 

        # Minimum stable power generated per technology "y" at hour "t" and contribution to regulation must be > min power
        st_cMinStablePower[y in THERM_COMMIT_REG_RSV, t=1:T, w=1:W], STEP[:st_vP][y,t,w]-STEP[:st_vREG][y,t,w] >= dfGen[y,:Min_Power]*dfGen[y,:Cap_Size]*STEP[:st_vCOMMIT][y,t,w] 

        # Maximum power generated per technology "y" at hour "t"  and contribution to regulation and reserves up must be < max power
        st_cMaxPowerWithRegRsv[y in THERM_COMMIT_REG_RSV, t=1:T, w=1:W], STEP[:st_vP][y,t,w]+STEP[:st_vREG][y,t,w]+STEP[:st_vRSV][y,t,w] <= rhinputs["pP_Max"][w][y,t]*dfGen[y,:Cap_Size]*STEP[:st_vCOMMIT][y,t,w]
    end)

	#### Define the Objective ####

	## assign probabilities to stochastic scenarios
	uniform_probs = 1 / W
	## redefine objective 
	scalar_st_eObj = uniform_probs .* ones(1,W) *  STEP[:st_eObj]
	# update the objective function expression
	@expression(STEP, single_st_eObj, scalar_st_eObj[1])


	## Define the objective function
	@objective(STEP, Min, single_st_eObj)

	## Power balance constraints
	# demand = generation + storage discharge - storage charge - demand deferral + deferred demand satisfaction - demand curtailment (NSE)
	#          + incoming power flows - outgoing power flows - flow losses - charge of heat storage + generation from NACC
	@constraint(STEP, st_cPowerBalance[t=1:T, z=1:Z, w=1:W], STEP[:st_ePowerBalance][t,z,w] == rhinputs["pD"][w][t,z]) 

	# PowerBalanceObj = sum(st_cPowerBalance[t=1:T, z=1:Z, w] for w in 1:W) # T x Z, 48 x 1

	## Record pre-solver time
	presolver_time = time() - presolver_start_time
    	#### Question - What do we do with this time now that we've split this function into 2?
	if mysetup["PrintModel"] == 1
		if modeloutput === nothing
			filepath = joinpath(pwd(), "YourModel.lp")
			JuMP.write_to_file(STEP, filepath)
		else
			filepath = joinpath(modeloutput, "YourModel.lp")
			JuMP.write_to_file(STEP, filepath)
		end
		println("Model Printed")
    end

    ### Solve STEP model ###
    println("Solving Model") 
    # STEP, solve_time = solve_model(STEP, mysetup)


    # set_attribute(STEP, "DualReductions", 0)
    # set_attribute(STEP, "BarHomogeneous", 1)
    optimize!(STEP)
    println("Debugging Solving Model")
    println("termination status : ", termination_status(STEP))
    println("result count       : ", result_count(STEP))
    println("primal status      : ", primal_status(STEP))
    println("dual status        : ", dual_status(STEP))
    if termination_status(STEP) == MOI.INFEASIBLE
        println("Model did not solve to optimality")
        compute_conflict!(STEP)
        iis_model, _ = copy_conflict(STEP)
        print(iis_model)
    end
    # println("termination status : ", termination_status(STEP))
    myinputs["solve_time"] = solve_time # Store the model solve time in myinputs

    # println("Debugging Solving Model")
    # compute_conflict!(STEP)
    # iis_model, _ = copy_conflict(STEP)
    # print(iis_model)


    #=======================================================================
    WRITE DECISION VARIABLES TO DICTIONARIES
    ========================================================================#
    

    # save all decision variables from time 1 - vSHUT, vSTART, vCOMMIT, vP, vCharge, vS, vRSV, vREG, eTotalCap
    # for however many generators, y, there are, save variables

    for gen in GENERATORS
        # sP[gen][r] = value(STEP[:st_vP][gen,1,1])
        pgen_dp[gen, r] = value(STEP[:st_vP][gen,1,1])
    end
    
    for zone in 1:1
        # sNSE[zone][r] = value(STEP[:st_vNSE][zone,1,1,1])
        nse_dp[zone, r] = value(STEP[:st_vNSE][zone,1,1,1])
        unmet_rsv_dp[zone,r] = value(STEP[:st_vUNMET_RSV][1,1])
    end
   

    for gen in THERM_COMMIT
        # push!(sSHUT[gen], value(EP[:vSHUT][gen,1]))
        # push!(sSTART[gen], value(EP[:vSTART][gen,1]))
        # push!(sCOMMIT[gen], value(EP[:vCOMMIT][gen,1]))   
        
        # Reserve and Regulation
        # sRSV[gen][r] = value(STEP[:st_vRSV][gen,1,1])
        rsv_dp[gen,r] = value(STEP[:st_vRSV][gen,1,1])
        # sREG[gen][r] = value(STEP[:st_vREG][gen,1,1])
        reg_dp[gen,r] = value(STEP[:st_vREG][gen,1,1])

        # commitment variables
        # sSHUT[gen][r] = value(STEP[:st_vSHUT][gen,1,1])
        shut_dp[gen,r] = value(STEP[:st_vSHUT][gen,1,1])
        # sSTART[gen][r] = value(STEP[:st_vSTART][gen,1,1])
        start_dp[gen, r] = value(STEP[:st_vSTART][gen,1,1])
        # sCOMMIT[gen][r] = value(STEP[:st_vCOMMIT][gen,1,1])
        commit_dp[gen,r] = value(STEP[:st_vCOMMIT][gen,1,1])
    end
    ### include Res and Reg variables XXX ???
    for sto in STOR_LIST
        # push!(sP[sto], value(EP[:vP][sto,1]))
        # push!(sS[sto], value(EP[:vS][sto,1]))
        # push!(sCHARGE[sto], value(EP[:vCHARGE][sto,1]))
        # println("the storage value is :", value(STEP[:st_vS][sto,1]))
        # sS[sto][r] = value(STEP[:st_vS][sto,1,1])
        s_dp[sto,r] = value(STEP[:st_vS][sto,1,1]) # / ModelScalingFactor # state of charge
        # sCHARGE[sto][r] = value(STEP[:st_vCHARGE][sto,1,1])
        charge_dp[sto,r] = value(STEP[:st_vCHARGE][sto,1,1])
    end


    #calculate price at r as the sum of all scenario prices at the first horizon time step
    elec_prices[r] = sum(dual.(STEP[:st_cPowerBalance])[1,1,:]) #* ModelScalingFactor # convert $/GWh to $/MWh
    # price[r] = transpose(dual.(STEP[:st_cPowerBalance])[:,:,1])[1] .* 1000 # incorrect price calculation
    reg_prices[r] = sum(dual.(STEP[:st_cReg])[1,:,:]) #* ModelScalingFactor
    rsv_prices[r]  = sum(dual.(STEP[:st_cRsvReq])[1,:,:]) #* ModelScalingFactor

    pri_dict["PowerBalance"][r] = sum(dual.(STEP[:st_cPowerBalance])[1,:,:])
    pri_dict["Reg"][r] = sum(dual.(STEP[:st_cReg])[1,:,:])
    pri_dict["RsvReq"][r] = sum(dual.(STEP[:st_cRsvReq])[1,:,:])

    # # save duals on discharge constraints
    # # IF STOR_ALL is non-empty then save duals
    # if !isempty(STOR_ALL)
    #     max_discharge_const_duals[r] = dual.(STEP[:st_Max_vP])[STOR_ALL[],:,:]
    #     max_charge_const_duals[r] = Matrix(dual.(STEP[:st_cSTOR_MaxRegRsvLink])[STOR_ALL[],:,:]) #length(STOR_ALL)
    #     soc_link_duals[r] = Matrix(dual.(STEP[:st_cSTOR_SOCLink])[STOR_ALL[],:,:])
    #     soc_int_duals[r] = Matrix(dual.(STEP[:st_cSTOR_SOCInt])[STOR_ALL[],:,:])
    # end

    prices_scen_array[r] = dual.(STEP[:st_cPowerBalance])[:,:,:] .* ModelScalingFactor


    #=======================================================================
    Updating Revenue and Costs
    =======================================================================#   
    energy_revs[:,r] = pgen_dp[:,r] .* elec_prices[r] .* ModelScalingFactor^2
    var_om_costs[:,r] = dfGen[:,:Var_OM_Cost_per_MWh].* pgen_dp[:,r] .* ModelScalingFactor^2
    fuel_costs[:,r] = myinputs["C_Fuel_per_MWh"][:,r] .* pgen_dp[:,r] .* ModelScalingFactor^2
    reg_revs[:,r] = pri_dict["Reg"][r] .* reg_dp[:,r] .* ModelScalingFactor^2
    rsv_revs[:,r] = pri_dict["RsvReq"][r] .* rsv_dp[:,r] .* ModelScalingFactor^2
    start_costs[:,r] = myinputs["C_Start"][:,r] .* start_dp[:,r] .* ModelScalingFactor^2
    charge_costs[:,r] = charge_dp[:,r] .* elec_prices[r] .* ModelScalingFactor^2

    nse_cost[1,r] = myinputs["pC_D_Curtail"][1] * nse_dp[1,r] .* ModelScalingFactor^2
    unmet_rsv_cost[1,r] = myinputs["pC_Rsv_Penalty"] * unmet_rsv_dp[1,r] .* ModelScalingFactor^2





    ## END LOOKAHEAD FOR LOOP


end ### Uncomment for Rolling Horizon Loop?


# #=======================================================================
# Save Files
# =======================================================================#


if !isdir(casepath)
    mkdir(casepath)
end

if !isdir(savepath)
    mkdir(savepath)
else 
    files = readdir(savepath)
    for file in files
        if !isdir(joinpath(savepath, file))
            rm(joinpath(savepath, file))
        end
    end
end



writedlm(savepath * "unit_shut.csv", shut_dp, ',')
writedlm(savepath *  "unit_start.csv", start_dp, ',')
writedlm(savepath * "unit_commit.csv", commit_dp, ',')
writedlm(savepath * "unit_rsv.csv", rsv_dp * ModelScalingFactor, ',')
writedlm(savepath * "unit_reg.csv", reg_dp * ModelScalingFactor, ',')
writedlm(savepath * "unit_pgen.csv", pgen_dp* ModelScalingFactor, ',')
writedlm(savepath * "unit_state_of_charge.csv", s_dp* ModelScalingFactor, ',')
writedlm(savepath * "unit_charge.csv", charge_dp* ModelScalingFactor, ',')
writedlm(savepath * "price_electricity.csv", elec_prices * ModelScalingFactor, ',')
writedlm(savepath * "prices_reg.csv", reg_prices * ModelScalingFactor, ',')
writedlm(savepath * "prices_rsv.csv", rsv_prices * ModelScalingFactor, ',')
writedlm(savepath * "zone_nse.csv", nse_dp * ModelScalingFactor, ',')
writedlm(savepath * "zone_unmet_rsv.csv", unmet_rsv_dp * ModelScalingFactor, ',')



#=======================================================================
Calculate Profits per Generator and Total Welfare
=======================================================================#


### components over whole year
fixed_om_costs_vec = dfGen[!,:Fixed_OM_Cost_per_MWyr].*dfGen[!,:Existing_Cap_MW] .* ModelScalingFactor^2
fixed_om_costs = reshape(fixed_om_costs_vec, (length(GENERATORS), 1))

total_energy_revs = sum(energy_revs, dims=2)
total_reg_revs = sum(reg_revs, dims=2)
total_rsv_revs = sum(rsv_revs, dims=2)

rev_per_gen = total_energy_revs + total_reg_revs + total_rsv_revs

total_var_om_costs = sum(var_om_costs, dims=2)
total_fuel_costs = sum(fuel_costs, dims=2)
total_start_costs = sum(start_costs, dims=2)
total_charge_costs = sum(charge_costs, dims=2)

cost_per_gen = total_var_om_costs + total_fuel_costs + total_start_costs + total_charge_costs + fixed_om_costs

# in dollars
operating_profit_per_gen = sum(energy_revs, dims=2) + sum(reg_revs, dims=2)  +
                     sum(rsv_revs, dims=2) - sum(var_om_costs, dims=2) -
                     sum(fuel_costs, dims=2) - sum(start_costs, dims=2) -
                     sum(charge_costs, dims=2) - fixed_om_costs;

total_welfare = sum(operating_profit_per_gen) - sum(nse_cost) - sum(unmet_rsv_cost);            

writedlm(savepath * "revenue_operating_profit_per_gen.csv", operating_profit_per_gen, ',')
writedlm(savepath * "revenue_total_welfare.csv", total_welfare, ',')
writedlm(savepath * "revenue_energy_revs.csv", energy_revs, ',')
writedlm(savepath * "revenue_reg_revs.csv", reg_revs, ',')
writedlm(savepath * "revenue_rsv_revs.csv", rsv_revs, ',')
writedlm(savepath * "revenue_var_om_costs.csv", var_om_costs, ',')
writedlm(savepath * "revenue_fuel_costs.csv", fuel_costs, ',')
writedlm(savepath * "revenue_start_costs.csv", start_costs, ',')
writedlm(savepath * "revenue_charge_costs.csv", charge_costs, ',')
writedlm(savepath * "revenue_nse_cost.csv", nse_cost, ',')
writedlm(savepath * "revenue_unmet_rsv_cost.csv", unmet_rsv_cost, ',')


# convert dfGen existing capacity back to MW
copy_dfGen = deepcopy(dfGen)
copy_dfGen[!,:Existing_Cap_MW] = copy_dfGen[!,:Existing_Cap_MW] .* ModelScalingFactor
copy_dfGen[!,:Existing_Cap_MWh] = copy_dfGen[!,:Existing_Cap_MWh] .* ModelScalingFactor
copy_dfGen[!,:Inv_Cost_per_MWyr] = copy_dfGen[!,:Inv_Cost_per_MWyr] .* ModelScalingFactor
copy_dfGen[!,:Inv_Cost_per_MWhyr] = copy_dfGen[!,:Inv_Cost_per_MWhyr] .* ModelScalingFactor
copy_dfGen[!,:Fixed_OM_Cost_per_MWyr] = copy_dfGen[!,:Fixed_OM_Cost_per_MWyr] .* ModelScalingFactor
CSV.write(casepath * "generator_characteristics.csv", copy_dfGen, header=true)


invest_costs_perMW_yr = dfGen[!,:Inv_Cost_per_MWyr] .* ModelScalingFactor
invest_costs_perMWhour_yr = dfGen[!,:Inv_Cost_per_MWhyr] .* ModelScalingFactor

total_inv_costs_MW_yr = dfGen[!,:Inv_Cost_per_MWyr].*dfGen[!,:Existing_Cap_MW] * ModelScalingFactor^2
total_inv_costs_MWhour_yr = dfGen[!,:Inv_Cost_per_MWhyr].*dfGen[!,:Existing_Cap_MWh] * ModelScalingFactor^2


operating_profit_per_gen_vec = operating_profit_per_gen[:]
total_inv_costs_MW_yr_vec = total_inv_costs_MW_yr[:]
total_inv_costs_MWhour_yr_vec = total_inv_costs_MWhour_yr[:]
diff = operating_profit_per_gen_vec - total_inv_costs_MW_yr_vec - total_inv_costs_MWhour_yr_vec;

# Create a DataFrame
df = DataFrame(generators = dfGen[!,:Resource],
                Capacity_MW = dfGen[!,:Existing_Cap_MW] * ModelScalingFactor,
                Capacity_MWh = dfGen[!,:Existing_Cap_MWh] * ModelScalingFactor,
                Inv_cost_MW = total_inv_costs_MW_yr_vec,
                Inv_cost_MWh = total_inv_costs_MWhour_yr_vec,
                Fixed_OM_cost_MW = fixed_om_costs[:],
                Fixed_OM_cost_MWh = zeros(num_gen,1)[:],
                Var_OM_cost_out = total_var_om_costs[:],
                Fuel_cost = total_fuel_costs[:],
                Var_OM_cost_in = zeros(num_gen,1)[:],
                StartCost = total_start_costs[:],
                Charge_cost = total_charge_costs[:],
                EnergyRevenue = total_energy_revs[:],
                RegRevenue = total_reg_revs[:],
                RsvRevenue = total_rsv_revs[:],
                Revenue = rev_per_gen[:],
                Cost = cost_per_gen[:],
                operating_profit_per_gen = operating_profit_per_gen_vec,
                diff = diff)

# Write the DataFrame to a CSV file
println("Writing operating profit results to CSV")
CSV.write(savepath * "00_operating_profits_results.csv", df, header=true)

#=======================================================================
Create HDF5 Files for saving Arrays of scenarios, duals, and prices
=======================================================================#

save_hdf5(savepath, Tend, "prices_scen_array", prices_scen_array)

STOR_ALL = myinputs["STOR_ALL"]

if !isempty(STOR_ALL)
    # print duals to hdf5
    # save_hdf5(savepath, Tend, "max_discharge_const_duals", max_discharge_const_duals)
    # save_hdf5(savepath, Tend, "max_charge_const_duals", max_charge_const_duals)
    # save_hdf5(savepath, Tend, "soc_link_duals", soc_link_duals)
    # save_hdf5(savepath, Tend, "soc_int_duals", soc_int_duals)
end


# save scenarios to hdf5
# save_hdf5(savepath, Tend, "load_scen_array", load_scen_array)
# save_hdf5(savepath, Tend, "solar_scen_array", solar_scen_array)
# save_hdf5(savepath, Tend, "wind_scen_array", wind_scen_array)



# #=
# =#
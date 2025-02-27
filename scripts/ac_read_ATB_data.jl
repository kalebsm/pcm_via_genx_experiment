using DataFrames
using XLSX
using Dates
using CSV
using PyCall

# Define the path to the ATB-calc directory
atb_calc_path = joinpath(dirname(pwd()), "data", "ATB-calc")



# Define the path to the Python script
python_script_path = joinpath(atb_calc_path, "lcoe_calculator", "process_all.py")

# Import the necessary Python modules
py"""
import sys
sys.path.append("$atb_calc_path/lcoe_calculator")
from process_all import ProcessAll
from tech_processors import LandBasedWindProc, UtilityPvProc, CoalProc, NaturalGasProc, UtilityBatteryProc
"""

# Display settings for Jupyter notebooks
display("text/html", "<style>.container { width:90% !important; }</style>")

if !@isdefined(atb_df) || atb_df === nothing
    # The below line MUST be updated to reflect the location of the ATB workbook on your computer
    atb_electricity_workbook = joinpath("..", "data", "2024 v2 Annual Technology Baseline Workbook Errata 7-19-2024.xlsx")
    
    # Process all technologies
    techs = [LandBasedWindProc, UtilityPvProc, CoalProc, NaturalGasProc, UtilityBatteryProc]

    # Initiate the processor with the workbook location and desired technologies
    processor = ProcessAll(atb_electricity_workbook, techs)

    start = now()
    processor.process()
    println("Processing completed in ", now() - start)

    atb_df = processor.data
end

# Define location of cost assumptions
generator_assumptions_path = joinpath("..", "data", "cases")

# a_upd_generator_df.csv path
a_upd_generator_df_path = joinpath("..", "data", "a_upd_generator_df.csv")

# Load in generator param data shell
upd_gen_df = CSV.read(a_upd_generator_df_path, DataFrame)
atb_upd_gen_df = deepcopy(upd_gen_df)

# Get the list of all files in the generator_assumptions_path directory
case_names_list = filter(x -> endswith(x, ".xlsx"), readdir(generator_assumptions_path))
case_names_list = replace.(case_names_list, ".xlsx" => "")
println(case_names_list)

# Load initialized generator df csv
gen_2_model_df = CSV.read("a_initialized_generator_df.csv", DataFrame)
gen_2_model_names = gen_2_model_df.Resource

# Get unique values from atb_df
atb_gen_names = unique(atb_df.DisplayName)

# Find the intersection of the two lists
inters_names = intersect(gen_2_model_names, atb_gen_names)

# Assumptions
assumed_case = "Market"
assumed_crpyears = 20
assumed_scenario = "Moderate"
assumed_year = 2022
assumed_interest = 0.8

# Write dictionary of parameters in GenX title vs ATB title
gen_2_atb_dict = Dict(
    "Inv_Cost_per_MWyr" => "CAPEX",
    "Fixed_OM_Cost_per_MWyr" => "Fixed O&M",
    "Var_OM_Cost_per_MWh" => "Variable O&M",
    "Heat Rate" => "Heat Rate"
)

for gen_name in inters_names
    # Get technology type
    tech_type = atb_df[atb_df.DisplayName .== gen_name, :Technology][1]
    
    atb_gen_df = atb_df[(atb_df.Case .== assumed_case) .& 
                        (atb_df.CRPYears .== assumed_crpyears) .& 
                        (atb_df.Scenario .== assumed_scenario) .& 
                        (atb_df.DisplayName .== gen_name), :]

    inv_cost_kilowatt = atb_gen_df[(atb_gen_df.Parameter .== "CAPEX"), Symbol(assumed_year)][1]
    fixed_om_kilowatt = atb_gen_df[(atb_gen_df.Parameter .== "Fixed O&M"), Symbol(assumed_year)][1]
    var_om_mw = atb_gen_df[(atb_gen_df.Parameter .== "Variable O&M"), Symbol(assumed_year)][1]

    Inv_Cost_per_MWyr = inv_cost_kilowatt * 1000 / assumed_crpyears
    Fixed_OM_cost_per_MWyr = fixed_om_kilowatt * 1000
    Var_OM_Cost_per_MWh = var_om_mw

    Heat_Rate_MMBTU_per_MWh = nothing
    # Get into resource specific costs
    if tech_type == "Utility-Scale Battery Storage"
        # Do nothing
    elseif tech_type == "UtilityPV"
        Var_OM_Cost_per_MWh = 0.1
    elseif tech_type == "LandbasedWind"
        Var_OM_Cost_per_MWh = 0.1
    elseif tech_type in ["Coal_FE", "NaturalGas_FE"]
        Heat_Rate_MMBTU_per_MWh = atb_gen_df[(atb_gen_df.Parameter .== "Heat Rate"), Symbol(assumed_year)][1]
    else
        throw(ArgumentError("Unhandled technology type: $tech_type"))
    end
    
    atb_upd_gen_df[atb_upd_gen_df.Resource .== gen_name, "Inv_Cost_per_MWyr"] .= Inv_Cost_per_MWyr
    atb_upd_gen_df[atb_upd_gen_df.Resource .== gen_name, "Fixed_OM_Cost_per_MWyr"] .= Fixed_OM_cost_per_MWyr
    atb_upd_gen_df[atb_upd_gen_df.Resource .== gen_name, "Var_OM_Cost_per_MWh"] .= Var_OM_Cost_per_MWh
    atb_upd_gen_df[atb_upd_gen_df.Resource .== gen_name, "Heat_Rate_MMBTU_per_MWh"] .= Heat_Rate_MMBTU_per_MWh
end

# Print out updated generator df
CSV.write("a_upd_generator_df.csv", atb_upd_gen_df)
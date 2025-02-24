using DataFrames, CSV, XLSX

# define location of cost assumptions
script_dir = @__DIR__
generator_assumptions_path = joinpath(script_dir, "..", "data", "cases")

# Get the list of all files in the generator_assumptions_path directory
case_names_list = String[]
unique_gen_names = Set{String}()

for xlsx_name in readdir(generator_assumptions_path)
    # save xlsx path
    xlsx_path = joinpath(generator_assumptions_path, xlsx_name)
    if isfile(xlsx_path)
        case_name = replace(xlsx_name, ".xlsx" => "")
        push!(case_names_list, case_name)

        # for every case, read the generator assumptions file and get a list of all unique 'Technical Name'
        df = DataFrame(XLSX.readtable(xlsx_path, "Sheet1"))
        if "Technical Name" in names(df)
            global unique_gen_names 
            unique_gen_names = union(unique_gen_names, unique(df[!, "Technical Name"]))
        end
    end
end

# sort names by alphabetical order
sorted_unique_gen_names = sort(collect(unique_gen_names))

println(case_names_list)
println(unique_gen_names)

num_unique_gen_names = length(unique_gen_names)
gen_parameter_names = [
    "Resource",
    "Model",
    "New_Build",
    "Can_Retire",
    "Zone",
    "THERM",
    "MUST_RUN",
    "STOR",
    "FLEX",
    "HYDRO",
    "VRE",
    "SOLAR",
    "WIND",
    "Num_VRE_Bins",
    "Existing_Cap_MW",
    "Existing_Cap_MWh",
    "Existing_Charge_Cap_MW",
    "Max_Cap_MW",
    "Max_Cap_MWh",
    "Min_Charge_Cap_MW",
    "Min_Cap_MW",
    "Min_Cap_MWh",
    "Inv_Cost_per_MWyr",
    "Inv_Cost_per_MWhyr",
    "Inv_Cost_Charge_per_MWyr",
    "Fixed_OM_Cost_per_MWyr",
    "Fixed_OM_Cost_per_MWhyr",
    "Fixed_OM_Cost_Charge_per_MWyr",
    "Var_OM_Cost_per_MWh",
    "Var_OM_Cost_per_MWh_In",
    "Heat_Rate_MMBTU_per_MWh",
    "Fuel",
    "Cap_Size",
    "Start_Cost_per_MW",
    "Start_Fuel_MMBTU_per_MW",
    "Up_Time",
    "Down_Time",
    "Ramp_Up_Percentage",
    "Ramp_Dn_Percentage",
    "Hydro_Energy_to_Power_Ratio",
    "Min_Power",
    "Self_Disch",
    "Eff_Up",
    "Eff_Down",
    "Min_Duration",
    "Max_Duration",
    "Max_Flexible_Demand_Advance",
    "Max_Flexible_Demand_Delay",
    "Flexible_Demand_Energy_Eff",
    "Reg_Max",
    "Rsv_Max",
    "Reg_Min",
    "Reg_Cost",
    "Rsv_Cost",
    "MinCapTag",
    "MinCapTag_1",
    "MinCapTag_2",
    "MinCapTag_3",
    "MGA",
    "Resource_Type",
    "CapRes_1",
    "ESR_1",
    "ESR_2",
    "region",
    "cluster",
    "LDS"
]


# Initialize a dataframe with gen names as rows and parameters as columns
gen_data = Dict(param => [missing for _ in unique_gen_names] for param in gen_parameter_names)
gen_df = DataFrame(gen_data)
gen_df.Resource = collect(unique_gen_names)
# set resource as first columns
gen_df = gen_df[:, ["Resource", gen_parameter_names[1:end-1]...]]


# print dataframe to csv
CSV.write("a_initialized_generator_df.csv", gen_df, writeheader=true)

# print a updatable version of dataframe
CSV.write("a_upd_generator_df.csv", gen_df, writeheader=true)
# println(gen_df)
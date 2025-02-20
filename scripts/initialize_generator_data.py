import pandas as pd
import numpy as np
import shutil
import os
import sys
from datetime import datetime as dt
import textwrap


# define location of cost assumptions
generator_assumptions_path = os.path.join('..', 'data', 'cases')
# Get the list of all files in the generator_assumptions_path directory
case_names_list = []

unique_gen_names = set()

for csv_name in os.listdir(generator_assumptions_path):
    # save xlsx path
    csv_path = os.path.join(generator_assumptions_path, csv_name)
    if os.path.isfile(csv_path):
        case_name = csv_name.replace('.xlsx', '')
        case_names_list.append(case_name)

        # for every case, read the generator assumptions file and get a list of all unique 'Technical Name'
        df = pd.read_excel(csv_path)
        if 'Technical Name' in df.columns:
            unique_gen_names.update(df['Technical Name'].unique())


# # sort names by alphabetical order
sorted_unique_gen_names = sorted(list(unique_gen_names))

wrapped_case_names_list = textwrap.fill(str(case_names_list), width=70)
wrapped_unique_gen_names = textwrap.fill(str(unique_gen_names), width=70)

# print(wrapped_case_names_list)
# print(wrapped_unique_gen_names)
num_unique_gen_names = len(unique_gen_names)
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
    "LDS",
]


wrapped_gen_parameter_names = textwrap.fill(str(gen_parameter_names), width=70)
# print(wrapped_gen_parameter_names)
unique_gen_names
# Initialize a dataframe with indices labeled 1 through to the number of gen names and parameters as columns
gen_df = pd.DataFrame(index=range(1, len(unique_gen_names) + 1), columns=gen_parameter_names)
gen_df['Resource'] = sorted(list(unique_gen_names))

# print dataframe to csv
gen_df.to_csv('a_initialized_generator_df.csv', index=False)

# print a updatable verison of dataframe
gen_df.to_csv('a_upd_generator_df.csv', index=True)
# print(gen_df)
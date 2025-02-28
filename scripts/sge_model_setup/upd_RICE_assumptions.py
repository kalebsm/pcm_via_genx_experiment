import pandas as pd
import numpy as np
import shutil
import os
import sys
from datetime import datetime as dt

# Add the root directory (my_package) to sys.path so Python can find 'utils'
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # One level up from 'scripts'
sys.path.append(root_path)

# Now import from utils
from utils.sge_utils import get_paths
data_path = get_paths('data')


def get_cleaned_cost_df(gen_costs_df, spec_gen_df):
    #
    costs = gen_costs_df[gen_costs_df['Generic Name'].isin(spec_gen_df['Generic Name'])]
    costs.drop('Generic Name', axis=1, inplace=True)
    # get the nameplate capacity of the generators
    nameplate_cap = spec_gen_df['NAMEPLATE (MWs)'].values
    # normalize the BIT_ST_coal_costs by the nameplate capacity
    costs_normalized = costs.div(nameplate_cap, axis=0)
    # change the titles of the columns to reflect the normalization
    costs_normalized.columns = ['No Load Cost ($/MW)', 'Cold Start Cost ($/MW)', 'Hot Start Cost ($/MW)']

    return costs_normalized

# a_upd_generator_df path
a_upd_generator_df_path = os.path.join(data_path, 'a_upd_generator_df.csv')
# read in upd_gen csv
upd_gen_df = pd.read_csv(a_upd_generator_df_path)
rice_upd_gen_df = upd_gen_df.copy()
# load in the RICE data
rice_data = rice_upd_gen_df[upd_gen_df['Resource'] == 'RICE']
# the data that should exist for RICE is:
reqd_thermal_data = [
                  'Resource',
                #   'Zone',
                #   'Model',
                #   'New_Build',
                #   'Can_Retire',
                #   'Existing_Cap_MW',
                #   'Max_Cap_MW',
                #   'Min_Cap_MW',
                  'Inv_Cost_per_MWyr',
                  'Fixed_OM_Cost_per_MWyr',
                  'Var_OM_Cost_per_MWh',
                  'Heat_Rate_MMBTU_per_MWh',
                  'Fuel',
                  'Cap_Size',
                  'Start_Cost_per_MW',
                  # 'Start_Fuel_MMBTU_per_MW',
                  'Up_Time',
                  'Down_Time',
                  'Ramp_Up_Percentage',
                  'Ramp_Dn_Percentage',
                  'Min_Power',
                #   'Reg_Max',
                #   'Reg_Min',
                #   'Reg_Cost',
                #   'Rsv_Cost',
                #   'region',
                #   'cluster',
                  ]
# for every column in the required data, check if 
# data exists in the RICE data
missing_data = []

for col in reqd_thermal_data:
    if rice_data[col].isnull().all():
        missing_data.append(col)

print("Missing data columns:\n", "\n".join(missing_data))
## develop generator costs
'''Data from https://www.eia.gov/analysis/studies/powerplants/capitalcost/pdf/capital_cost_AEO2020.pdf'''

Heat_Rate_MMBTU_per_MWh = 9.717

Cap_Size = 21.4
Min_Power=  0.5


gen_lifetime = 20
Inv_Cost_per_MWyr = 1810 * 1000 / gen_lifetime
Inv_Cost_per_MWhyr = 0 * 1000 / gen_lifetime

Fixed_OM_Cost_per_MWyr = 35.16 * 1000
Var_OM_Cost_per_MWh = 5.69
### additional parameters from FERC data
# load FERC data
# define location of ferc data
ferc_data_loc = os.path.join('..', 'data', 'ferc_generator_parameters')

# Get the sheet names
# assume summer data
ferc_data_path = os.path.join(ferc_data_loc, '20120724-4012_Generator_Data_Summer.xlsx')
ferc_excel_file = pd.ExcelFile(ferc_data_path)

sheet_names = ferc_excel_file.sheet_names

# Create a dictionary to store the separated dataframes
separated_dataframes = {}

# Iterate over each sheet and store the data in the dictionary
for sheet_name in sheet_names:
    separated_dataframes[sheet_name] = ferc_excel_file.parse(sheet_name)

# Process for Cleaning Generator Characteristics
gen_characteristics = separated_dataframes['Generator Characteristics']
gen_characteristics.columns = separated_dataframes['Generator Characteristics'].iloc[0]
gen_characteristics.drop(0, inplace=True)
gen_characteristics.reset_index(drop=True, inplace=True)
# gen_characteristics.drop_duplicates(gen_characteristics.columns[duplicate_index], axis=1)
gen_characteristics= gen_characteristics.loc[:,~gen_characteristics.columns.duplicated()].copy()

# Process for Cleaning Generator costs
gen_costs = separated_dataframes['Generator Offer Curve'].iloc[:,24:28]
gen_costs.columns = gen_costs.iloc[1]
gen_costs.drop([0,1], inplace=True)
gen_costs.reset_index(drop=True, inplace=True)
gen_costs.rename(columns={'1': 'index'}, inplace=True)
gen_costs.rename(columns={np.nan: 'Generic Name'}, inplace=True)

# the important paramaters are:
raw_key_params = ['NAMEPLATE (MWs)','RAMP UP (MW/min)', 'RAMP DOWN (MW/min)', 'Economic Minimum (MW)', 'MIN_DOWN_TIME (hr)', 'MIN_RUN_TIME (hr)']

ramp_params = ['RAMP UP (MW/min)', 'RAMP DOWN (MW/min)']
capacity_param = ['NAMEPLATE (MWs)']

# Divide each row of ramp_params by NAMEPLATE (MWs) and multiply by 60 to convert to per hour
percent_ramp =  gen_characteristics[ramp_params].div(gen_characteristics['NAMEPLATE (MWs)'], axis=0) * 60
gen_characteristics[ramp_params] = percent_ramp

# change the name of the columns to reflect the change in units for ramping
gen_characteristics.rename(columns={'RAMP UP (MW/min)': 'PERC RAMP UP', 'RAMP DOWN (MW/min)': 'PERC RAMP DOWN'}, inplace=True)

# update key param names
cleaned_key_params = ['NAMEPLATE (MWs)','PERC RAMP UP', 'PERC RAMP DOWN', 'Economic Minimum (MW)', 'MIN_DOWN_TIME (hr)', 'MIN_RUN_TIME (hr)']
# assume fuel is DFO
fuel_id = 'DFO'
Fuel = fuel_id
# assume prime mover is GT
prime_mover = 'GT'

rice = gen_characteristics[(gen_characteristics['Energy_Source_1 (Fuel)'] == fuel_id) 
                                & (gen_characteristics['PRIMEMOVER'] == prime_mover)]

cleaned_cost_df = get_cleaned_cost_df(gen_costs, rice)
mean_costs = cleaned_cost_df.mean()


## develop generator parameters
Start_Cost_per_MW = mean_costs.mean()

# parameters 
Ramp_Up_Percentage = rice['PERC RAMP UP'].mean()
Ramp_Dn_Percentage =  rice['PERC RAMP DOWN'].mean()


Down_Time = rice['MIN_DOWN_TIME (hr)'].min()
Up_Time = rice['MIN_RUN_TIME (hr)'].min()
# update Rice data in the upd_gen_df
rice_upd_gen_df.loc[rice_upd_gen_df['Resource'] == 'RICE', [
    'Inv_Cost_per_MWyr', 'Fixed_OM_Cost_per_MWyr', 'Var_OM_Cost_per_MWh', 
    'Heat_Rate_MMBTU_per_MWh', 'Cap_Size', 'Start_Cost_per_MW', 
    'Up_Time', 'Down_Time', 'Ramp_Up_Percentage', 'Ramp_Dn_Percentage', 
    'Min_Power', 'Fuel'
]] = [
    Inv_Cost_per_MWyr, Fixed_OM_Cost_per_MWyr, Var_OM_Cost_per_MWh, 
    Heat_Rate_MMBTU_per_MWh, Cap_Size, Start_Cost_per_MW, 
    Up_Time, Down_Time, Ramp_Up_Percentage, Ramp_Dn_Percentage, 
    Min_Power, Fuel
]
# double check no missing data
upd_rice_data = upd_gen_df.loc[upd_gen_df['Resource'] == 'RICE']

# for every column in the required data, check if 
# data exists in the RICE data
missing_data = []

for col in reqd_thermal_data:
    if upd_rice_data[col].isnull().all():
        missing_data.append(col)

print("Missing data columns:\n", "\n".join(missing_data))
# save updated data to csv
rice_upd_gen_df.to_csv(a_upd_generator_df_path, index=False)
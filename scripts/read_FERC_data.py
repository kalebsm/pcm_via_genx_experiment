import pandas as pd
import numpy as np
import shutil
import os
import sys
from datetime import datetime as dt

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

# define location of ferc data
ferc_data_loc = os.path.join('..', 'data', 'ferc_generator_parameters')

# Get the sheet names
# assume summer data
ferc_data_path = os.path.join(ferc_data_loc, '20120724-4012_Generator_Data_Summer.xlsx')
ferc_excel_file = pd.ExcelFile(ferc_data_path)

# define location of cost assumptions
generator_assumptions_path = os.path.join('..', 'data', 'cases')

# a_upd_generator_df path
a_upd_generator_df_path = os.path.join('data', 'a_upd_generator_df.csv')
# read in updatable df
upd_gen_df = pd.read_csv(a_upd_generator_df_path)
# create copy of upd_gen_df to update
ferc_upd_gen_df = upd_gen_df.copy()
# read in data_source_comparisons
manual_db_rel = pd.read_csv(os.path.join('data','manual_db_rel.csv'))
unique_gen = upd_gen_df['Resource']

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
ferc_2_genx_dic = {
    'PERC RAMP UP': 'Ramp_Up_Percentage',
    'PERC RAMP DOWN': 'Ramp_Down_Percentage',
    'Economic Minimum (MW)': 'Min_Power',
    'MIN_DOWN_TIME (hr)': 'Down_Time',
    'MIN_RUN_TIME (hr)': 'Up_Time',
    'NAMEPLATE (MWs)': 'Cap_Size',
}

# # initialize lists for all genx parameters
# Ramp_Up_Percentage = [None] * len(unique_gen)
# Ramp_Down_Percentage = [None] * len(unique_gen)
# Min_Power = [None] * len(unique_gen)
# Down_Time = [None] * len(unique_gen)
# Up_Time = [None] * len(unique_gen)
# Cap_Size = [None] * len(unique_gen)


# get prime move and fuel connections
for gen_name in unique_gen:
# for gen_name in unique_gen[0:1]:
    # get fuel type
    fuel_id = manual_db_rel[manual_db_rel['Resource'] == gen_name]['Fuel ID'].values[0]
    # get primary mover
    mover_id = manual_db_rel[manual_db_rel['Resource'] == gen_name]['Primary Mover ID'].values[0]

    print(fuel_id, mover_id)
    # if 'none' then skip
    if fuel_id == None or mover_id == None:
        ferc_upd_gen_df.loc[ferc_upd_gen_df['Resource'] == gen_name, 'Fuel'] = str('None')
        print(f'No fuel or mover for {gen_name}')
        continue

    # get the specific generator data
    specified_gen_df = gen_characteristics[(gen_characteristics['Energy_Source_1 (Fuel)'] == fuel_id)
                                            & (gen_characteristics['PRIMEMOVER'] == mover_id)]
    
    if specified_gen_df.empty:
        print(f'No data for {gen_name} given fuel_id: {fuel_id} and mover_id: {mover_id}')
        continue
    
    cleaned_cost_df = get_cleaned_cost_df(gen_costs, specified_gen_df)
    # ^ maybe this should be printed out as a csv file

    ### Decision processes for choosing costs and parameters
    # costs
    mean_costs = cleaned_cost_df.mean()
    Start_Cost_per_MW = mean_costs.mean()

    Cap_Size = specified_gen_df['NAMEPLATE (MWs)'].mean()

    # parameters 
    Ramp_Up_Percentage = specified_gen_df['PERC RAMP UP'].mean()
    Ramp_Dn_Percentage =  specified_gen_df['PERC RAMP DOWN'].mean()
    Min_Power =  specified_gen_df['Economic Minimum (MW)'].mean() / Cap_Size

    Down_Time =  specified_gen_df['MIN_DOWN_TIME (hr)'].min()
    Up_Time =  specified_gen_df['MIN_RUN_TIME (hr)'].min()

    # save results to ferc_upd_gen_df
    ferc_upd_gen_df.loc[ferc_upd_gen_df['Resource'] == gen_name, 'Start_Cost_per_MW'] = Start_Cost_per_MW
    ferc_upd_gen_df.loc[ferc_upd_gen_df['Resource'] == gen_name, 'Cap_Size'] = Cap_Size
    ferc_upd_gen_df.loc[ferc_upd_gen_df['Resource'] == gen_name, 'Ramp_Up_Percentage'] = Ramp_Up_Percentage
    ferc_upd_gen_df.loc[ferc_upd_gen_df['Resource'] == gen_name, 'Ramp_Dn_Percentage'] = Ramp_Dn_Percentage
    ferc_upd_gen_df.loc[ferc_upd_gen_df['Resource'] == gen_name, 'Min_Power'] = Min_Power
    ferc_upd_gen_df.loc[ferc_upd_gen_df['Resource'] == gen_name, 'Down_Time'] = Down_Time
    ferc_upd_gen_df.loc[ferc_upd_gen_df['Resource'] == gen_name, 'Up_Time'] = Up_Time
    ferc_upd_gen_df.loc[ferc_upd_gen_df['Resource'] == gen_name, 'Fuel'] = fuel_id


# save ferc_upd_gen_df to csv
ferc_upd_gen_df.to_csv('a_upd_generator_df.csv', index=False)
import pandas as pd
import numpy as np
import shutil
import os
import sys
from datetime import datetime as dt
import textwrap


# read in upd_gen csv
upd_gen_df = pd.read_csv('a_upd_generator_df.csv')
smr_upd_gen_df = upd_gen_df.copy()
# load in SMR data
smr_foak_cost_df = pd.read_csv(os.path.join('..', 'data', 'SMR_data', 'FOAK_Costs.csv'))
smr_foak_cost_df
# list SMR types
smr_types = [
    'HTGR',
    'Microreactor',
    'PBR-HTGR',
    'iPWR_Pack',
]

# isolate SMR rows
smr_rows = smr_upd_gen_df[smr_upd_gen_df['Resource'].isin(smr_types)]

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

# determine missing data
missing_data = {}

for index, row in smr_rows.iterrows():
    missing_columns = [col for col in reqd_thermal_data if pd.isna(row[col])]
    if missing_columns:
        missing_data[row['Resource']] = missing_columns

# for resource, columns in missing_data.items():
#     print(resource)
#     print(textwrap.fill(", ".join(columns), width=70))
#     print()
# assumptions

gen_lifetime = 20
for index, row in smr_foak_cost_df.iterrows():
    # get the resource
    resource = row['Sites']

    Cap_Size = row['Power in MWe']

    Min_Power = row['MSL in MWe'] / Cap_Size

    Down_Time = row['MDT in hours']
    # assume up time is same as down time
    Up_Time = Down_Time

    Ramp_Dn_Percentage = row['Ramp Rate (fraction of capacity/hr)']

    # assume ramp up is same as ramp down
    Ramp_Up_Percentage = Ramp_Dn_Percentage

    Inv_Cost_per_MWyr = row['CAPEX $/kWe'] * 1000 / gen_lifetime

    Fixed_OM_Cost_per_MWyr = row['FOPEX $/kWe'] * 1000 / gen_lifetime

    Var_OM_Cost_per_MWh = row['VOM in $/MWh-e']

    # fuel Cost
    Start_Cost_per_MW = row['Startupfixedcost in $'] / Cap_Size

    Heat_Rate_MMBTU_per_MWh = 10.46 # hard coded from GenX

    Fuel = 'UM'

    # add data to smr_upd_gen_df
    smr_upd_gen_df.loc[smr_upd_gen_df['Resource'] == resource, 'Cap_Size'] = Cap_Size
    smr_upd_gen_df.loc[smr_upd_gen_df['Resource'] == resource, 'Min_Power'] = Min_Power
    smr_upd_gen_df.loc[smr_upd_gen_df['Resource'] == resource, 'Down_Time'] = Down_Time
    smr_upd_gen_df.loc[smr_upd_gen_df['Resource'] == resource, 'Up_Time'] = Up_Time
    smr_upd_gen_df.loc[smr_upd_gen_df['Resource'] == resource, 'Ramp_Dn_Percentage'] = Ramp_Dn_Percentage
    smr_upd_gen_df.loc[smr_upd_gen_df['Resource'] == resource, 'Ramp_Up_Percentage'] = Ramp_Up_Percentage
    smr_upd_gen_df.loc[smr_upd_gen_df['Resource'] == resource, 'Inv_Cost_per_MWyr'] = Inv_Cost_per_MWyr
    smr_upd_gen_df.loc[smr_upd_gen_df['Resource'] == resource, 'Fixed_OM_Cost_per_MWyr'] = Fixed_OM_Cost_per_MWyr
    smr_upd_gen_df.loc[smr_upd_gen_df['Resource'] == resource, 'Var_OM_Cost_per_MWh'] = Var_OM_Cost_per_MWh
    smr_upd_gen_df.loc[smr_upd_gen_df['Resource'] == resource, 'Heat_Rate_MMBTU_per_MWh'] = Heat_Rate_MMBTU_per_MWh
    smr_upd_gen_df.loc[smr_upd_gen_df['Resource'] == resource, 'Start_Cost_per_MW'] = Start_Cost_per_MW
    smr_upd_gen_df.loc[smr_upd_gen_df['Resource'] == resource, 'Fuel'] = Fuel

second_check_smr_df = smr_upd_gen_df[smr_upd_gen_df['Resource'].isin(smr_types)]

# check that no batteries are missing data
missing_data = {}

for index, row in second_check_smr_df.iterrows():
    missing_columns = [col for col in reqd_thermal_data if pd.isna(row[col])]
    if missing_columns:
        missing_data[row['Resource']] = missing_columns

if not missing_data:
    print("No SMRs are missing data.")
else:
    for resource, columns in missing_data.items():
        print(resource)
        print(textwrap.fill(", ".join(columns), width=70))
        print()
# save to csv
smr_upd_gen_df.to_csv('a_upd_generator_df.csv', index=False)
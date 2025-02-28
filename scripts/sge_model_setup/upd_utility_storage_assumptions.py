import pandas as pd
import numpy as np
import os
import shutil
import sys
from datetime import datetime as dt
import textwrap


# Add the root directory (my_package) to sys.path so Python can find 'utils'
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # One level up from 'scripts'
sys.path.append(root_path)

# Now import from utils
from utils.sge_utils import get_paths
scripts_path = get_paths('scripts')
data_path = get_paths('data')
genx_research_path = get_paths('genx_research')
spcm_research_path = get_paths('spcm_research')
figures_path = get_paths('figures')
atb_calc_path = get_paths('atb-calc')


# a_upd_generator_df path
a_upd_generator_df_path = os.path.join(data_path, 'a_upd_generator_df.csv')
# read in upd_gen csv
upd_gen_df = pd.read_csv(a_upd_generator_df_path)
bess_upd_gen_df = upd_gen_df.copy()
# load utility-scale battery storage data
utility_scale_battery_df = bess_upd_gen_df[bess_upd_gen_df['Resource'].str.contains('Utility-Scale Battery Storage', na=False)]
# print(utility_scale_battery_df)

reqd_bess_data = [
                #   'Zone',
                #   'Model',
                #   'New_Build',
                #   'Can_Retire',
                #   'Existing_Cap_MW',
                #   'Max_Cap_MW',
                #   'Max_Cap_MWh',
                #   'Min_Cap_MW',
                  # 'Min_Cap_MWh',
                  'Inv_Cost_per_MWyr',
                  'Inv_Cost_per_MWhyr',
                  'Fixed_OM_Cost_per_MWyr',
                  'Fixed_OM_Cost_per_MWhyr',
                  'Var_OM_Cost_per_MWh',
                  'Var_OM_Cost_per_MWh_In',
                  'Self_Disch',
                  'Eff_Up',
                  'Eff_Down',
                  'Min_Duration',
                  'Max_Duration',
                #   'Reg_Max',
                #   'Rsv_Max',
                #   'Reg_Cost',
                #   'Rsv_Cost',
                #   'region',
                #   'cluster',
]


# for each battery, check that all required data is present
missing_data = {}

for index, row in utility_scale_battery_df.iterrows():
    missing_columns = [col for col in reqd_bess_data if pd.isna(row[col])]
    if missing_columns:
        missing_data[row['Resource']] = missing_columns

for resource, columns in missing_data.items():
    print(resource)
    print(textwrap.fill(", ".join(columns), width=70))
    print()

# assume costs are accounted for in MWyr instead of MWhyr
bess_upd_gen_df['Inv_Cost_per_MWhyr'] = 0
bess_upd_gen_df['Fixed_OM_Cost_per_MWhyr'] = 0
bess_upd_gen_df['Var_OM_Cost_per_MWh_In'] = 0
bess_upd_gen_df['Self_Disch'] = 0

# assume efficiency is 85%
bess_upd_gen_df['Eff_Up'] = 0.85
bess_upd_gen_df['Eff_Down'] = 0.85
# loop through each battery and set min duration and max duration equal to the number in their name
for index, row in bess_upd_gen_df.iterrows():
    if 'Battery Storage' in row['Resource']:
        duration = int(row['Resource'].split('-')[-1].replace('Hr', ''))
        bess_upd_gen_df.at[index, 'Min_Duration'] = duration
        bess_upd_gen_df.at[index, 'Max_Duration'] = duration
# check that no batteries are missing data
missing_data = {}
# select only battery storage resources
second_check_bess_df = bess_upd_gen_df[bess_upd_gen_df['Resource'].str.contains('Battery Storage', na=False)]

for index, row in second_check_bess_df.iterrows():
    missing_columns = [col for col in reqd_bess_data if pd.isna(row[col])]
    if missing_columns:
        missing_data[row['Resource']] = missing_columns

if not missing_data:
    print("No batteries are missing data.")
else:
    for resource, columns in missing_data.items():
        print(resource)
        print(textwrap.fill(", ".join(columns), width=70))
        print()


bess_upd_gen_df.to_csv(a_upd_generator_df_path, index=False)
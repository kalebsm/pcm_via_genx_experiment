import pandas as pd
import numpy as np
import os
import sys

# Add the root directory (my_package) to sys.path so Python can find 'utils'
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # One level up from 'scripts'
sys.path.append(root_path)

# Now import from utils
from utils.sge_utils import get_paths
data_path = get_paths('data')

# read in data_source_comparisons
manual_db_rel = pd.read_csv(os.path.join(data_path,'manual_db_rel.csv'))


# a_upd_generator_df path
a_upd_generator_df_path = os.path.join(data_path, 'a_upd_generator_df.csv')
# read in updatable df
upd_gen_df = pd.read_csv(a_upd_generator_df_path)
model_param_upd_df = upd_gen_df.copy()
model_param_upd_df['Max_Cap_MW'] = -1
model_param_upd_df['Max_Cap_MWh'] = -1
model_param_upd_df['Min_Cap_MW'] = 0
model_param_upd_df['Min_Cap_MWh'] = 0
model_param_upd_df['Max_Charge_Cap_MW'] = -1
model_param_upd_df['region'] = 'ERCOT'
model_param_upd_df['cluster'] = 0

model_param_upd_df['Zone'] = 1

model_param_upd_df['Reg_Min'] = 0
model_param_upd_df['Reg_Max'] = 0
model_param_upd_df['Rsv_Cost'] = 0
model_param_upd_df['Reg_Cost'] = 0
model_param_upd_df['Start_Fuel_MMBTU_per_MW'] = 0

# for now set existing capacity to 0
model_param_upd_df['Existing_Cap_MW'] = 0
model_param_upd_df['Existing_Cap_MWh'] = 0

model_param_upd_df['Model'] = 1
# for each row in updatable df, get the corresponding energy type from manual_db_rel
for index, row in model_param_upd_df.iterrows():
    # get the energy type from manual_db_rel
    energy_type = manual_db_rel.loc[manual_db_rel['Resource'] == row['Resource']]['ATB Technology Name'].values[0]
    # print(row)
    if energy_type == 'NaturalGas_FE' or energy_type == 'Coal_FE' or energy_type == 'Nuclear':
        model_param_upd_df.loc[index,'THERM'] = 1
        model_param_upd_df.loc[index,'Model'] = 1
        model_param_upd_df.loc[index,'Reg_Max'] = 0.25
        model_param_upd_df.loc[index,'Rsv_Max'] = 0.50
    elif energy_type == 'UtilityPV':
        model_param_upd_df.loc[index,'VRE'] = 1
        model_param_upd_df.loc[index,'SOLAR'] = 1
        model_param_upd_df.loc[index,'Num_VRE_Bins'] = 1
        model_param_upd_df.loc[index,'ESR_1'] = 1
        model_param_upd_df.loc[index,'ESR_2'] = 1

        model_param_upd_df.loc[index,'Reg_Max'] = 0
        model_param_upd_df.loc[index,'Rsv_Max'] = 0
    elif energy_type == 'LandbasedWind':
        model_param_upd_df.loc[index,'VRE'] = 1
        model_param_upd_df.loc[index,'WIND'] = 1
        model_param_upd_df.loc[index,'Num_VRE_Bins'] = 1
        model_param_upd_df.loc[index,'ESR_1'] = 1
        model_param_upd_df.loc[index,'ESR_2'] = 1

        model_param_upd_df.loc[index,'Reg_Max'] = 0
        model_param_upd_df.loc[index,'Rsv_Max'] = 0
    elif energy_type == 'Hydro':
        model_param_upd_df.loc[index,'HYDRO'] = 1
    elif energy_type == 'Utility-Scale Battery Storage':
        model_param_upd_df.loc[index,'STOR'] = 1
        model_param_upd_df.loc[index,'Reg_Max'] = 0.25
        model_param_upd_df.loc[index,'Rsv_Max'] = 0.50
    else:
        pass

# save model_param_upd_df to csv
model_param_upd_df.to_csv(a_upd_generator_df_path, index=False)
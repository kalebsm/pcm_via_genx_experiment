import pandas as pd
import numpy as np
import os

# read in data_source_comparisons
manual_db_rel = pd.read_csv(os.path.join('data','manual_db_rel.csv'))


# a_upd_generator_df path
a_upd_generator_df_path = os.path.join('data', 'a_upd_generator_df.csv')
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
    energy_type = manual_db_rel.loc[manual_db_rel['Resource'] == row['Resource']]['Energy Type'].values[0]
    # print(row)
    if energy_type == 'Thermal':
        model_param_upd_df.at[index,'THERM'] = 1
        model_param_upd_df.at[index,'Model'] = 1
        model_param_upd_df.at[index,'Reg_Max'] = 0.25
        model_param_upd_df.at[index,'Rsv_Max'] = 0.50
    elif energy_type == 'Solar':
        model_param_upd_df.at[index,'VRE'] = 1
        model_param_upd_df.at[index,'SOLAR'] = 1
        model_param_upd_df.at[index,'Num_VRE_Bins'] = 1
        model_param_upd_df.at[index,'ESR_1'] = 1
        model_param_upd_df.at[index,'ESR_2'] = 1

        model_param_upd_df.at[index,'Reg_Max'] = 0
        model_param_upd_df.at[index,'Rsv_Max'] = 0
    elif energy_type == 'Wind':
        model_param_upd_df.at[index,'VRE'] = 1
        model_param_upd_df.at[index,'WIND'] = 1
        model_param_upd_df.at[index,'Num_VRE_Bins'] = 1
        model_param_upd_df.at[index,'ESR_1'] = 1
        model_param_upd_df.at[index,'ESR_2'] = 1

        model_param_upd_df.at[index,'Reg_Max'] = 0
        model_param_upd_df.at[index,'Rsv_Max'] = 0
    elif energy_type == 'Hydro':
        model_param_upd_df.at[index,'HYDRO'] = 1
    elif energy_type == 'Storage':
        model_param_upd_df.at[index,'STOR'] = 1
        model_param_upd_df.at[index,'Reg_Max'] = 0.25
        model_param_upd_df.at[index,'Rsv_Max'] = 0.50
    else:
        pass
model_param_upd_df
# save model_param_upd_df to csv
model_param_upd_df.to_csv('a_upd_generator_df.csv', index=False)
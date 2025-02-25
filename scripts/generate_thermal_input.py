import pandas as pd
import numpy as np
import os
from get_case_names import get_case_names


# define location of cost assumptions
generator_assumptions_path = os.path.join('data', 'cases')

# generate research systems folder

# define path locations for CEM and LACs where inputs are going
genx_cem_loc = os.path.join('GenX.jl', 'research_systems')
spcm_lac_loc = os.path.join('SPCM', 'research_systems')

# Get the list of all files in the generator_assumptions_path directory
case_names_list = get_case_names(generator_assumptions_path)

# a_upd_generator_df path
a_upd_generator_df_path = os.path.join('data', 'a_upd_generator_df.csv')
# read in upd_gen csv
upd_gen_df = pd.read_csv(a_upd_generator_df_path)
# read in data_source_comparisons
manual_db_rel = pd.read_csv(os.path.join('..', 'data','manual_db_rel.csv'))
# the data that should exist for RICE is:
reqd_thermal_data = [
                  'Resource',
                  'Zone',
                  'Model',
                  'New_Build',
                  'Can_Retire',
                  'Existing_Cap_MW',
                  'Max_Cap_MW',
                  'Min_Cap_MW',
                  'Inv_Cost_per_MWyr',
                  'Fixed_OM_Cost_per_MWyr',
                  'Var_OM_Cost_per_MWh',
                  'Heat_Rate_MMBTU_per_MWh',
                  'Fuel',
                  'Cap_Size',
                  'Start_Cost_per_MW',
                  'Start_Fuel_MMBTU_per_MW',
                  'Up_Time',
                  'Down_Time',
                  'Ramp_Up_Percentage',
                  'Ramp_Dn_Percentage',
                  'Min_Power',
                  'Reg_Max',
                  'Rsv_Max',
                  'Reg_Cost',
                  'Rsv_Cost',
                  'region',
                  'cluster',
                  ]
# get list of 'Resources' in manual_db_rel that are thermal 'Energy Type'
thermal_resources = manual_db_rel[manual_db_rel['Energy Type'] == 'Thermal']['Resource'].tolist()

# get dataframe from upd_gen_df that has 'Resource' in thermal_resources
thermal_df = upd_gen_df[upd_gen_df['Resource'].isin(thermal_resources)]

# extract thermal data of required columns
reqd_thermal_df = thermal_df[reqd_thermal_data]
reqd_thermal_df

for case_name in case_names_list:
# for case_name in case_names_list[0:1]:
    # load cem and lac paths
    genx_cem_resources_path = os.path.join(genx_cem_loc, case_name, 'resources')
    spcm_lac_resources_path = os.path.join(spcm_lac_loc, case_name, 'resources')

    # read in cost assumption in generator_assumptions_path for case
    case_assumptions = pd.read_excel(os.path.join(generator_assumptions_path, case_name + '.xlsx'))

    # create a new copy of the reqd_thermal_df for each case
    case_thermal_df = reqd_thermal_df.copy()

    # only take the resources in case_thermal_df that are in case_assumptions
    case_thermal_df = case_thermal_df[case_thermal_df['Resource'].isin(case_assumptions['Technical Name'])]

    for index, row in case_assumptions.iterrows():
        # get the resource name
        resource = row['Technical Name']
        # multiply the cost in case_thermal_df by the corresponding factor in cost_assumptions
        case_thermal_df.loc[case_thermal_df['Resource'] == resource, 'Inv_Cost_per_MWyr'] = \
            case_thermal_df.loc[case_thermal_df['Resource'] == resource, 'Inv_Cost_per_MWyr'] * row['Inv_Cost_per_MWyr_factor']
        case_thermal_df.loc[case_thermal_df['Resource'] == resource, 'Fixed_OM_Cost_per_MWyr'] = \
            case_thermal_df.loc[case_thermal_df['Resource'] == resource, 'Fixed_OM_Cost_per_MWyr'] * row['Fixed_OM_Cost_per_MWyr_factor']
        case_thermal_df.loc[case_thermal_df['Resource'] == resource, 'Var_OM_Cost_per_MWh'] = \
            case_thermal_df.loc[case_thermal_df['Resource'] == resource, 'Var_OM_Cost_per_MWh'] * row['Var_OM_Cost_per_MWh_factor']
        
    # create a cem copy of the case_thermal_df
    cem_case_thermal_df = case_thermal_df.copy()
    # set 'New_Build' to 1
    cem_case_thermal_df['New_Build'] = 1
    # set 'Can_Retire' to -1
    cem_case_thermal_df['Can_Retire'] = 0

    # create a lac copy of the case_thermal_df
    lac_case_thermal_df = case_thermal_df.copy()
    # set 'New_Build' to -1
    lac_case_thermal_df['New_Build'] = -1
    # set 'Can_Retire' to -1
    lac_case_thermal_df['Can_Retire'] = 0

    # save the case_thermal_df to genx_cem_resources_path
    cem_case_thermal_df.to_csv(os.path.join(genx_cem_resources_path, 'Thermal.csv'), index=False)
    # save the case_thermal_df to spcm_lac_resources_path
    lac_case_thermal_df.to_csv(os.path.join(spcm_lac_resources_path, 'Thermal.csv'), index=False)


    # # sort the case_vre_df by 'Resource' alphabetically
    # sorted_case_thermal_df = case_thermal_df.sort_values(by='Resource')
    # sorted_cem_case_thermal_df = cem_case_thermal_df.sort_values(by='Resource')
    # sorted_lac_case_thermal_df = lac_case_thermal_df.sort_values(by='Resource')
    
    # # save the case_thermal_df to genx_cem_resources_path
    # sorted_cem_case_thermal_df.to_csv(os.path.join(genx_cem_resources_path, 'Thermal.csv'), index=False)
    # # save the case_thermal_df to spcm_lac_resources_path
    # sorted_lac_case_thermal_df.to_csv(os.path.join(spcm_lac_resources_path, 'Thermal.csv'), index=False)
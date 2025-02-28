import pandas as pd
import numpy as np
import os
import sys
from get_case_names import get_case_names


# Add the root directory (my_package) to sys.path so Python can find 'utils'
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # One level up from 'scripts'
sys.path.append(root_path)

# Now import from utils
from utils.sge_utils import get_paths
data_path = get_paths('data')
genx_research_path = get_paths('genx_research')
spcm_research_path = get_paths('spcm_research')
scenario_generation_path = get_paths('scenario_generation')

# define location of cost assumptions
generator_assumptions_path = os.path.join(data_path, 'cases')

# Get the list of all files in the generator_assumptions_path directory
case_names_list = get_case_names(generator_assumptions_path)

# a_upd_generator_df path
a_upd_generator_df_path = os.path.join(data_path, 'a_upd_generator_df.csv')
# read in upd_gen csv
upd_gen_df = pd.read_csv(a_upd_generator_df_path)
# read in data_source_comparisons
manual_db_rel = pd.read_csv(os.path.join(data_path,'manual_db_rel.csv'))
reqd_storage_data = [
                  'Resource',
                  'Zone',
                  'Model',
                  'New_Build',
                  'Can_Retire',
                  'Existing_Cap_MW',
                  'Existing_Cap_MWh',
                  'Max_Cap_MW',
                  'Max_Cap_MWh',
                  'Min_Cap_MW',
                  'Min_Cap_MWh',
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
                  'Reg_Max',
                  'Rsv_Max',
                  'Reg_Cost',
                  'Rsv_Cost',
                  'region',
                  'cluster',
]
# get list of 'Resources' in manual_db_rel that are storage 'ATB Technology Name'
storage_resources = manual_db_rel.loc[manual_db_rel['ATB Technology Name'] == 'Utility-Scale Battery Storage', 'Resource'].unique()

# get dataframe from upd_gen_df that has 'resource' in storage_resources
storage_df = upd_gen_df.loc[upd_gen_df['Resource'].isin(storage_resources)]
storage_df
reqd_storage_df = storage_df[reqd_storage_data]

for case_name in case_names_list:
# for case_name in case_names_list[0:1]:
    # load cem and lac paths
    genx_cem_resources_path = os.path.join(genx_research_path, case_name, 'resources')
    spcm_lac_resources_path = os.path.join(spcm_research_path, case_name, 'resources')

    # read in cost assumption in generator_assumptions_path for case
    case_assumptions = pd.read_csv(os.path.join(generator_assumptions_path, case_name + '.csv'))

    # create a new copy of the reqd_storage_df for each case
    case_storage_df = reqd_storage_df.copy()

    # only take the resources in case_storage_df that are in case_assumptions
    case_storage_df = case_storage_df[case_storage_df['Resource'].isin(case_assumptions['Technical Name'])]

    for index, row in case_assumptions.iterrows():
        resource = row['Technical Name']
        # multiply the cost in case_storage_df by corresponding factor in cost
                # multiply the cost in case_storage_df by the corresponding factor in cost_assumptions
        case_storage_df.loc[case_storage_df['Resource'] == resource, 'Inv_Cost_per_MWyr'] = \
            case_storage_df.loc[case_storage_df['Resource'] == resource, 'Inv_Cost_per_MWyr'] * row['Inv_Cost_per_MWyr_factor']
        case_storage_df.loc[case_storage_df['Resource'] == resource, 'Inv_Cost_per_MWhyr'] = \
            case_storage_df.loc[case_storage_df['Resource'] == resource, 'Inv_Cost_per_MWhyr'] * row['Inv_Cost_per_MWhyr_factor']
        case_storage_df.loc[case_storage_df['Resource'] == resource, 'Fixed_OM_Cost_per_MWyr'] = \
            case_storage_df.loc[case_storage_df['Resource'] == resource, 'Fixed_OM_Cost_per_MWyr'] * row['Fixed_OM_Cost_per_MWyr_factor']
        case_storage_df.loc[case_storage_df['Resource'] == resource, 'Fixed_OM_Cost_per_MWhyr'] = \
            case_storage_df.loc[case_storage_df['Resource'] == resource, 'Fixed_OM_Cost_per_MWhyr'] * row['Fixed_OM_Cost_per_MWhyr_factor']
        case_storage_df.loc[case_storage_df['Resource'] == resource, 'Var_OM_Cost_per_MWh'] = \
            case_storage_df.loc[case_storage_df['Resource'] == resource, 'Var_OM_Cost_per_MWh'] * row['Var_OM_Cost_per_MWh_factor']
        
    # create a cem copy of the case_storage_df
    cem_case_storage_df = case_storage_df.copy()
    # set 'New_Build' to 1
    cem_case_storage_df['New_Build'] = 1
    # set 'Can_Retire' to -1
    cem_case_storage_df['Can_Retire'] = 0

    # create a lac copy of the case_storage_df
    lac_case_storage_df = case_storage_df.copy()
    # set 'New_Build' to -1
    lac_case_storage_df['New_Build'] = -1
    # set 'Can_Retire' to -1
    lac_case_storage_df['Can_Retire'] = 0

    # save the case_storage_df to genx_cem_resources_path
    cem_case_storage_df.to_csv(os.path.join(genx_cem_resources_path, 'Storage.csv'), index=False)
    # save the case_storage_df to spcm_lac_resources_path
    lac_case_storage_df.to_csv(os.path.join(spcm_lac_resources_path, 'Storage.csv'), index=False)

    # # sort the case_storage_df by 'Resource' alphabetically
    # sorted_case_storage_df = case_storage_df.sort_values(by='Resource')
    # sorted_cem_case_storage_df = cem_case_storage_df.sort_values(by='Resource')
    # sorted_lac_case_storage_df = lac_case_storage_df.sort_values(by='Resource')

    # # # delete the existing storage.csv in genx_cem_resources_path
    # # if os.path.exists(os.path.join(genx_cem_resources_path, 'storage.csv')):
    # #     os.remove(os.path.join(genx_cem_resources_path, 'storage.csv'))
    # # save the case_storage_df to genx_cem_resources_path
    # sorted_cem_case_storage_df.to_csv(os.path.join(genx_cem_resources_path, 'Storage.csv'), index=False)
    # # save the case_storage_df to spcm_lac_resources_path
    # sorted_lac_case_storage_df.to_csv(os.path.join(spcm_lac_resources_path, 'Storage.csv'), index=False)
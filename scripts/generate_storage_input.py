import pandas as pd
import numpy as np
import os

# read in upd_gen csv
upd_gen_df = pd.read_csv('a_upd_generator_df.csv')
# read in data_source_comparisons
manual_db_rel = pd.read_csv(os.path.join('..', 'data','manual_db_rel.csv'))
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
# get list of 'Resources' in manual_db_rel that are storage 'Energy Type'
storage_resources = manual_db_rel.loc[manual_db_rel['Energy Type'] == 'Storage', 'Resource'].unique()

# get dataframe from upd_gen_df that has 'resource' in storage_resources
storage_df = upd_gen_df.loc[upd_gen_df['Resource'].isin(storage_resources)]
storage_df
reqd_storage_df = storage_df[reqd_storage_data]
# define location of cost assumptions
generator_assumptions_path = os.path.join('..', 'data', 'cases')
# define path locations for CEM and LACs where inputs are going
genx_cem_loc = os.path.join('..', 'GenX.jl', 'research_systems')
spcm_lac_loc = os.path.join('..', 'SPCM', 'research_systems')

# Get the list of all files in the generator_assumptions_path directory
case_names_list = []
for xlsx_name in os.listdir(generator_assumptions_path):
    if os.path.isfile(os.path.join(generator_assumptions_path, xlsx_name)):
        case_name = xlsx_name.replace('.xlsx', '')
        case_names_list.append(case_name)


for case_name in case_names_list:
# for case_name in case_names_list[0:1]:
    # load cem and lac paths
    genx_cem_resources_path = os.path.join(genx_cem_loc, case_name, 'resources')
    spcm_lac_resources_path = os.path.join(spcm_lac_loc, case_name, 'resources')

    # read in cost assumption in generator_assumptions_path for case
    case_assumptions = pd.read_excel(os.path.join(generator_assumptions_path, case_name + '.xlsx'))

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
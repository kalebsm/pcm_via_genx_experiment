import pandas as pd
import numpy as np
import os


# read in upd_gen csv
upd_gen_df = pd.read_csv('a_upd_generator_df.csv')
# read in data_source_comparisons
manual_db_rel = pd.read_csv(os.path.join('..', 'data','manual_db_rel.csv'))
reqd_vre_data = [
                  'Resource',
                  'Zone',
                  'Num_VRE_Bins',
                  'New_Build',
                  'Can_Retire',
                  'Existing_Cap_MW',
                  'Max_Cap_MW',
                  'Min_Cap_MW',
                  'Inv_Cost_per_MWyr',
                  'Fixed_OM_Cost_per_MWyr',
                  'Var_OM_Cost_per_MWh',
                  'Reg_Max',
                  'Rsv_Max',
                  'Reg_Cost',
                  'Rsv_Cost',
                  'region',
                  'cluster',
]
# get list of 'Resources' in manual_db_rel that are either Solar or Wind 'Energy Type'
vre_resources = manual_db_rel[manual_db_rel['Energy Type'].isin(['Vre'])]['Resource'].tolist()

# get dataframe from upd_gen_df that has 'Resource' in vre_resources
vre_df = upd_gen_df[upd_gen_df['Resource'].isin(vre_resources)]
vre_df
reqd_vre_df = vre_df[reqd_vre_data]
reqd_vre_df
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

    # create a new copy of the reqd_vre_df for each case
    case_vre_df = reqd_vre_df.copy()

    # only take the resources in case_vre_df that are in case_assumptions
    case_vre_df = case_vre_df[case_vre_df['Resource'].isin(case_assumptions['Technical Name'])]

    for index, row in case_assumptions.iterrows():
        resource = row['Technical Name']
        # multiply the cost in case_vre_df by corresponding factor in cost
                # multiply the cost in case_vre_df by the corresponding factor in cost_assumptions
        case_vre_df.loc[case_vre_df['Resource'] == resource, 'Inv_Cost_per_MWyr'] = \
            case_vre_df.loc[case_vre_df['Resource'] == resource, 'Inv_Cost_per_MWyr'] * row['Inv_Cost_per_MWyr_factor']
        case_vre_df.loc[case_vre_df['Resource'] == resource, 'Fixed_OM_Cost_per_MWyr'] = \
            case_vre_df.loc[case_vre_df['Resource'] == resource, 'Fixed_OM_Cost_per_MWyr'] * row['Fixed_OM_Cost_per_MWyr_factor']
        # case_vre_df.loc[case_vre_df['Resource'] == resource, 'Var_OM_Cost_per_MWh'] = \
        #     case_vre_df.loc[case_vre_df['Resource'] == resource, 'Var_OM_Cost_per_MWh'] * row['Var_OM_Cost_per_MWh_factor']
        
    # create a cem copy of the case_vre_df
    cem_case_vre_df = case_vre_df.copy()
    # set 'New_Build' to 1
    cem_case_vre_df['New_Build'] = 1
    # set 'Can_Retire' to -1
    cem_case_vre_df['Can_Retire'] = 0

    # create a lac copy of the case_vre_df
    lac_case_vre_df = case_vre_df.copy()
    # set 'New_Build' to -1
    lac_case_vre_df['New_Build'] = -1
    # set 'Can_Retire' to -1
    lac_case_vre_df['Can_Retire'] = 0

    # # delete the existing vre.csv in genx_cem_resources_path
    # if os.path.exists(os.path.join(genx_cem_resources_path, 'vre.csv')):
    #     os.remove(os.path.join(genx_cem_resources_path, 'vre.csv'))
    # save the case_vre_df to genx_cem_resources_path
    cem_case_vre_df.to_csv(os.path.join(genx_cem_resources_path, 'Vre.csv'), index=False)
    # save the case_vre_df to spcm_lac_resources_path
    lac_case_vre_df.to_csv(os.path.join(spcm_lac_resources_path, 'Vre.csv'), index=False)


    # # sort the case_vre_df by 'Resource' alphabetically
    # sorted_case_vre_df = case_vre_df.sort_values(by='Resource')
    # sorted_cem_case_vre_df = cem_case_vre_df.sort_values(by='Resource')
    # sorted_lac_case_vre_df = lac_case_vre_df.sort_values(by='Resource')

    # # # delete the existing vre.csv in genx_cem_resources_path
    # # if os.path.exists(os.path.join(genx_cem_resources_path, 'vre.csv')):
    # #     os.remove(os.path.join(genx_cem_resources_path, 'vre.csv'))
    # # save the case_vre_df to genx_cem_resources_path
    # sorted_cem_case_vre_df.to_csv(os.path.join(genx_cem_resources_path, 'Vre.csv'), index=False)
    # # save the case_vre_df to spcm_lac_resources_path
    # sorted_lac_case_vre_df.to_csv(os.path.join(spcm_lac_resources_path, 'Vre.csv'), index=False)

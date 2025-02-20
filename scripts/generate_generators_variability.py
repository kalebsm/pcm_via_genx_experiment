import pandas as pd
import numpy as np
import os

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


# read in data_source_comparisons
manual_db_rel = pd.read_csv(os.path.join('..', 'data','manual_db_rel.csv'))
# load in a_upd_generator.csv
full_generator_df = pd.read_csv('a_upd_generator_df.csv')
# load in ercot actuals data
ercot_actuals_loc = os.path.join('..', 'scenario_generation', 'sequential_NORTA', 'data')
ercot_actuals_df = pd.read_csv(ercot_actuals_loc + '/actuals_ercot2018.csv')

lac_length = len(ercot_actuals_df)
period = range(0,lac_length)
cem_length = lac_length - 50
# get energy type 

# create dataframe for cf data
cem_generators_variability = pd.DataFrame({'Time_Index': range(0,cem_length)})
lac_generators_variability = pd.DataFrame({'Time_Index': range(0,lac_length)})

# set unique energy types for resource
resource_types = ['Thermal', 'Vre', 'Storage'] # add other resources as needed e.g. Vre_Stor

# loop through resource types and get the variability data
for resource_type in resource_types:
    # extract resources of that type
    resources_of_type = manual_db_rel[manual_db_rel['Energy Type'] == resource_type]['Resource']
    # print(resources_of_type)
    if resource_type == 'Thermal':
        # thermal_resources = resources_of_type[resources_of_type['Energy Type'].str.contains('Thermal')]
        for gen_name in resources_of_type:
            cem_generators_variability[gen_name] = [1] * cem_length
            lac_generators_variability[gen_name] = [1] * lac_length
    
    # if name contains Wind, get wind
    if resource_type == 'Vre':
        wind_resources = resources_of_type[resources_of_type.str.contains('Wind')]
        for gen_name in wind_resources:
            cem_generators_variability[gen_name] = ercot_actuals_df['wind'][:cem_length]
            lac_generators_variability[gen_name] = ercot_actuals_df['wind'][:lac_length]

    # if name contains PV or Solar, get solar
    if resource_type == 'Vre':
        solar_resources = resources_of_type[resources_of_type.str.contains('PV|Solar')]
        for gen_name in solar_resources:
            cem_generators_variability[gen_name] = ercot_actuals_df['solar'][:cem_length]
            lac_generators_variability[gen_name] = ercot_actuals_df['solar'][:lac_length]

    # if name contains Storage, get storage
    if resource_type == 'Storage':
        # storage_resources = resources_of_type[resources_of_type['Resource'].str.contains('Storage|Battery')]
        for gen_name in resources_of_type:
            cem_generators_variability[gen_name] = [1] * cem_length
            lac_generators_variability[gen_name] = [1] * lac_length


for case_name in case_names_list:
# for case_name in case_names_list[0:1]:
    # print(case_name)
    # load cem and lac paths
    genx_cem_system_path = os.path.join(genx_cem_loc, case_name, 'system')
    spcm_lac_system_path = os.path.join(spcm_lac_loc, case_name, 'system')

    # read in cost assumption in generator_assumptions_path for case
    case_assumptions = pd.read_excel(os.path.join(generator_assumptions_path, case_name + '.xlsx'))

    # get set of resources in case assumptions
    resources = case_assumptions['Technical Name']

    # save to df
    #extract dataframe only of resrouces in case assumptions
    sorted_cem_generators_variability = cem_generators_variability[['Time_Index'] + list(resources)]
    sorted_lac_generators_variability = lac_generators_variability[['Time_Index'] + list(resources)]

    # # after time_index, sort other columns alphabetically
    # sorted_cem_generators_variability = cem_generators_variability[['Time_Index'] + sorted(cem_generators_variability.columns[1:])]
    # sorted_lac_generators_variability = lac_generators_variability[['Time_Index'] + sorted(lac_generators_variability.columns[1:])]

    # save to csv in cem and lac paths
    sorted_cem_generators_variability.to_csv(os.path.join(genx_cem_system_path, \
                                                   'Generators_variability.csv'), index=False)
    sorted_lac_generators_variability.to_csv(os.path.join(spcm_lac_system_path, \
                                                    'Generators_variability.csv'), index=False)

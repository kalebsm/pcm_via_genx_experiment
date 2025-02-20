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


# get set of regions
# XXX not implemented
# regions = set of regions
regions = 'ERCOT'

# create a dataframe of resources and 'Min_Cap_1', 'Min_Cap_2', and 'Min_Cap_3' columns
co2_caps = pd.DataFrame()
co2_caps[''] = regions
co2_caps['Network_zones'] = 'z1'
co2_caps['CO_2_Cap_Zone_1'] = 1
co2_caps['CO_2_Max_tons_MWh_1'] = 0.05
co2_caps['CO_2_Max_Mtons_1'] = 0.018

for case_name in case_names_list:
# for case_name in case_names_list[0:1]:
    # load cem and lac paths
    genx_cem_policies_path = os.path.join(genx_cem_loc, case_name, 'policies')
    spcm_lac_policies_path = os.path.join(spcm_lac_loc, case_name, 'policies')

    # get case assumptions
    case_assumptions = pd.read_excel(os.path.join(generator_assumptions_path, case_name + '.xlsx'))



    # save resource_min_caps to genx_cem_resources_path
    co2_caps.to_csv(os.path.join(genx_cem_policies_path, \
                            'CO2_cap.csv'), index=False)
    co2_caps.to_csv(os.path.join(spcm_lac_policies_path, \
                            'CO2_cap.csv'), index=False)
    # # remove Resource_minimum_capacity_requirements from spcm_lac_policies_path
    # os.remove(os.path.join(spcm_lac_policies_path, 'Resource_minimum_capacity_requirement.csv'))
    # # remove '.csv' from genx_cem_policies_path
    # os.remove(os.path.join(genx_cem_policies_path, '.csv'))


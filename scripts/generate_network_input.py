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
print(case_names_list)
# get set of regions
# XXX not implemented
# regions = set of regions
regions = 'ERCOT'
zones = 'z1'

# create dataframe for network
network = pd.DataFrame({
    '': [regions],
    'Network_zones': [zones],
    'Network_Lines': [1],
    'Start_Zone': [1],
    'End_Zone': [1],
    'Line_Max_Flow_MW': [2950],
    'transmission_path_name': ['ERCOT_to_ERCOT'],
    'distance_mile': [123.0584],
    'Line_Loss_Percentage': [0.012305837],
    'Line_Max_Reinforcement_MW': [2950],
    'Line_Reinforcement_Cost_per_MWyr': [12060],
    'DerateCapRes_1': [0.95],
    'CapREs_Excl_1': [0]
})


for case_name in case_names_list:
# for case_name in case_names_list[0:1]:
    # load cem and lac paths
    genx_cem_system_path = os.path.join(genx_cem_loc, case_name, 'system')
    spcm_lac_system_path = os.path.join(spcm_lac_loc, case_name, 'system')

        # save resource_min_caps to genx_cem_resources_path
    network.to_csv(os.path.join(genx_cem_system_path, \
                            'Network.csv'), index=False)
    network.to_csv(os.path.join(spcm_lac_system_path, \
                            'Network.csv'), index=False)
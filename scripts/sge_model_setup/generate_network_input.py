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
    genx_cem_system_path = os.path.join(genx_research_path, case_name, 'system')
    spcm_lac_system_path = os.path.join(spcm_research_path, case_name, 'system')

        # save resource_min_caps to genx_cem_resources_path
    network.to_csv(os.path.join(genx_cem_system_path, \
                            'Network.csv'), index=False)
    network.to_csv(os.path.join(spcm_lac_system_path, \
                            'Network.csv'), index=False)
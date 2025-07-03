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


for case_name in case_names_list:
# for case_name in case_names_list[0:1]:
    # load cem and lac paths
    genx_cem_resources_path = os.path.join(genx_research_path, case_name, 'resources')
    spcm_lac_resources_path = os.path.join(spcm_research_path, case_name, 'resources')

    # get case assumptions
    case_assumptions = pd.read_csv(os.path.join(generator_assumptions_path, case_name + '.csv'))

    # get set of resources in case assumptions
    resources = case_assumptions['Technical Name']

    # create a dataframe of resources and 'Min_Cap_1', 'Min_Cap_2', and 'Min_Cap_3' columns
    resource_min_caps = pd.DataFrame()
    resource_min_caps['Resource'] = resources
    resource_min_caps['Min_Cap_1'] = 0
    resource_min_caps['Min_Cap_2'] = 0
    resource_min_caps['Min_Cap_3'] = 0

    # save resource_min_caps to genx_cem_resources_path
    resource_min_caps.to_csv(os.path.join(genx_cem_resources_path, \
                            'policy_assignments', 'Resource_minimum_capacity_requirement.csv'), index=False)
    resource_min_caps.to_csv(os.path.join(spcm_lac_resources_path, \
                            'policy_assignments', 'Resource_minimum_capacity_requirement.csv'), index=False)


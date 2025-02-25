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


for case_name in case_names_list:
# for case_name in case_names_list[0:1]:
    # load cem and lac paths
    genx_cem_resources_path = os.path.join(genx_cem_loc, case_name, 'resources')
    spcm_lac_resources_path = os.path.join(spcm_lac_loc, case_name, 'resources')

    # get case assumptions
    case_assumptions = pd.read_excel(os.path.join(generator_assumptions_path, case_name + '.xlsx'))

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


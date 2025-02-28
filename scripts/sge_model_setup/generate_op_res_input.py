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


# create dataframe for network
op_res = pd.DataFrame({
    'Reg_Req_Percent_Demand': [0.01],
    'Reg_Req_Percent_VRE': [0.0032],
    'Rsv_Req_Percent_Demand': [0.033],
    'Rsv_Req_Percent_VRE': [0.0795],
    'Unmet_Rsv_Penalty_Dollar_per_MW': [1000],
    'Dynamic_Contingency': [0],
    'Static_Contingency_MW': [0]
})
for case_name in case_names_list:
# for case_name in case_names_list[0:1]:
    # load cem and lac paths
    genx_cem_system_path = os.path.join(genx_research_path, case_name, 'system')
    spcm_lac_system_path = os.path.join(spcm_research_path, case_name, 'system')

        # save resource_min_caps to genx_cem_resources_path
    op_res.to_csv(os.path.join(genx_cem_system_path, \
                            'Operational_reserves.csv'), index=False)
    op_res.to_csv(os.path.join(spcm_lac_system_path, \
                            'Operational_reserves.csv'), index=False)
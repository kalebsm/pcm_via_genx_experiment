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
    genx_cem_system_path = os.path.join(genx_cem_loc, case_name, 'system')
    spcm_lac_system_path = os.path.join(spcm_lac_loc, case_name, 'system')

        # save resource_min_caps to genx_cem_resources_path
    op_res.to_csv(os.path.join(genx_cem_system_path, \
                            'Operational_reserves.csv'), index=False)
    op_res.to_csv(os.path.join(spcm_lac_system_path, \
                            'Operational_reserves.csv'), index=False)
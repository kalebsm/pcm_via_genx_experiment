import pandas as pd
import numpy as np
import os
import sys
import yaml
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

data = {
    'OverwriteResults': 1,
    'PrintModel': 0,
    'NetworkExpansion': 0,
    'Trans_Loss_Segments': 1,
    'OperationalReserves': 1,
    'EnergyShareRequirement': 0,
    'CapacityReserveMargin': 0,
    'CO2Cap': 0,
    'StorageLosses': 0,
    'MinCapReq': 0,
    'MaxCapReq': 0,
    'Solver': 'Gurobi',
    'ParameterScale': 1,
    'WriteShadowPrices': 1,
    'UCommit': 2,
    'TimeDomainReductionFolder': 'TDR_Results',
    'TimeDomainReduction': 0,
    'ModelingToGenerateAlternatives': 0,
    'ModelingtoGenerateAlternativeSlack': 0.1,
    'ModelingToGenerateAlternativeIterations': 3,
    'MethodofMorris': 0,
    'OutputFullTimeSeries': 1
}


for case_name in case_names_list:
# for case_name in case_names_list[0:1]:
    # load cem and lac paths
    genx_cem_settings_path = os.path.join(genx_research_path, case_name, 'settings')
    spcm_lac_settings_path = os.path.join(spcm_research_path, case_name, 'settings')

    with open(os.path.join(genx_cem_settings_path, 'genx_settings.yml'), 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    
    with open(os.path.join(spcm_lac_settings_path, 'genx_settings.yml'), 'w') as file:
        yaml.dump(data, file, default_flow_style=False)



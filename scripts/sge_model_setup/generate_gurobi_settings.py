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
    'Feasib_Tol': 1.0e-05,          
    'Optimal_Tol': 1e-5,            
    'TimeLimit': 110000,            
    'Pre_Solve': 1,                
    'Method': 1,                    

    'MIPGap': 1e-3,                  
    'BarConvTol': 1.0e-08,         
    'NumericFocus': 0,              
    'Crossover': -1,                
    'PreDual': 0,                    
    'AggFill': 10,                  
}


for case_name in case_names_list:
# for case_name in case_names_list[0:1]:
    # load cem and lac paths
    genx_cem_settings_path = os.path.join(genx_research_path, case_name, 'settings')
    spcm_lac_settings_path = os.path.join(spcm_research_path, case_name, 'settings')

    with open(os.path.join(genx_cem_settings_path, 'gurobi_settings.yml'), 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    
    with open(os.path.join(spcm_lac_settings_path, 'gurobi_settings.yml'), 'w') as file:
        yaml.dump(data, file, default_flow_style=False)



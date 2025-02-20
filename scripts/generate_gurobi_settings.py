import pandas as pd
import numpy as np
import os
import yaml

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
import yaml

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
    genx_cem_settings_path = os.path.join(genx_cem_loc, case_name, 'settings')
    spcm_lac_settings_path = os.path.join(spcm_lac_loc, case_name, 'settings')

    with open(os.path.join(genx_cem_settings_path, 'gurobi_settings.yml'), 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    
    with open(os.path.join(spcm_lac_settings_path, 'gurobi_settings.yml'), 'w') as file:
        yaml.dump(data, file, default_flow_style=False)



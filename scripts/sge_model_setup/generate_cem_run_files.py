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


julia_code = """
using GenX
using Gurobi

run_genx_case!(dirname(@__FILE__), Gurobi.Optimizer)
"""

# with open('run_genx_case.jl', 'w') as file:
#     file.write(julia_code)


for case_name in case_names_list:
# for case_name in case_names_list[0:1]:
    # load cem and lac paths
    genx_cem_path = os.path.join(genx_research_path, case_name)
    spcm_lac_path = os.path.join(spcm_research_path, case_name)

    # # delete 'run_genx_case.jl' if it exists
    # if os.path.exists(os.path.join(genx_cem_path, 'run_genx_case.jl')):
    #     os.remove(os.path.join(genx_cem_path, 'run_genx_case.jl'))
    # # write the julia_code to genx cem path
    with open(os.path.join(genx_cem_path, 'Run.jl'), 'w') as file:
        file.write(julia_code)
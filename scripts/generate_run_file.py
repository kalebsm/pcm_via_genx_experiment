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
    genx_cem_path = os.path.join(genx_cem_loc, case_name)
    spcm_lac_path = os.path.join(spcm_lac_loc, case_name)

    # # delete 'run_genx_case.jl' if it exists
    # if os.path.exists(os.path.join(genx_cem_path, 'run_genx_case.jl')):
    #     os.remove(os.path.join(genx_cem_path, 'run_genx_case.jl'))
    # # write the julia_code to genx cem path
    with open(os.path.join(genx_cem_path, 'Run.jl'), 'w') as file:
        file.write(julia_code)
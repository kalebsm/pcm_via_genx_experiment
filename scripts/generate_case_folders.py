import pandas as pd
import numpy as np
import shutil
import os
import sys
from datetime import datetime as dt

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

# create a function that takes the current case and loc and generates folders: 
# policies, resources, settings, system
def create_case_folders(case_name, base_loc):
    folders = ['policies', 'resources', 'settings', 'system']
    for folder in folders:
        # print(base_loc, case_name, folder)
        folder_path = os.path.join(base_loc, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if folder == 'resources':
            policy_assignments_path = os.path.join(folder_path, 'policy_assignments')
            if not os.path.exists(policy_assignments_path):
                os.makedirs(policy_assignments_path)

# create folders in GenX and SPCM research systems for each case
for case_name in case_names_list:
    genx_cem_unabr_case_loc = os.path.join(genx_cem_loc, case_name)
    genx_cem_abbr_case_loc = os.path.join(genx_cem_loc, case_name + '_abbr')
    spcm_lac_case_loc = os.path.join(spcm_lac_loc, case_name)

    if not os.path.exists(genx_cem_unabr_case_loc):
        os.makedirs(genx_cem_unabr_case_loc)  
    create_case_folders(case_name, genx_cem_unabr_case_loc)
    if not os.path.exists(genx_cem_abbr_case_loc):
        os.makedirs(genx_cem_abbr_case_loc)
    create_case_folders(case_name, genx_cem_abbr_case_loc)
    if not os.path.exists(spcm_lac_case_loc):
        os.makedirs(spcm_lac_case_loc)
    create_case_folders(case_name, spcm_lac_case_loc)
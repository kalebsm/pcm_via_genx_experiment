import pandas as pd
import numpy as np
import shutil
import os
import sys
from datetime import datetime as dt
from get_case_names import get_case_names


# define location of cost assumptions
generator_assumptions_path = os.path.join('data', 'cases')

# generate research systems folder

# define path locations for CEM and LACs where inputs are going
genx_cem_loc = os.path.join('GenX.jl', 'research_systems')
spcm_lac_loc = os.path.join('SPCM', 'research_systems')

# Get the list of all files in the generator_assumptions_path directory
case_names_list = get_case_names(generator_assumptions_path)

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

    # create the unabbreviated case folder
    if not os.path.exists(genx_cem_unabr_case_loc):
        os.makedirs(genx_cem_unabr_case_loc) 
        # print message
        print(f"Created {genx_cem_unabr_case_loc} folder.")
    else:
        print(f"{genx_cem_unabr_case_loc} folder already exists.")

    create_case_folders(case_name, genx_cem_unabr_case_loc)

    # create the abbreviated case folder
    if not os.path.exists(genx_cem_abbr_case_loc):
        os.makedirs(genx_cem_abbr_case_loc)
        print(f"Created {genx_cem_abbr_case_loc} folder.")
    else:
        print(f"{genx_cem_abbr_case_loc} folder already exists.")

    create_case_folders(case_name, genx_cem_abbr_case_loc)

    # create the SPCM/LAC case folder
    if not os.path.exists(spcm_lac_case_loc):
        os.makedirs(spcm_lac_case_loc)
        print(f"Created {spcm_lac_case_loc} folder.")
    else:
        print(f"{spcm_lac_case_loc} folder already exists.")

    create_case_folders(case_name, spcm_lac_case_loc)


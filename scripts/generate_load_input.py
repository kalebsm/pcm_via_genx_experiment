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


# load in ercot actuals data
ercot_actuals_loc = os.path.join('scenario_generation', 'sequential_NORTA', 'data')
ercot_actuals_df = pd.read_csv(ercot_actuals_loc + '/actuals_ercot2018.csv')
lac_length = len(ercot_actuals_df)

period = range(0,lac_length)

cem_length = lac_length - 50
cem_length
cem_load_actuals = ercot_actuals_df['load'][range(0,cem_length)]
lac_load_actuals = ercot_actuals_df['load']
cem_load_data = pd.DataFrame({'Voll': [5000] + [None] * (cem_length - 1), 
                          'Demand_Segment': [1] + [None] * (cem_length -1),
                          'Cost_of_Demand_Curtailment_per_MW': [1] + [None] * (cem_length - 1),
                          'Max_Demand_Curtailment': [1] + [None] * (cem_length - 1),
                          '$/MWh': [2000] + [None] * (cem_length - 1),
                          'Rep_Periods': [1] + [None] * (cem_length-1),
                          'Timesteps_per_Rep_Period': [cem_length] + [None] * (cem_length - 1),
                          'Sub_Weights': [cem_length] + [None] * (cem_length - 1),
                          'Time_Index': range(1,cem_length+1),
                          'Demand_MW_z1': cem_load_actuals})
lac_load_data = pd.DataFrame({'Voll': [5000] + [None] * (lac_length - 1), 
                          'Demand_Segment': [1] + [None] * (lac_length -1),
                          'Cost_of_Demand_Curtailment_per_MW': [1] + [None] * (lac_length - 1),
                          'Max_Demand_Curtailment': [1] + [None] * (lac_length - 1),
                          '$/MWh': [2000] + [None] * (lac_length - 1),
                          'Rep_Periods': [1] + [None] * (lac_length-1),
                          'Timesteps_per_Rep_Period': [cem_length] + [None] * (lac_length - 1),
                          'Sub_Weights': [cem_length] + [None] * (lac_length - 1),
                          'Time_Index': range(1,lac_length+1),
                          'Demand_MW_z1': lac_load_actuals})


for case_name in case_names_list:
# for case_name in case_names_list[0:1]:
    # load cem and lac paths
    genx_cem_system_path = os.path.join(genx_cem_loc, case_name, 'system')
    spcm_lac_system_path = os.path.join(spcm_lac_loc, case_name, 'system')

        # save resource_min_caps to genx_cem_resources_path
    cem_load_data.to_csv(os.path.join(genx_cem_system_path, \
                            'Demand_data.csv'), index=False)
    lac_load_data.to_csv(os.path.join(spcm_lac_system_path, \
                            'Demand_data.csv'), index=False)
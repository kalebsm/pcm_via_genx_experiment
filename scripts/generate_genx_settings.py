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
    genx_cem_settings_path = os.path.join(genx_cem_loc, case_name, 'settings')
    spcm_lac_settings_path = os.path.join(spcm_lac_loc, case_name, 'settings')

    with open(os.path.join(genx_cem_settings_path, 'genx_settings.yml'), 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    
    with open(os.path.join(spcm_lac_settings_path, 'genx_settings.yml'), 'w') as file:
        yaml.dump(data, file, default_flow_style=False)



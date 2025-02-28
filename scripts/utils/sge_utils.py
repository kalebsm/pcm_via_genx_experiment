import os
from pathlib import Path

def get_paths(path_type):
    env_var_map = {
        'scripts': 'SCRIPTS_PATH',
        'data': 'DATA_PATH',
        'genx_research': 'GENX_RESEARCH_PATH',
        'spcm_research': 'SPCM_RESEARCH_PATH',
        'figures': 'FIGURES_PATH',
        'atb-calc': 'ATB_CALC_PATH',
        'scenario_generation': 'SCENARIO_GENERATION_PATH'
    }
    
    default_path_map = {
        'scripts': Path(__file__).parents[2] / 'scripts',
        'data': Path(__file__).parents[2] / 'data',
        'genx_research': Path(__file__).parents[2] / 'GenX.jl' / 'research_systems',
        'spcm_research': Path(__file__).parents[2] / 'SPCM' / 'research_systems',
        'figures': Path(__file__).parents[2] / 'figures',
        'atb-calc': Path(__file__).parents[2] / 'ATB-calc',
        'scenario_generation': Path(__file__).parents[2] / 'scenario_generation'
    }
    
    env_var = env_var_map.get(path_type)
    if env_var:
        path = os.getenv(env_var)
        if path:
            return Path(path)
    
    return default_path_map.get(path_type)

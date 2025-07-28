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

model_types = ['pf', 'dlac-p', 'dlac-i', 'slac']


julia_code = r"""
# Load the dev-tracked package
using SequentialNorta
# Load the required packages
using Gurobi
using YAML
using BenchmarkTools
using JuMP
using DataFrames
using CSV
using DelimitedFiles
using Distributions
using Random
using HDF5
using Plots
using TimeZones
using TSFrames
using Tables
using LinearSolve
using LinearAlgebra
using Dates
using DelimitedFiles
using StatsBase

case = dirname(@__FILE__)
folder_name_parts = split(case, Sys.iswindows() ? "\\" : "/")
case_name = folder_name_parts[end]

# determine model type from file name after Run_
model_type = split(split(basename(@__FILE__), ".jl")[1], "_")[3]

test_dictionary = Dict(
    "test_scenario_path" => 1,
    "test_scenario_lookahead_path" => 0,
    "test_prices_scen_path" => 1
)

model_context = initialize_policy_model(case);

model_ep, inputs, context, results = run_policy_instance(model_context, model_type, test_dictionary; write_results=true);
"""

# with open('run_genx_case.jl', 'w') as file:
#     file.write(julia_code)


for case_name in case_names_list:
# for case_name in case_names_list[0:1]:
    # load cem and lac paths
    genx_cem_path = os.path.join(genx_research_path, case_name)
    spcm_lac_path = os.path.join(spcm_research_path, case_name)

    for model in model_types:
        # # delete 'run_genx_case.jl' if it exists
        # if os.path.exists(os.path.join(genx_cem_path, 'run_genx_case.jl')):
        #     os.remove(os.path.join(genx_cem_path, 'run_genx_case.jl'))
        # # write the julia_code to genx cem path
        with open(os.path.join(spcm_lac_path, f'Run_spcm_{model}.jl'), 'w') as file:
            file.write(julia_code)
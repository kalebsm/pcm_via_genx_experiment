import pandas as pd
import numpy as np
import seaborn as sb
import datetime
import os
import sys
from get_unique_resources_data import get_unique_resources_data
import matplotlib.pyplot as plt

case_names = [    
              "Thermal_Base",
              "2_Hr_BESS", 
              "2_Hr_BESS_Fuelx2",
              "4_Hr_BESS",
              "4_Hr_BESS_Fuelx2",
              "4_Hr_BESS_Fuelx3",
              "4_Hr_BESS_Fuelx4",
              "6_Hr_BESS",
              "6_Hr_BESS_Fuelx2",
              "8_Hr_BESS",
              "8_Hr_BESS_Fuelx2",
              "10_Hr_BESS",
              "10_Hr_BESS_Fuelx2",
              ]

current_dir = os.getcwd()
print(current_dir)

plots_path = os.path.join(current_dir, 'plots') + "/"
tables_path = os.path.join(current_dir, 'tables') + "/"
latex_path = os.path.join(current_dir, 'latex') + "/"
if not os.path.exists(plots_path):
    os.makedirs(plots_path)
if not os.path.exists(tables_path):
    os.makedirs(tables_path)
if not os.path.exists(latex_path):
    os.makedirs(latex_path)

cem_path = os.path.join(os.path.dirname(current_dir), 'GenX.jl', 'research_systems')
policies_path = os.path.join(os.path.dirname(current_dir), 'SPCM', 'research_systems')

date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# get unique generator names
unique_resources, cases_resources_capacities = get_unique_resources_data(case_names, policies_path)

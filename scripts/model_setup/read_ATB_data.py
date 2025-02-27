
import pandas as pd
import numpy as np
import shutil
import os
import sys
from datetime import datetime as dt
from get_case_names import get_case_names

# Add the path to the ATB-calc directory to the system path
sys.path.insert(0, os.path.join('ATB-calc'))
from lcoe_calculator.process_all import ProcessAll
from lcoe_calculator.tech_processors import (ALL_TECHS,
    OffShoreWindProc, LandBasedWindProc, DistributedWindProc,
    UtilityPvProc, CommPvProc, ResPvProc, UtilityPvPlusBatteryProc,
    CspProc, GeothermalProc, HydropowerProc, PumpedStorageHydroProc,
    CoalProc, NaturalGasProc, NuclearProc, BiopowerProc,
    UtilityBatteryProc, CommBatteryProc, ResBatteryProc,
    CoalRetrofitProc, NaturalGasRetrofitProc, NaturalGasFuelCellProc)

from IPython.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))
if 'atb_df' not in globals() or atb_df is None:
    # The below line MUST be updated to reflect the location of the ATB workbook on your computer
    # atb_electricity_workbook = 'C:\\Users\\ks885\Documents\\aa_research\Data\Energy\\NRELs_ATB\ATB-calc\\data\\2023-ATB-Data_Master_v9.0.xlsx'
    atb_electricity_workbook = os.path.join('data', '2024 v2 Annual Technology Baseline Workbook Errata 7-19-2024.xlsx')
    # atb_electricity_workbook = os.path.join('..', 'data', '2024 v1 Annual Technology Baseline Workbook Original 6-24-2024.xlsx')
    # ---- Comment/uncomment the below lines to process all techs or a subset of techs
    # Process all technologies
    techs = [LandBasedWindProc, UtilityPvProc, CoalProc, NaturalGasProc, UtilityBatteryProc]

    # Initiate the processor with the workbook location and desired technologies
    processor = ProcessAll(atb_electricity_workbook, techs)

    start = dt.now()
    processor.process()
    print('Processing completed in ', dt.now() - start)

    atb_df = processor.data

# a_upd_generator_df path
a_upd_generator_df_path = os.path.join('data', 'a_upd_generator_df.csv')
# load in generator param data shell
upd_gen_df = pd.read_csv(a_upd_generator_df_path)
atb_upd_gen_df = upd_gen_df.copy()

# define location of cost assumptions
generator_assumptions_path = os.path.join('data', 'cases')
# Get the list of all files in the generator_assumptions_path directory
case_names_list = get_case_names(generator_assumptions_path)



# # load initialized generator df csv
# gen_2_model_df = pd.read_csv(os.path.join('data', 'a_initialized_generator_df.csv'))
gen_2_model_names = atb_upd_gen_df['Resource']


# so then we need to go through unique gen names, get technology type associated, 
# get list of gen names in ATB
atb_gen_names = atb_df['DisplayName'].unique()


# find the intersection of the two lists
inters_names = list(set(gen_2_model_names).intersection(set(atb_gen_names)))


assumed_case = 'Market'
assumed_crpyears = 20
assumed_scenario =  'Moderate'
assumed_year = 2022
assumed_interest = 0.8

# write dictionary of parameters in GenX title vs ATB title
gen_2_atb_dict = {
    "Inv_Cost_per_MWyr": "CAPEX",
    "Fixed_OM_Cost_per_MWyr": "Fixed O&M",
    "Var_OM_Cost_per_MWh": "Variable O&M",
    "Heat Rate": "Heat Rate",
}
# for gen_name in inters_names[0:1]:
for gen_name in inters_names:
    # get technology type
    tech_type = atb_df[atb_df['DisplayName'] == gen_name]['Technology'].values[0]
    
    atb_gen_df = atb_df[(atb_df['Case'] == assumed_case) 
                        & (atb_df['CRPYears'] == assumed_crpyears) 
                         & (atb_df['Scenario'] == assumed_scenario) 
                         & (atb_df['DisplayName'] == gen_name)]


    inv_cost_kilowatt = atb_gen_df[atb_gen_df['Parameter'] == 'CAPEX'][assumed_year].values[0]
    fixed_om_kilowatt = atb_gen_df[atb_gen_df['Parameter'] == 'Fixed O&M'][assumed_year].values[0]
    var_om_mw = atb_gen_df[atb_gen_df['Parameter'] == 'Variable O&M'][assumed_year].values[0]


    Inv_Cost_per_MWyr = inv_cost_kilowatt * 1000 / assumed_crpyears
    Fixed_OM_cost_per_MWyr = fixed_om_kilowatt * 1000
    Var_OM_Cost_per_MWh = var_om_mw


    Heat_Rate_MMBTU_per_MWh = None
    # get into resource specific costs
    if tech_type == 'Utility-Scale Battery Storage':
        pass
    elif tech_type == 'UtilityPV':
        Var_OM_Cost_per_MWh = 0.1
    elif tech_type == 'LandbasedWind':
        Var_OM_Cost_per_MWh = 0.1
    elif tech_type == 'Coal_FE' or tech_type == 'NaturalGas_FE':
        Heat_Rate_MMBTU_per_MWh = atb_gen_df[atb_gen_df['Parameter'] == 'Heat Rate'][assumed_year].values[0]
    else:
        raise ValueError(f"Unhandled technology type: {tech_type}")
    
    atb_upd_gen_df.loc[atb_upd_gen_df['Resource'] == gen_name, 'Inv_Cost_per_MWyr'] = Inv_Cost_per_MWyr
    atb_upd_gen_df.loc[atb_upd_gen_df['Resource'] == gen_name,'Fixed_OM_Cost_per_MWyr'] = Fixed_OM_cost_per_MWyr
    atb_upd_gen_df.loc[atb_upd_gen_df['Resource'] == gen_name,'Var_OM_Cost_per_MWh'] = Var_OM_Cost_per_MWh
    atb_upd_gen_df.loc[atb_upd_gen_df['Resource'] == gen_name,'Heat_Rate_MMBTU_per_MWh'] = Heat_Rate_MMBTU_per_MWh

    # save the 
    # atb_gen_df = atb_df[(atb_df['DisplayName'] == gen_name)]

# print out updated generator df
atb_upd_gen_df.to_csv(a_upd_generator_df_path, index=False)


# print completion message
print('Updated generator df saved to: ', a_upd_generator_df_path)
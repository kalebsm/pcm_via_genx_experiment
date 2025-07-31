import pandas as pd
import numpy as np
import h5py
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

# a_upd_generator_df path
a_upd_generator_df_path = os.path.join(data_path, 'a_upd_generator_df.csv')

# load in a_upd_generator.csv
full_generator_df = pd.read_csv(a_upd_generator_df_path)

# load in ercot actuals data
ercot_actuals_loc = os.path.join(scenario_generation_path, 'sequential_norta', 'data')
ercot_actuals_df = pd.read_csv(ercot_actuals_loc + '/actuals_ercot2018.csv')

# with h5py.File(os.path.join(ercot_actuals_loc, 'BA_load_actuals_2018.h5'), 'r') as f:
#     df_meta = pd.DataFrame(f['meta'][...])
#     time_index = pd.to_datetime(f['time_index'][...].astype(str))
#     load_actuals = f['actuals'][...]
#     # Assuming the array is 1-dimensional
#     ercot_actuals_df = pd.DataFrame({'time_index': time_index, 'load': load_actuals})


lac_length = len(ercot_actuals_df)
period = range(0,lac_length)
decision_length = lac_length - 50

# for some reason, fuel lengths need to be one longer than the actual data
cem_fuel_length = decision_length + 1
lac_fuel_length = decision_length + 1
# read in data_source_comparisons
manual_db_rel = pd.read_csv(os.path.join(data_path,'manual_db_rel.csv'))

for case_name in case_names_list:
# for case_name in case_names_list[0:1]:
    # load cem and lac paths
    genx_cem_system_path = os.path.join(genx_research_path, case_name, 'system')
    spcm_lac_system_path = os.path.join(spcm_research_path, case_name, 'system')

    # read in cost assumption in generator_assumptions_path for case
    case_assumptions = pd.read_csv(os.path.join(generator_assumptions_path, case_name + '.csv'))

    # get set of resources in case assumptions
    resources = case_assumptions['Technical Name']
    # get fuel types for each resource from full_generator_df
    fuel_types = full_generator_df[full_generator_df['Resource'].isin(resources)]['Fuel'].unique()
    # replace nan with 'None'
    fuel_types = np.where(pd.isna(fuel_types), 'None', fuel_types)

    # create a dataframe to put fuel timeseries in
    cem_fuel_timeseries = pd.DataFrame()
    lac_fuel_timeseries = pd.DataFrame()

    cem_fuel_timeseries['Time_Index'] = range(0, cem_fuel_length)
    lac_fuel_timeseries['Time_Index'] = range(0, lac_fuel_length)

    # find the first_fuel_cost_factor in case_assumptions for each fuel type
    first_fuel_cost_factors = {}
    for fuel in fuel_types:
        if fuel == 'None':
            first_fuel_cost_factors[fuel] = 0
            continue
        else:
            first_occurence_idx = manual_db_rel[manual_db_rel['Fuel ID'] == fuel].index[0]
        # get generator name at first occurrence index
        first_gen_name = manual_db_rel.at[first_occurence_idx, 'Resource']

        first_fuel_cost_factors[fuel] = case_assumptions[case_assumptions['Technical Name'] == first_gen_name]['Fuel_Cost_factor'].values[0]

    
    # use hardcoded fuel costs for now
    fuel_cost_df = pd.DataFrame({'BIT': 2.37 * first_fuel_cost_factors['BIT'],
                                'NG': 5.28 * first_fuel_cost_factors['NG'],
                                'DFO': 6 * first_fuel_cost_factors['DFO'],
                                'UM': 0.86 * first_fuel_cost_factors['UM'],
                                'None': 0}, index=[0])
    
    for fuel in fuel_types:
        # get a timeseries of fuel costs at for fuel column using fuel_cost_df at length of cem and lac
        cem_fuel_timeseries[fuel] = [fuel_cost_df.at[0, fuel]] * (cem_fuel_length) # for some reason, needs one extra
        lac_fuel_timeseries[fuel] = [fuel_cost_df.at[0, fuel]] * (lac_fuel_length)


        # save resource_min_caps to genx_cem_resources_path
    cem_fuel_timeseries.to_csv(os.path.join(genx_cem_system_path, \
                            'Fuels_data.csv'), index=False)
    lac_fuel_timeseries.to_csv(os.path.join(spcm_lac_system_path, \
                            'Fuels_data.csv'), index=False)
    
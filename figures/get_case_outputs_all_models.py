import os
import pandas as pd
import numpy as np
from get_decision_variable_map import get_decision_variable_map

def get_case_outputs_all_models(cem_path, policies_path, case_name, decision_variable_name, model_types_list):
    cem_results_path = os.path.join(cem_path, case_name, 'results')

    # Create a DataFrame with decision_variable_names as the index
    decision_variable_map = get_decision_variable_map()

    # Get the corresponding file names from the decision_variable_map
    cem_output_name = decision_variable_map.loc[decision_variable_name, 'CEM']
    policies_file_name = decision_variable_map.loc[decision_variable_name, 'Policies']

    if cem_output_name is None or policies_file_name is None:
        raise ValueError(f"File names for decision variable '{decision_variable_name}' are not defined in the map.")

    # Read and process CEM data
    cem_file_path = os.path.join(cem_results_path, cem_output_name + '.csv')
    cem_data_raw = pd.read_csv(cem_file_path)
    # if 'Zone' and 'AnnualSum' are in the first rows of the first column then remove the first two rows
    if 'Zone' in cem_data_raw.iloc[:, 0].values \
                and 'AnnualSum' in cem_data_raw.iloc[:, 0].values:
        cem_data = cem_data_raw.drop([0,1])  # Remove the first two rows
    elif 'Zone' in cem_data_raw.iloc[:, 0].values:
        cem_data = cem_data_raw.drop([0])  # Remove the first row
    else:
        # print("not dropping any rows")
        cem_data = cem_data_raw

    cem_data.reset_index(drop=True, inplace=True)

    # Read and process Policies data for each model type
    policies_data_dict = {}
    for model_type in model_types_list:
        policies_results_path = os.path.join(policies_path, case_name, f'results_{model_type}')
        policies_file_path = os.path.join(policies_results_path, policies_file_name + '.csv')
        policies_data = pd.read_csv(policies_file_path)
        # if shape of policies_dadta is has more columns than rows, transpose it
        if policies_data.shape[0] < policies_data.shape[1]:
            policies_data = policies_data.transpose()  # Transpose the dataframe
        else:
            # if shape of policies_data is has more rows than columns, dont transpose?
            pass
        # Add the resource names to the policies data
        # policies_data.columns = resource_list
        policies_data_dict[model_type] = policies_data

    # find the names of columns that are in both cem_data and policies_data
    columns_2_print = list(set(cem_data.columns).intersection(set(policies_data_dict[model_types_list[0]].columns)))

    # take only columns that are in both cem_data and policies_data
    cem_data = cem_data[columns_2_print]
    for model_type in model_types_list:
        policies_data_dict[model_type] = policies_data_dict[model_type][columns_2_print]

    return cem_data, policies_data_dict, columns_2_print
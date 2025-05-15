
import pandas as pd
import numpy as np

def get_unique_resources_data(case_names, policies_path):
    """
    Extracts and returns a set of unique technologies from the provided data.

    Parameters:
        data (list): A list of dictionaries or objects containing technology information.

    Returns:
        unique_resources: A set of unique technologies.
        cases_resources_capacities: a dataframe of the cases and unique resource capacities in GW

    """
    # initialize list of unique resources
    unique_resources = []

    # initialize dataframe of cases and generator capacity
    cases_resources_capacities = pd.DataFrame({'Case Name': case_names})

    # find the set of resources in the 4 hour batteries cases
    for case_name in case_names:
        # read in the generator characteristics
        # load generator characteristics from resources folder
        thermal_dfGen = pd.read_csv(policies_path + '\\' + case_name + '\\resources' + '\\Thermal.csv')
        vre_dfGen = pd.read_csv(policies_path + '\\' + case_name + '\\resources' + '\\Vre.csv')
        storage_dfGen = pd.read_csv(policies_path + '\\' + case_name + '\\resources' + '\\Storage.csv')

        # combine all resources to dfGen
        dfGen = pd.concat([thermal_dfGen, vre_dfGen, storage_dfGen], ignore_index=True)
        gen_capacity_gw = dfGen['Existing_Cap_MW'] / 1000

        # find the index of the storage in the generator list
        resources = dfGen['Resource'].unique()

        for resource in resources:
            if resource not in unique_resources:
                print('adding resource: ' + resource + ' from case: ' + case_name)
                unique_resources.append(resource)
                # add the resource to the dataframe as a column if it doesn't exist
                if resource not in cases_resources_capacities.columns:
                    cases_resources_capacities[resource] = 0
            # add the capacity to the dataframe
            # find the index of the resource in the dataframe
            resource_index = dfGen[dfGen['Resource'] == resource].index[0]
            # update the capacity for the current case and resource
            cases_resources_capacities.loc[cases_resources_capacities['Case Name'] == case_name, resource] = gen_capacity_gw[resource_index]
                
    # end for loop



    return unique_resources, cases_resources_capacities
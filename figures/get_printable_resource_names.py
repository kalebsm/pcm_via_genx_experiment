import pandas as pd
import numpy as np

def get_printable_resource_names(data):

    """
    Replace values in a specified column of a DataFrame based on a dictionary.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The column in which to replace values.
        replace_dict (dict): A dictionary where keys are old values and values are new values.

    Returns:
        pd.DataFrame: The DataFrame with replaced values.
    """

    # Create a dictionary from the replace statements
    simplify_resource_names = {
        "Coal-new": "Coal",
        "NG Combustion Turbine (F-Frame)": "NG CT",
        "NG 2-on-1 Combined Cycle (F-Frame)": "NG CC",
        "NG 2-on-1 Combined Cycle (H-Frame) 95% CCS": "NG CC",
        "Utility PV - Class 1": "Solar",
        "Land-Based Wind - Class 1 - Technology 1": "Wind",
        "PBR-HTGR": "SMR",
        "iPWR_Pack": "SMR",
        "HGTR": "SMR",
        "Microreactor": "SMR",
        "RICE": "IC",
        "Utility-Scale Battery Storage - 2Hr": "BESS",
        "Utility-Scale Battery Storage - 4Hr": "BESS",
        "Utility-Scale Battery Storage - 6Hr": "BESS",
        "Utility-Scale Battery Storage - 8Hr": "BESS",
        "Utility-Scale Battery Storage - 10Hr": "BESS",
    }
    # # if in elements
    # df = df.replace(simplify_resource_names)
    # print("Simplifying resource names")

    data_copy = data.copy()

    # new_df = df_copy.assign(**{col: df_copy[col].replace(simplify_resource_names) for col in df_copy.columns})
    if type(data) == pd.DataFrame:
        # print("Data is a DataFrame")
        data_copy = data_copy.rename(columns=simplify_resource_names)
        # print(data_copy.columns)
    elif type(data) == dict:
        # print("Data is a dict")
        for key, value in data.items():
            new_key = simplify_resource_names.get(key, key)
            if isinstance(value, dict):
                data_copy[new_key] = {
                simplify_resource_names.get(k, k): v for k, v in value.items()}
            elif isinstance(value, pd.DataFrame):
                data_copy[new_key] = value.rename(columns=simplify_resource_names)
            else:
                data_copy[new_key] = value
            if new_key != key:
                del data_copy[key]
    elif type(data) == list:
        data_copy = [simplify_resource_names.get(x, x) for x in data]
        # get unique set but keep the order
        data_copy = list(dict.fromkeys(data_copy))
    elif type(data) == np.ndarray:
        data_copy = np.array([simplify_resource_names.get(x, x) for x in data])
        # get unique set but keep the order
        _, idx = np.unique(data_copy, return_index=True)
        data_copy = data_copy[np.sort(idx)]
    else:
        raise ValueError("Input data must be a DataFrame or a list.")


    return data_copy
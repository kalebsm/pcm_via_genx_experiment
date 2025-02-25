import pandas as pd
import numpy as np
import shutil
import os
import sys
from datetime import datetime as dt

def get_case_names(generator_assumptions_path):
    """
    Get the list of case names from the generator assumptions path.

    Parameters:
    generator_assumptions_path (str): The path to the directory containing the generator assumptions CSV files.

    Returns:
    list: A list of case names.
    """
    case_names_list = []
    for csv_name in os.listdir(generator_assumptions_path):
        if os.path.isfile(os.path.join(generator_assumptions_path, csv_name)):
            case_name = csv_name.replace('.csv', '')
            case_names_list.append(case_name)
    return case_names_list

# Example usage
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_case_names.py <generator_assumptions_path>")
        sys.exit(1)
    generator_assumptions_path = sys.argv[1]
    case_names = get_case_names(generator_assumptions_path)
    print(case_names)
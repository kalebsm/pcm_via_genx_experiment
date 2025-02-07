import numpy as np
import pandas as pd



def print_prime_mover_description(string, prime_mover_codes):
    description = prime_mover_codes.loc[prime_mover_codes['Prime Mover Codes'] == string, 'Description'].values
    if len(description) > 0:
        print(description[0])
    else:
        print("No description found for the given string.")

def print_fuel_codes_description(string, fuel_codes):
    description = fuel_codes.loc[fuel_codes['Energy Source Code'] == string, 'Description'].values
    if len(description) > 0:
        print(description[0])
    else:
        print("No description found for the given string.")


# Get the sheet names
excel_file = pd.ExcelFile('20120724-4012_Generator_Data_Summer.xlsx')
sheet_names = excel_file.sheet_names

# Create a dictionary to store the separated dataframes
separated_dataframes = {}

# Iterate over each sheet and store the data in the dictionary
for sheet_name in sheet_names:
    separated_dataframes[sheet_name] = excel_file.parse(sheet_name)


# Process for Cleaning Generator Characteristics
gen_characteristics = separated_dataframes['Generator Characteristics']
gen_characteristics.columns = separated_dataframes['Generator Characteristics'].iloc[0]
gen_characteristics.drop(0, inplace=True)
gen_characteristics.reset_index(drop=True, inplace=True)

# Process for Cleaning Generator costs
gen_costs = separated_dataframes['Generator Offer Curve'].iloc[:,24:28]
gen_costs.columns = gen_costs.iloc[1]
gen_costs.drop([0,1], inplace=True)
gen_costs.reset_index(drop=True, inplace=True)
gen_costs.rename(columns={'1': 'index'}, inplace=True)
gen_costs.rename(columns={np.nan: 'Generic Name'}, inplace=True)
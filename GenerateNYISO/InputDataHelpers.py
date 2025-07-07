"""
Input Data Helpers

Functions to help format input data from NYGrid base case
"""

import os
import csv
import shutil
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def create_nuclear_hourly_timeseries(data_folder):
    """
    Process daily nuclear generation data to create an hourly timeseries
    for nuclear generators in zones B and C using weighted average of capacity factors
    """

    nuclear_daily_file = os.path.join(data_folder, "nuclearGenDaily_2019.csv")
    
    print(f"Reading nuclear daily generation data from {nuclear_daily_file}...")
    nuclear_df = pd.read_csv(nuclear_daily_file)
    
    # Convert timestamp to datetime
    nuclear_df['TimeStamp'] = pd.to_datetime(nuclear_df['TimeStamp'])
    
    # Nuclear plant information - assign each plant to its zone and capacity
    nuclear_plants = {
        'FitzPatrick': {'Zone': 'C', 'Capacity': 842.0},     # Nine Mile Point area, Zone C
        'Ginna': {'Zone': 'B', 'Capacity': 582.0},           # Near Rochester, Zone B
        'IndianPoint2': {'Zone': 'H', 'Capacity': 1020.0},   # Buchanan, Zone H
        'IndianPoint3': {'Zone': 'H', 'Capacity': 1040.0},   # Buchanan, Zone H
        'NineMilePoint1': {'Zone': 'C', 'Capacity': 621.0},  # Oswego, Zone C
        'NineMilePoint2': {'Zone': 'C', 'Capacity': 1267.0}  # Oswego, Zone C
    }
    
    # Get plants in zones B and C
    zone_b_plants = [name for name, info in nuclear_plants.items() if info['Zone'] == 'B']
    zone_c_plants = [name for name, info in nuclear_plants.items() if info['Zone'] == 'C']
    
    print(f"Zone B plants: {zone_b_plants}")
    print(f"Zone C plants: {zone_c_plants}")
    
    # Function to expand daily data to hourly
    def expand_to_hourly(df):
        # Create a list to hold hourly data
        hourly_data = []
        
        # For each day in the input dataframe
        for _, row in df.iterrows():
            day_start = row['TimeStamp']
            
            # For each hour of the day
            for hour in range(24):
                timestamp = day_start + timedelta(hours=hour)
                
                # Copy the capacity factors and generation values
                hourly_row = {'TimeStamp': timestamp}
                for plant in nuclear_plants.keys():
                    hourly_row[f'{plant}CF'] = row[f'{plant}CF']
                    hourly_row[f'{plant}Gen'] = row[f'{plant}Gen'] / 24.0  # Distribute daily generation across hours
                
                hourly_data.append(hourly_row)
        
        # Convert to dataframe and sort by timestamp
        hourly_df = pd.DataFrame(hourly_data)
        return hourly_df.sort_values('TimeStamp')
    
    # Expand daily data to hourly
    print("Expanding daily data to hourly...")
    hourly_nuclear_df = expand_to_hourly(nuclear_df)
    
    # Function to calculate weighted average CF for a zone
    def calculate_zone_cf(df, zone_plants):
        if not zone_plants:
            return pd.Series(0, index=df.index)
        
        # Calculate weighted sum of generation and total capacity
        total_gen = pd.Series(0, index=df.index)
        total_capacity = 0
        
        for plant in zone_plants:
            capacity = nuclear_plants[plant]['Capacity']
            total_gen += df[f'{plant}Gen']
            total_capacity += capacity
        
        # Calculate weighted average CF
        if total_capacity > 0:
            return total_gen / total_capacity
        else:
            return pd.Series(0, index=df.index)
    
    # Calculate zone B and C capacity factors
    print("Calculating zone capacity factors...")
    hourly_nuclear_df['Zone_B_CF'] = calculate_zone_cf(hourly_nuclear_df, zone_b_plants)
    hourly_nuclear_df['Zone_C_CF'] = calculate_zone_cf(hourly_nuclear_df, zone_c_plants)
    
    # Create output dataframe with Time_Index and capacity factors
    output_df = pd.DataFrame()
    output_df['TimeStamp'] = hourly_nuclear_df['TimeStamp']
    output_df['Time_Index'] = range(1, len(hourly_nuclear_df) + 1)
    
    # Include both CF columns
    output_df['Nuclear_B'] = hourly_nuclear_df['Zone_B_CF']
    output_df['Nuclear_C'] = hourly_nuclear_df['Zone_C_CF']
    
    # Clean up capacity factors (cap at 1.0 and handle NaN/Inf)
    for cf_col in ['Nuclear_B', 'Nuclear_C']:
        output_df[cf_col] = output_df[cf_col].fillna(0)
        output_df.loc[output_df[cf_col] > 1.0, cf_col] = 1.0
        output_df.loc[output_df[cf_col] < 0.0, cf_col] = 0.0
        output_df.loc[output_df[cf_col].isin([np.inf, -np.inf]), cf_col] = 0.0
    
    # Drop the TimeStamp column for the final output
    final_output_df = output_df.drop(columns=['TimeStamp'])
    
    # Save the hourly nuclear capacity factor data to CSV
    output_path = os.path.join(data_folder, 'nuclearHourlyCF_2019.csv')
    final_output_df.to_csv(output_path, index=False)
    print(f"Hourly nuclear capacity factor data created at {output_path}")

    return

def create_hydro_hourly_timeseries(data_folder):
    """
    Process monthly hydro generation data to create an hourly timeseries
    for hydro generators
    """
    hydro_monthly_file = os.path.join(data_folder, "hydroGenMonthly_2019.csv")
    
    print(f"Reading hydro monthly generation data from {hydro_monthly_file}...")
    hydro_df = pd.read_csv(hydro_monthly_file)
    
    # Convert timestamp to datetime
    hydro_df['TimeStamp'] = pd.to_datetime(hydro_df['TimeStamp'])
    
    # Hydro plant information - assign each plant to its zone
    hydro_plants = {
        'rmn': {'Name': 'Robert Moses Niagara', 'Zone': 'A'},  # Niagara Falls, Zone A
        'stl': {'Name': 'St. Lawrence', 'Zone': 'D'}          # St. Lawrence River, Zone D
    }
    
    # Function to expand monthly data to hourly
    def expand_to_hourly(df):
        # Create a list to hold hourly data
        hourly_data = []
        
        # For each month in the input dataframe
        for idx, row in df.iterrows():
            month_start = row['TimeStamp']
            # Get the next month
            if month_start.month == 12:
                next_month = datetime(month_start.year + 1, 1, 1)
            else:
                next_month = datetime(month_start.year, month_start.month + 1, 1)
            
            # Calculate number of hours in this month
            delta = next_month - month_start
            hours_in_month = delta.days * 24
            
            # For each hour of the month
            for hour in range(hours_in_month):
                timestamp = month_start + timedelta(hours=hour)
                
                # Copy the capacity factors
                hourly_row = {'TimeStamp': timestamp}
                for plant in hydro_plants.keys():
                    hourly_row[f'{plant}CF'] = row[f'{plant}CF']
                
                hourly_data.append(hourly_row)
        
        # Convert to dataframe and sort by timestamp
        hourly_df = pd.DataFrame(hourly_data)
        return hourly_df.sort_values('TimeStamp')
    
    # Expand monthly data to hourly
    print("Expanding monthly data to hourly...")
    hourly_hydro_df = expand_to_hourly(hydro_df)
    
    # Create output dataframe with Time_Index and capacity factors
    output_df = pd.DataFrame()
    output_df['TimeStamp'] = hourly_hydro_df['TimeStamp']
    output_df['Time_Index'] = range(1, len(hourly_hydro_df) + 1)
    
    # Include CF columns, named according to GenX convention
    for plant, info in hydro_plants.items():
        column_name = f"Hydro_{info['Zone']}"
        output_df[column_name] = hourly_hydro_df[f'{plant}CF']
    
    # Clean up capacity factors (cap at 1.0 and handle NaN/Inf)
    for zone in ['A', 'D']:
        cf_col = f'Hydro_{zone}'
        output_df[cf_col] = output_df[cf_col].fillna(0)
        output_df.loc[output_df[cf_col] > 1.0, cf_col] = 1.0
        output_df.loc[output_df[cf_col] < 0.0, cf_col] = 0.0
        output_df.loc[output_df[cf_col].isin([np.inf, -np.inf]), cf_col] = 0.0
    
    # Drop the TimeStamp column for the final output
    final_output_df = output_df.drop(columns=['TimeStamp'])
    
    # Save the hourly hydro capacity factor data to CSV
    output_path = os.path.join(data_folder, 'hydroHourlyCF_2019.csv')
    final_output_df.to_csv(output_path, index=False)
    print(f"Hourly hydro capacity factor data created at {output_path}")
    return

def process_gold_book_data(gold_book_path, output_path=None):
    """
    Process Gold Book data to generate zonal capacity information by technology and fuel type.
    
    This function reads the Gold Book data, aggregates nameplate capacities for each 
    technology type at a zonal level, with special handling for thermal generators to
    group by fuel type as well.
    
    Args:
        gold_book_path (str): Path to the Gold Book CSV file
        output_path (str, optional): Path to save the processed data
        
    Returns:
        dict: Dictionary containing aggregated capacity data by zone, technology, and fuel type
    """
    import pandas as pd
    import os
    
    print(f"Reading Gold Book data from {gold_book_path}...")
    
    # Read the Gold Book data
    gold_book_df = pd.read_csv(gold_book_path)
    
    # Check column names and clean up if needed
    print(f"Gold Book columns: {gold_book_df.columns.tolist()}")
    
    # Handle potential variations in column naming
    name_plate_col = next((col for col in gold_book_df.columns if 'Name Plate' in col), 'Name Plate Rating (MW)')
    unit_type_col = next((col for col in gold_book_df.columns if 'Unit_Type' in col), 'Unit_Type')
    fuel_1_col = next((col for col in gold_book_df.columns if 'Fuel_1' in col), 'Fuel_1')
    fuel_2_col = next((col for col in gold_book_df.columns if 'Fuel_2' in col), 'Fuel_2')
    zone_col = 'Zone'
    
    # Map columns for clarity
    column_mapping = {
        name_plate_col: 'Nameplate_MW',
        unit_type_col: 'Technology',
        fuel_1_col: 'Primary_Fuel',
        fuel_2_col: 'Secondary_Fuel',
        zone_col: 'Zone'
    }
    
    # Create a working copy with standardized column names
    df = gold_book_df.rename(columns={k: v for k, v in column_mapping.items() if k in gold_book_df.columns})
    df = df.replace(',','', regex=True)
    df.dropna(subset='Operator', inplace=True)

    # Convert nameplate rating to numeric, handling any non-numeric values
    df['Nameplate_MW'] = pd.to_numeric(df['Nameplate_MW'], errors='coerce')
    
    # Fill NaN values for technology and fuel
    df['Technology'] = df['Technology'].fillna('Unknown')
        # Gold Book codes for unit types and fuel types
    unit_type_codes = {
        'CC': 'CombinedCycle',
        'CT': 'CombustionTurbine', 
        'CG': 'Cogeneration',
        'CW': 'CombinedCycle',  # Waste Heat Only (CC)
        'ES': 'Battery',
        'FW': 'Flywheel',
        'FC': 'FuelCell',
        'GT': 'CombustionTurbine',
        'HY': 'Hydro',
#         'IC': 'InternalCombustion',
#         'JE': 'JetEngine',
        'NB': 'Nuclear',
        'NP': 'Nuclear',
        'PS': 'PumpedHydro',
        'PV': 'SolarUtility',
        'ST': 'SteamTurbine',
        'WT': 'WindLand'
    }
    
    fuel_type_codes = {
#         'BUT': 'Butane',
        'FO2': 'FuelOil2',
        'FO4': 'FuelOil4',
        'FO6': 'FuelOil6',
#         'JF': 'JetFuel',
        # 'KER': 'Kerosene',
#         'MTE': 'MethaneBio',
        'NG': 'NaturalGas',
        'OT': 'Other',
#         'REF': 'Refuse',
        'SUN': 'Solar',
        'UR': 'Uranium',
        'WAT': 'Water',
        'WD': 'Wood',
        'WND': 'Wind'
    }

    # Convert codes to uppercase for matching
    df['Technology'] = df['Technology'].str.upper()
        
    # Create mapping from codes to standard names
    unit_type_mapping = {code.upper(): value for code, value in unit_type_codes.items()}
        
    # Apply mapping, keeping original if not found
    df['Technology_Mapped'] = df['Technology'].map(unit_type_mapping)
    df = df.dropna(subset='Technology_Mapped')
        
    # Use the mapped technology
    df['Technology'] = df['Technology_Mapped']
    df = df.drop('Technology_Mapped', axis=1)

    # Map fuel types using known codes
    if 'Primary_Fuel' in df.columns:
        # First convert to uppercase for consistency
        df['Primary_Fuel'] = df['Primary_Fuel'].str.upper().fillna('UNKNOWN')
        
        # Map known fuel codes
        fuel_mapping = {code.upper(): value for code, value in fuel_type_codes.items()}
        
        # Apply mapping, keeping original value if not found
        df['Primary_Fuel_Mapped'] = df['Primary_Fuel'].map(fuel_mapping)
        # Use the mapped fuel type
        df['Primary_Fuel'] = df['Primary_Fuel_Mapped']
        df = df.drop('Primary_Fuel_Mapped', axis=1)
    else:
        df['Primary_Fuel'] = 'UNKNOWN'
    
    # Determine if a generator is thermal
    thermal_technologies = ['CombinedCycle', 'CombustionTurbine', 'SteamTurbine']
    df['Is_Thermal'] = df['Technology'].isin(thermal_technologies)
    df[df['Is_Thermal']].dropna(subset='Primary_Fuel', inplace=True)
    # Initialize result dictionary
    capacity_by_zone = {}
    
    # Step 1: Aggregate thermal generators by zone, technology, and fuel type
    if df['Is_Thermal'].any():
        thermal_df = df[df['Is_Thermal']].copy()
        
        # Create a unique fuel identifier for thermal generators
        thermal_df['Fuel_Type'] = thermal_df['Primary_Fuel']
        
        # Group by zone, technology, and fuel type
        thermal_agg = thermal_df.groupby(['Zone', 'Technology', 'Fuel_Type']).agg({
            'Nameplate_MW': 'sum'
        }).reset_index()
        
        # Add to results dictionary
        for _, row in thermal_agg.iterrows():
            zone = row['Zone']
            tech = row['Technology']
            fuel = row['Fuel_Type']
            capacity = row['Nameplate_MW']
            
            if zone not in capacity_by_zone:
                capacity_by_zone[zone] = {}
            
            # For thermal, include fuel type in the key
            resource_key = f"{tech}_{fuel}"
            capacity_by_zone[zone][resource_key] = capacity
    
    # Step 2: Aggregate non-thermal generators by zone and technology
    non_thermal_df = df[~df['Is_Thermal']].copy()
    non_thermal_agg = non_thermal_df.groupby(['Zone', 'Technology']).agg({
        'Nameplate_MW': 'sum'
    }).reset_index()
    
    # Add to results dictionary
    for _, row in non_thermal_agg.iterrows():
        zone = row['Zone']
        tech = row['Technology']
        capacity = row['Nameplate_MW']
        
        if zone not in capacity_by_zone:
            capacity_by_zone[zone] = {}
        
        # For non-thermal, use just the technology as the key
        capacity_by_zone[zone][tech] = capacity
    
    # Create a summary DataFrame for easy viewing and export
    summary_rows = []
    for zone, techs in capacity_by_zone.items():
        for tech, capacity in techs.items():
            # Split technology and fuel for thermal resources
            if '_' in tech and any(thermal_tech in tech for thermal_tech in thermal_technologies):
                tech_name, fuel = tech.split('_', 1)
                summary_rows.append({
                    'Zone': zone,
                    'Technology': tech_name,
                    'Fuel': fuel,
                    'Capacity_MW': capacity
                })
            else:
                summary_rows.append({
                    'Zone': zone,
                    'Technology': tech,
                    'Fuel': 'None',
                    'Capacity_MW': capacity
                })
    summary_df = pd.DataFrame(summary_rows)
    
    # Print zone totals
    zone_totals = summary_df.groupby('Zone')['Capacity_MW'].sum()
    print("\nTotal capacity by zone (MW):")
    for zone, total in zone_totals.items():
        print(f"Zone {zone}: {total:.2f} MW")
    
    # Print technology totals
    tech_totals = summary_df.groupby('Technology')['Capacity_MW'].sum()
    print("\nTotal capacity by technology (MW):")
    for tech, total in tech_totals.items():
        print(f"{tech}: {total:.2f} MW")
    
    # Save to file if output path is provided
    if output_path:
        summary_df.to_csv(output_path, index=False)
        print(f"\nSaved processed data to {output_path}")
    
    return {
        'capacity_by_zone': capacity_by_zone,
        'summary_df': summary_df
    }

def process_atb_data(atb_path, output_path=None):
    """
    Process NREL ATB data to extract cost parameters for GenX.
    
    This function reads the ATB data and extracts key cost parameters:
    - Annualized CAPEX ($/MW-YR)
    - Annualized fixed O&M ($/MW-YR)
    - Variable O&M ($/MWh)
    - For existing units, yearly going forward costs
    
    Args:
        atb_path (str): Path to the ATB CSV file
        output_path (str, optional): Path to save the processed data
        
    Returns:
        dict: Dictionary containing processed cost data by technology
    """
    
    print(f"Reading ATB data from {atb_path}...")
    
    # Read the ATB data
    atb_df = pd.read_csv(atb_path)
    
    # Check column names
    print(f"ATB columns: {atb_df.columns.tolist()}")
    
    # Filter data for the moderate cost case
        # Filter data for the moderate cost case
    # Filter data for the moderate cost case
    moderate_case = atb_df[
        (atb_df['core_metric_case'] == 'Market') &
        (atb_df['scenario'] == 'Moderate') & 
        ((atb_df['core_metric_variable'] == 2024) | ((atb_df['technology'] == 'Nuclear') & 
                                                     (atb_df['core_metric_variable'] == 2030))) & 
        (atb_df['crpyears'] == 20)
    ]
    moderate_case['combined'] = moderate_case['technology'].astype(str) +' '+moderate_case['techdetail'].astype(str)
    # Map technologies to GenX technology types
    # For Natural Gas, we need to differentiate between different subtypes
    technology_mapping = {
        '4Hr Battery Storage': 'Battery',
        'LandbasedWind': 'WindLand',
        'OffshoreWind': 'WindOffshore',
        'UtilityPV': 'SolarUtility',
        'CommPV': 'SolarBTM',
#         'ResPV': 'SolarBTM',
        'Hydropower': 'Hydro',
        'Nuclear': 'Nuclear',
        'Pumped Storage': 'PumpedHydro',
#         'Biopower': 'InternalCombustion',
        'Combined Cycle': 'CombinedCycle',
        'Combustion Turbine': 'CombustionTurbine',
        'Coal': 'SteamTurbine'
    }
    
    # Initialize dictionary to store processed cost data
    tech_costs = {}
    
    # First, process explicitly mapped technologies
    for atb_tech, genx_tech in technology_mapping.items():
        tech_data = moderate_case[moderate_case['combined'].str.contains(atb_tech, case=False, na=False)]
        
        # Initialize cost parameters for this technology
        tech_costs[genx_tech] = {
            'Ann_Inv_MWYr': 0,    # Annualized CAPEX ($/MW-yr)
            'Fix_OM_MW': 0,        # Fixed O&M ($/MW-yr)
            'Var_OM': 0            # Variable O&M ($/MWh)
        }
        
        # Extract CAPEX
        capex_data = tech_data[tech_data['core_metric_parameter'].str.contains('CAPEX', case=False, na=False)]
        if not capex_data.empty:
            total_capex = capex_data['value'].mean()
            crf = 0.094  # Capital recovery factor for 20 years @ 7%
            tech_costs[genx_tech]['Ann_Inv_MWYr'] = total_capex * crf * 1000  # Convert to $/MW
        
        # Extract Fixed O&M
        fom_data = tech_data[tech_data['core_metric_parameter'].str.contains('Fixed O&M', case=False, na=False)]
        if not fom_data.empty:
            tech_costs[genx_tech]['Fix_OM_MW'] = fom_data['value'].mean() * 1000  # Convert to $/MW
        
        # Extract Variable O&M
        vom_data = tech_data[tech_data['core_metric_parameter'].str.contains('Variable O&M', case=False, na=False)]
        if not vom_data.empty:
            tech_costs[genx_tech]['Var_OM'] = vom_data['value'].mean()
    
    summary_rows = []
    for tech, costs in tech_costs.items():
        row = {'Technology': tech}
        row.update(costs)
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save to file if output path is provided
    if output_path:
        summary_df.to_csv(output_path, index=False)
        print(f"\nSaved processed ATB data to {output_path}")
    
    # Print summary
    print("\nProcessed ATB cost data summary:")
    print(summary_df.to_string())
    
    return {
        'tech_costs': tech_costs,
        'summary_df': summary_df
    }
def create_thermal_capacity_factors(gold_book_path, output_path=None, year=2019):
    """
    Create a thermal capacity factor CSV file using seasonal capacity/nameplate ratios
    from the Gold Book data. This generates a synthetic hourly capacity factor profile
    for each thermal generator type by zone.
    
    Args:
        gold_book_path (str): Path to the Gold Book CSV file
        output_path (str, optional): Path to save the thermal capacity factor CSV file
                                     If None, defaults to "thermalHourlyCF_{year}.csv"
        year (int, optional): Year for the capacity factors, used for file naming
        
    Returns:
        pandas.DataFrame: DataFrame containing the hourly capacity factors
    """
    import pandas as pd
    import numpy as np
    import os
    from datetime import datetime, timedelta
    
    print(f"Creating thermal capacity factors from Gold Book data: {gold_book_path}")
    
    # Set default output path if not provided
    output_path = os.path.join(output_path,f"thermalHourlyCF_{year}.csv")
    
    # Read the Gold Book data
    gold_book_df = pd.read_csv(gold_book_path)
    
    # Handle potential variations in column naming
    name_plate_col = next((col for col in gold_book_df.columns if 'Name Plate' in col), 'Name Plate Rating (MW)')
    unit_type_col = next((col for col in gold_book_df.columns if 'Unit_Type' in col), 'Unit_Type')
    fuel_1_col = next((col for col in gold_book_df.columns if 'Fuel_1' in col), 'Fuel_1')
    capability_summer_col = next((col for col in gold_book_df.columns if 'Capability_Summer' in col), 'Capability_Summer')
    capability_winter_col = next((col for col in gold_book_df.columns if 'Capability_Winter' in col), 'Capability_Winter')
    zone_col = 'Zone'
    operator_col = 'Operator'
    
    # Map columns for clarity
    column_mapping = {
        name_plate_col: 'Nameplate_MW',
        unit_type_col: 'Technology',
        fuel_1_col: 'Primary_Fuel',
        capability_summer_col: 'Summer_MW',
        capability_winter_col: 'Winter_MW',
        zone_col: 'Zone',
        operator_col: 'Operator'
    }
    
    # Create a working copy with standardized column names
    df = gold_book_df.rename(columns={k: v for k, v in column_mapping.items() if k in gold_book_df.columns})
    
    # Drop rows with missing operator (these are usually non-operational units)
    df.dropna(subset=['Operator'], inplace=True)
    df = df.replace(',','', regex=True)
    # Convert capacity values to numeric, handling any non-numeric values
    for col in ['Nameplate_MW', 'Summer_MW', 'Winter_MW']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            print(f"Warning: {col} not found in Gold Book data")
            df[col] = np.nan
    
    # Fill missing values with reasonable defaults
    df['Winter_MW'] = df['Winter_MW'].fillna(df['Nameplate_MW'])
    df['Summer_MW'] = df['Summer_MW'].fillna(df['Nameplate_MW'])
    
    # Map unit types to standardized technology names
    unit_type_codes = {
        'CC': 'CombinedCycle',
        'CT': 'CombustionTurbine',
        'CG': 'Cogeneration',
        'CW': 'CombinedCycle',  # Waste Heat Only (CC)
        'GT': 'CombustionTurbine',
        'IC': 'InternalCombustion',
        'JE': 'JetEngine',
        'ST': 'SteamTurbine'
    }
    
    # Convert codes to uppercase for matching
    if 'Technology' in df.columns:
        df['Technology'] = df['Technology'].str.upper()
        
        # Create mapping from codes to standard names
        unit_type_mapping = {code.upper(): value for code, value in unit_type_codes.items()}
        
        # Apply mapping, keeping original if not found
        df['Technology_Mapped'] = df['Technology'].map(unit_type_mapping)
        df['Technology_Mapped'] = df['Technology_Mapped'].fillna(df['Technology'])
        
        # Use the mapped technology
        df['Technology'] = df['Technology_Mapped']
        df = df.drop('Technology_Mapped', axis=1)
    else:
        print(f"Warning: Technology column not found in Gold Book data")
        df['Technology'] = 'Unknown'
    fuel_type_codes = {
        'FO2': 'FuelOil2',
        'FO4': 'FuelOil4',
        'FO6': 'FuelOil6',
#         'JF': 'JetFuel',
        # 'KER': 'Kerosene',
#         'MTE': 'MethaneBio',
        'NG': 'NaturalGas',
#         'OT': 'Other',
#         'REF': 'Refuse',
        'SUN': 'Solar',
        'UR': 'Uranium',
        'WAT': 'Water',
        'WD': 'Wood',
        'WND': 'Wind'
    }

    # Map fuel types using known codes
    if 'Primary_Fuel' in df.columns:
        # First convert to uppercase for consistency
        df['Primary_Fuel'] = df['Primary_Fuel'].str.upper().fillna('UNKNOWN')
        
        # Map known fuel codes
        fuel_mapping = {code.upper(): value for code, value in fuel_type_codes.items()}
        
        # Apply mapping, keeping original value if not found
        df['Primary_Fuel_Mapped'] = df['Primary_Fuel'].map(fuel_mapping)
        df = df.dropna(subset='Primary_Fuel_Mapped')
        
        # Use the mapped fuel type
        df['Primary_Fuel'] = df['Primary_Fuel_Mapped']
        df = df.drop('Primary_Fuel_Mapped', axis=1)
    else:
        df['Primary_Fuel'] = 'UNKNOWN'
    

    # Filter for thermal technologies only
    thermal_technologies = ['CombinedCycle', 'CombustionTurbine', 'SteamTurbine']
    thermal_df = df[df['Technology'].isin(thermal_technologies)].copy()
    
    print(f"Found {len(thermal_df)} thermal generators")
    
    # Calculate capacity factors
    thermal_df['Summer_CF'] = thermal_df['Summer_MW'] / thermal_df['Nameplate_MW']
    thermal_df['Winter_CF'] = thermal_df['Winter_MW'] / thermal_df['Nameplate_MW']
    
    # Clean up unreasonable values (cap between 0.1 and 1.0)
    thermal_df['Summer_CF'] = thermal_df['Summer_CF'].clip(0.1, 1.0)
    thermal_df['Winter_CF'] = thermal_df['Winter_CF'].clip(0.1, 1.0)
    
    # Calculate average capacity factor for each zone and technology combination
    zonal_tech_cf = thermal_df.groupby(['Zone', 'Technology', 'Primary_Fuel']).agg({
        'Summer_CF': 'mean',
        'Winter_CF': 'mean',
        'Nameplate_MW': 'sum'  # For weighting
    }).reset_index()
    
    print(f"Aggregated to {len(zonal_tech_cf)} zone-technology combinations")
    
    # Define seasons by month
    # Winter: Dec, Jan, Feb, Mar
    # Spring: Apr, May
    # Summer: Jun, Jul, Aug, Sep
    # Fall: Oct, Nov
    
    # Create a date range for the entire year
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31, 23)  # Last hour of the year
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # Create a dataframe with Time_Index and all hours
    hourly_df = pd.DataFrame({'Time_Index': range(1, len(date_range) + 1)})
    hourly_df['datetime'] = date_range
    hourly_df['month'] = hourly_df['datetime'].dt.month
    
    # Define season for each month
    season_map = {
        1: 'Winter',  # Jan
        2: 'Winter',  # Feb
        3: 'Winter',  # Mar
        4: 'Spring',  # Apr
        5: 'Spring',  # May
        6: 'Summer',  # Jun
        7: 'Summer',  # Jul
        8: 'Summer',  # Aug
        9: 'Summer',  # Sep
        10: 'Fall',   # Oct
        11: 'Fall',   # Nov
        12: 'Winter'  # Dec
    }
    hourly_df['season'] = hourly_df['month'].map(season_map)
    
    # For each zone-technology combination, create hourly capacity factors
    for _, row in zonal_tech_cf.iterrows():
        zone = row['Zone']
        tech = row['Technology']
        fuel = row['Primary_Fuel']
        summer_cf = row['Summer_CF']
        winter_cf = row['Winter_CF']
        
        # Calculate spring and fall as average of summer and winter
        spring_cf = (summer_cf + winter_cf) / 2
        fall_cf = (summer_cf + winter_cf) / 2
        
        # Create resource identifier
        resource_name = f"{tech}_{fuel}_{zone}"
        
        # Add small random variations to make it more realistic
        np.random.seed(hash(resource_name) % 10000)  # Use hash for deterministic randomness
        
        # Assign capacity factors by season with small random variations (±5%)
        hourly_df[resource_name] = np.where(
            hourly_df['season'] == 'Summer', 
            summer_cf, 
            np.where(
                hourly_df['season'] == 'Winter',
                winter_cf,
                np.where(
                    hourly_df['season'] == 'Spring',
                    spring_cf,
                    fall_cf
                )
            )
        )
        
        # Clip to ensure values stay between 0 and 1
        hourly_df[resource_name] = hourly_df[resource_name].clip(0.1, 1.0)
    
    # Add planned outages (simulate for ~5% of hours)
    for resource_name in [col for col in hourly_df.columns if col not in ['Time_Index', 'datetime', 'month', 'season']]:
        # Number of outage periods (2-4 per year)
        num_outages = np.random.randint(2, 5)
        
        for _ in range(num_outages):
            # Select random outage start day (avoiding consecutive outages)
            outage_start_idx = np.random.randint(0, len(hourly_df) - 336)  # Avoid last 2 weeks
            
            # Outage duration (24-168 hours, i.e., 1-7 days)
            outage_duration = np.random.randint(24, 169)
            
            # Apply outage (reduced capacity factor)
            outage_cf = np.random.uniform(0.1, 0.4)  # Significant reduction but not zero
            hourly_df.loc[outage_start_idx:outage_start_idx + outage_duration, resource_name] = outage_cf
    
    # Remove temporary columns
    result_df = hourly_df.drop(['datetime', 'month', 'season'], axis=1)
    
    # Save to CSV
    result_df.to_csv(output_path, index=False)
    print(f"Created thermal capacity factor file with {len(result_df.columns) - 1} resources at {output_path}")
    
    return result_df

if __name__ == "__main__":
    data_folder = '../NYISO_Data/'
    # Convert to hourly resolution
    create_hydro_hourly_timeseries(data_folder)
    create_nuclear_hourly_timeseries(data_folder)

    # Process starting Capacities
    gb_in = os.path.join(data_folder, 'Gold Book Matching.csv')
    gb_out = os.path.join(data_folder, 'NYCA2024_Summary.csv')
    process_gold_book_data(gb_in,gb_out)

    #Process Thermal CFs
    create_thermal_capacity_factors(gb_in, output_path=data_folder, year=2019)

    #Process ATB Data
    atb_in = os.path.join(data_folder, 'ATB.csv')
    atb_out = os.path.join(data_folder, 'ATB_Summary.csv')
    process_atb_data(atb_in,atb_out)
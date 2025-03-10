#!/usr/bin/env python3
"""
NYISO GenX Zonal Case Generator

This script generates a GenX case for the NYISO system with updated data.
It creates necessary directories and files for modeling the New
 York power system at the zonal level.
"""

import os
import csv
import shutil
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

class NYISOGenXGenerator_Zonal:
    """Class for generating a GenX case for the NYISO system at the zonal level"""
    
    def __init__(self, case_name="NYISO_Case", base_dir=None):
        """
        Initialize the NYISO GenX case generator with empty technology configurations
        
        Args:
            case_name (str): Name of the case folder
            base_dir (str): Base directory for the case (default: current directory)
        """
        self.case_name = case_name
        
        # Set paths
        if base_dir:
            self.base_dir = base_dir
        else:
            self.base_dir = os.getcwd()
            
        self.case_path = os.path.join(self.base_dir, self.case_name)
        self.data_path = '../NYISO_Data'
        
        # Define the NYISO zones
        self.zones = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
        self.zone_to_number = {zone: i for i, zone in enumerate(self.zones, 1)}
        
        # Define zonal scale factors using NYCA as base 1.0
        # Original values: NYCA: 132.98, G-J: 174.72, NYC: 229.11, Long Island: 186.37
        nyca_base = 132.98
        self.zonal_scale_factors = {
            'A': 1.0,                 # WEST (NYCA base)
            'B': 1.0,                 # GENESE (NYCA base)
            'C': 1.0,                 # CENTRL (NYCA base)
            'D': 1.0,                 # NORTH (NYCA base)
            'E': 1.0,                 # MHK VL (NYCA base)
            'F': 1.0,                 # CAPITL (NYCA base)
            'G': 174.72 / nyca_base,  # HUD VL (G-J)
            'H': 174.72 / nyca_base,  # MILLWD (G-J)
            'I': 174.72 / nyca_base,  # DUNWOD (G-J)
            'J': 229.11 / nyca_base,  # N.Y.C.
            'K': 186.37 / nyca_base   # LONGIL
        }
        
        # Initialize empty capacity dictionaries for each technology type
        # These will be populated from data rather than hard-coded
        self.technology_capacities = {
            # Dictionary structure will be:
            # 'TechnologyName': {zone: capacity}
            # e.g., 'WindLand': {'A': 2692, 'B': 390, ...}
        }
        
        # Keep track of all technologies and resource types for reference
        self.all_technologies = set()
        self.thermal_technologies = {
            'CombinedCycle', 
            'CombustionTurbine', 
            'SteamTurbine'
#             'InternalCombustion',
#             'JetEngine'
        }
        # Generic technology parameters and defaults
        self.technology_params = {
            # Default parameters for each technology
            # Will be used if specific parameters aren't provided
        }
        
        # Keep track of all thermal resources with their fuel types
        self.thermal_resources = {}  # {resource_name: {'zone': zone, 'tech': tech, 'fuel': fuel, 'capacity': cap}}
    
    def load_technology_capacities(self, data):
        """
        Load technology capacities from provided data
        
        Args:
            data (DataFrame or dict): Data containing technology capacities by zone
                Expected format for DataFrame: columns ['Zone', 'Technology', 'Fuel', 'Capacity_MW']
                Expected format for dict: {'capacity_by_zone': {zone: {tech: capacity}}}
        """
        if isinstance(data, dict) and 'capacity_by_zone' in data:
            capacity_by_zone = data['capacity_by_zone']
            
            # Extract and organize technology capacities
            for zone, techs in capacity_by_zone.items():
                for tech_key, capacity in techs.items():
                    # Determine if this is a thermal resource with fuel type
                    if '_' in tech_key and any(thermal_tech in tech_key for thermal_tech in self.thermal_technologies):
                        tech, fuel = tech_key.split('_', 1)
                        resource_name = f"{tech}_{fuel}_{zone}"
                        
                        # Add to thermal resources
                        if tech not in self.thermal_resources:
                            self.thermal_resources[resource_name] = {
                                'zone': zone, 
                                'tech': tech, 
                                'fuel': fuel, 
                                'capacity': capacity
                            }
                        
                        # Also add to general technology capacities
                        if tech not in self.technology_capacities:
                            self.technology_capacities[tech] = {zone: 0 for zone in self.zones}
                        
                        # Add capacity
                        self.technology_capacities[tech][zone] = capacity + self.technology_capacities[tech].get(zone, 0)
                        
                        # Add to all technologies
                        self.all_technologies.add(tech)
                    else:
                        # Non-thermal resource
                        tech = tech_key
                        
                        # Add to technology capacities
                        if tech not in self.technology_capacities:
                            self.technology_capacities[tech] = {zone: 0 for zone in self.zones}
                        
                        # Add capacity
                        self.technology_capacities[tech][zone] = capacity
                        
                        # Add to all technologies
                        self.all_technologies.add(tech)
            
            print(f"Loaded technology capacities for {len(self.technology_capacities)} technologies")
            
        elif isinstance(data, pd.DataFrame):
            # Expected columns: Zone, Technology, Fuel, Capacity_MW
            required_cols = ['Zone', 'Technology', 'Capacity_MW']
            if not all(col in data.columns for col in required_cols):
                print(f"Error: DataFrame must contain columns {required_cols}")
                return
            
            # Handle Fuel column if present
            has_fuel = 'Fuel' in data.columns
            
            # Process each row
            for _, row in data.iterrows():
                zone = row['Zone']
                tech = row['Technology']
                capacity = row['Capacity_MW']
                
                # Add to all technologies
                self.all_technologies.add(tech)
                
                # Check if this is a thermal technology
                is_thermal = tech in self.thermal_technologies
                
                if is_thermal and has_fuel and row['Fuel'] != 'None':
                    # This is a thermal resource with fuel
                    fuel = row['Fuel']
                    resource_name = f"{tech}_{fuel}_{zone}"
                    
                    # Add to thermal resources
                    self.thermal_resources[resource_name] = {
                        'zone': zone, 
                        'tech': tech, 
                        'fuel': fuel, 
                        'capacity': capacity
                    }
                
                # Add to general technology capacities
                if tech not in self.technology_capacities:
                    self.technology_capacities[tech] = {zone: 0 for zone in self.zones}
                
                # Add capacity
                self.technology_capacities[tech][zone] = capacity + self.technology_capacities[tech].get(zone, 0)
            
            print(f"Loaded technology capacities for {len(self.technology_capacities)} technologies")
        
        else:
            print("Error: Unsupported data format. Provide either a dictionary or DataFrame.")
    
    def create_case_structure(self):
        """Create the folder structure for a GenX case and populate it with necessary files"""
        
        # Create the main directory structure
        directories = [
            f"{self.case_path}/settings",
            f"{self.case_path}/system",
#             f"{self.case_path}/policies",
            f"{self.case_path}/resources",
#             f"{self.case_path}/resources/policy_assignments"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        
        # Create a simple README.md file
        with open(f"{self.case_path}/README.md", "w") as readme:
            readme.write(f"# {self.case_name}\n\nThis is a GenX case for NYISO with 11 zones (A, B, C, D, E, F, G, H, I, J, K).")
        
        # Create empty Run.jl file
        with open(f"{self.case_path}/Run.jl", "w") as run_file:
            run_file.write("# GenX Run file for NYISO case\n")
        
        print(f"GenX case structure for {self.case_name} created successfully")
    
    def create_nyiso_network_csv(self):
        """Create a Network.csv file for NYISO with 11 zones (H and I separated)"""
        output_file = f"{self.case_path}/system/Network.csv"
        
        # Define connections and transfer limits using zone letters
        # (we'll convert to numbers when writing the file)
        connections = [
            ('A', 'B', 5133),
            ('B', 'C', 1600),
            ('C', 'E', 8432),
            ('D', 'E', 4161),
            ('E', 'F', 3600),
            ('E', 'G', 9279),
            ('F', 'G', 4600),
            ('G', 'H', 7356),  # Half of original G↔HI capacity
            ('H', 'I', 5000),  # New connection between H and I (estimated)
            ('I', 'J', 8675),  # Original HI↔J connection now from I
            ('I', 'K', 4520),  # Original HI↔K connection now from I
            ('J', 'K', 300)
        ]
        
        # Create the headers and first rows with zone definitions
        headers = ["Network_zones"]
        zone_numbers = []
        for i, zone in enumerate(self.zones, 1):
            zone_numbers.append(f"z{i}")
        
        # Add Network_Lines, Start_Zone, End_Zone, Line_Max_Flow_MW
        headers.extend(["Network_Lines", "Start_Zone", "End_Zone", "Line_Max_Flow_MW"])
        
        # Ensure system directory exists
        system_dir = os.path.dirname(output_file)
        os.makedirs(system_dir, exist_ok=True)
        
        # Open the file for writing
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write the header row
            writer.writerow(headers)
            # Create rows for each connection, converting zone letters to numbers
            for i, (start_zone_letter, end_zone_letter, limit) in enumerate(connections, 1):
                # Convert zone letters to numbers
                start_zone_number = self.zone_to_number[start_zone_letter]
                end_zone_number = self.zone_to_number[end_zone_letter]
                
                # Create a row with empty values for zone columns, then add connection data
                try: 
                    connection_row = [zone_numbers[i-1]] + [i, start_zone_number, end_zone_number, limit]
                except:
                    connection_row = [''] + [i, start_zone_number, end_zone_number, limit]
                
                writer.writerow(connection_row)
        
        print(f"Successfully created {output_file}")
    
    def create_demand_data_csv(self, year="2019"):
        """
        Create Demand_data.csv file for GenX from hourly load data
        """
        input_file = os.path.join(self.data_path, f"loadHourly_{year}.csv")
        output_file = f"{self.case_path}/system/Demand_data.csv"
        
        # Read the hourly load data
        print(f"Reading load data from {input_file}...")
        df = pd.read_csv(input_file)
        
        # Convert timestamp to datetime
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        
        # Create a pivot table with timestamp as index and zones as columns
        load_pivot = df.pivot_table(
            index='TimeStamp', 
            columns='ZoneID', 
            values='Load', 
            aggfunc='sum'
        )
        
        # Check for missing zones and add them with default values
        for zone in self.zones:
            if zone not in load_pivot.columns:
                print(f"Warning: Zone {zone} not found in input data. Adding with zero load.")
                load_pivot[zone] = 0
        
        # Sort columns to ensure zones are in order
        load_pivot = load_pivot[sorted(load_pivot.columns)]
        
        # Number of hours in the dataset
        n_hours = len(load_pivot)
        print(f"Dataset contains {n_hours} hours of data")
        
        # Create the output dataframe with the correct structure
        # First, create rows for the metadata that appears only once
        metadata_rows = pd.DataFrame({
            'Voll': [10000],
            'Demand_Segment': [1],
            'Cost_of_Demand_Curtailment_per_MW': [1],
            'Max_Demand_Curtailment': [1],
            'Rep_Periods': [1],
            'Timesteps_per_Rep_Period': [n_hours],
            'Sub_Weights': [n_hours]
        })
        
        # Now create a dataframe for the time series data
        timeseries_rows = pd.DataFrame()
        timeseries_rows['Time_Index'] = list(range(1, n_hours + 1))
        
        # Add demand columns for each zone
        for zone in self.zones:
            if zone in load_pivot.columns:
                timeseries_rows[f'Demand_MW_z{self.zone_to_number[zone]}'] = load_pivot[zone].values
            else:
                timeseries_rows[f'Demand_MW_z{self.zone_to_number[zone]}'] = [0] * n_hours
        
        # Now concatenate the two dataframes
        demand_data = pd.concat([metadata_rows, timeseries_rows], axis=1)
        demand_data.to_csv(output_file, index=False)
        print(f"Demand_data.csv created at {output_file}")
    
    def create_fuels_data_csv(self, year="2019"):
        """
        Create Fuels_data.csv file for GenX from weekly fuel price data,
        ensuring exactly 8760 hours for a full year
        """
        input_file = os.path.join(self.data_path, f"fuelPriceWeekly_{year}.csv")
        output_file = f"{self.case_path}/system/Fuels_data.csv"

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Read the weekly fuel price data
        print(f"Reading fuel price data from {input_file}...")
        df = pd.read_csv(input_file)

        # Convert timestamp to datetime
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])

        # Get the list of fuels (all columns except TimeStamp)
        fuel_columns = [col for col in df.columns if col != 'TimeStamp']

        # Add 'None' for renewable resources
        all_fuels = fuel_columns + ['None']

        # Define CO2 emissions intensity for each fuel (tons/MMBtu)
        co2_intensity = {
            'NG_A2E': 0.05306,
            'NG_F2I': 0.05306,
            'NG_J': 0.05306,
            'NG_K': 0.05306,
            'FO2_UPNY': 0.07315,
            'FO2_DSNY': 0.07315,
            'FO6_UPNY': 0.07415,
            'FO6_DSNY': 0.07415,
            'coal_NY': 0.10316,
#             'Kerosene_NY': 0.07315,  
            'None': 0
        }

        # Create a fixed date range for exactly 8760 hours (full year)
        # Start at January 1, 2019 00:00
        start_date = datetime(int(year), 1, 1)
        # End at December 31, 2019 23:00 (last hour of the year)
        end_date = datetime(int(year), 12, 31, 23)

        # Create a range of hourly timestamps - exactly 8760 hours
        hourly_dates = pd.date_range(start=start_date, end=end_date, freq='H')

        # Verify we have exactly 8760 hours
        assert len(hourly_dates) == 8760, f"Expected 8760 hours, got {len(hourly_dates)}"

        # Create a new dataframe with hourly timestamps
        hourly_df = pd.DataFrame({'TimeStamp': hourly_dates})

        # For each weekly entry in the input data, find matching hours in our fixed range
        for idx, row in df.iterrows():
            week_start = row['TimeStamp']
            if idx < len(df) - 1:
                week_end = df.iloc[idx + 1]['TimeStamp'] - timedelta(hours=1)
            else:
                # For the last week, only go up to the end of year
                week_end = min(week_start + timedelta(days=6, hours=23), end_date)

            # Get mask for this week's hours
            mask = (hourly_df['TimeStamp'] >= week_start) & (hourly_df['TimeStamp'] <= week_end)

            # Assign fuel prices to these hours
            for fuel in fuel_columns:
                hourly_df.loc[mask, fuel] = row[fuel]

        # Fill any NaN values with the last valid value (in case of missing data)
        for fuel in fuel_columns:
            hourly_df[fuel] = hourly_df[fuel].ffill()
            # If there are still NaN values at the beginning, fill with first valid value
            hourly_df[fuel] = hourly_df[fuel].bfill()

        # Add 'None' fuel with zero price for all hours
        hourly_df['None'] = 0

        # Create the final Fuels_data.csv structure
        # First row: fuel names
        fuel_names = ['Time_Index'] + all_fuels

        # Second row: CO2 intensity
        co2_row = [''] + [co2_intensity.get(fuel, 0) for fuel in all_fuels]

        # Create output dataframe
        fuels_data = pd.DataFrame(columns=fuel_names)

        # Add the CO2 intensity row as the first row
        fuels_data.loc[0] = co2_row

        # Add hourly fuel prices with proper Time_Index
        hourly_df['Time_Index'] = range(1, len(hourly_df) + 1)

        # Reorder columns to match the required format
        cols_to_use = ['Time_Index'] + all_fuels
        hourly_df = hourly_df[cols_to_use]

        # Append the hourly data to the output dataframe
        fuels_data = pd.concat([fuels_data, hourly_df], ignore_index=True)

        # Double-check that we have exactly 8760 rows of data + 1 row for CO2 intensity
        assert len(fuels_data) == 8761, f"Expected 8761 rows (8760 hours + 1 header), got {len(fuels_data)}"

        # Save without index to match the required format
        fuels_data.to_csv(output_file, index=False)
        print(f"Fuel_data.csv created at {output_file}")
        
    def map_fuel_to_genx(self, fuel, zone):
        """Map a fuel type to the appropriate GenX fuel name based on zone"""
        # Map natural gas by zone
        if isinstance(fuel,float):
            pass
        else:
            if fuel.lower() in ['natural_gas', 'naturalgas', 'natural gas', 'ng']:
                if zone in ['A', 'B', 'C', 'D', 'E']:
                    return 'NG_A2E'
                elif zone in ['F', 'G', 'H', 'I']:
                    return 'NG_F2I'
                elif zone == 'J':
                    return 'NG_J'
                elif zone == 'K':
                    return 'NG_K'

            # Map fuel oil by zone
            if fuel.lower() in ['fuel_oil_2', 'fueloil2', 'fuel oil 2', 'fo2']:
                if zone in ['J', 'K']:
                    return 'FO2_DSNY'
                else:
                    return 'FO2_UPNY'

            if fuel.lower() in ['fuel_oil_6', 'fueloil6', 'fuel oil 6', 'fo6']:
                if zone in ['J', 'K']:
                    return 'FO6_DSNY'
                else:
                    return 'FO6_UPNY'

            # Map coal
            if fuel.lower() in ['coal']:
                return 'coal_NY'
#             if fuel.lower() in ['Kerosene']:
#                 return 'Kerosene_NY'
        # For all other fuels, use None
        return 'None'
    def extract_resource_names(self):
        """
        Extract all resource names from resource files in the case folder
        
        Returns:
            dict: Dictionary mapping resource types to lists of resource names
        """
        resources_folder = os.path.join(self.case_path, "resources")
        resource_files = {
            "Thermal": "Thermal.csv",
            "Hydro": "Hydro.csv",
            "Storage": "Storage.csv",
            "Vre": "Vre.csv"
        }
        
        resource_names = {}
        
        for resource_type, filename in resource_files.items():
            file_path = os.path.join(resources_folder, filename)
            if os.path.exists(file_path):
                print(f"Reading resources from {file_path}...")
                try:
                    df = pd.read_csv(file_path)
                    if 'Resource' in df.columns:
                        resource_names[resource_type] = df['Resource'].tolist()
                    else:
                        print(f"Warning: No 'Resource' column found in {filename}")
                        resource_names[resource_type] = []
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    resource_names[resource_type] = []
            else:
                print(f"Warning: {filename} not found in resources folder")
                resource_names[resource_type] = []
        
        return resource_names
    

    def create_generators_variability(self):
        import re
        """
        Create Generators_variability.csv by reading resource names and matching with capacity factor data
        based on resource names and zones.
        """
        # Step 1: Extract all resource names from resource files
        resource_names = self.extract_resource_names()
        # Flatten all resource names into a single list
        all_resources = []
        for resource_list in resource_names.values():
            all_resources.extend(resource_list)

        print(f"Found {len(all_resources)} resources across all resource files")

        # Step 2: Read data sources
        data_files = {
            "merged": os.path.join(self.data_path, "merged_2019.csv"),
            "thermal": os.path.join(self.data_path, "thermalHourlyCF_2019.csv"),
            "nuclear": os.path.join(self.data_path, "nuclearHourlyCF_2019.csv"),
            "hydro": os.path.join(self.data_path, "hydroHourlyCF_2019.csv")
        }

        # Step 3: Create an empty dataframe with just Time_Index
        num_hours = 8760
        variability_df = pd.DataFrame({'Time_Index': range(1, num_hours + 1)})

        # Step 4: Process thermal resources
        if os.path.exists(data_files['thermal']):
            print(f"Reading thermal data from {data_files['thermal']}...")
            thermal_df = pd.read_csv(data_files['thermal'])

            # Get thermal resources from our resource names
            thermal_resources = resource_names.get('Thermal', [])

            # Match thermal resource names with columns in thermal_df
            thermal_columns_processed = 0
            for resource in thermal_resources:
                # Extract the zone and technology info from the resource name
                # Typical format is "TechType_Fuel_Zone_Existing" or "TechType_Fuel_Zone_New"
                parts = resource.split('_')

                # Check if we have enough parts to extract technology and zone
                if len(parts) >= 3:
                    tech_type = parts[0]
                    zone = None

                    # Try to find the zone in the resource name
                    for part in parts:
                        if part in self.zones:
                            zone = part
                            break

                    if zone:
                        # Look for matching columns in thermal_df
                        # Different patterns might exist in the data file
                        # Try different patterns for matching
                        possible_patterns = [
                            f"{tech_type}_{zone}",  # e.g., "CombinedCycle_A"
                            f"{zone}_{tech_type}",  # e.g., "A_CombinedCycle"
                            f"{tech_type}.*{zone}",  # Any pattern with tech and zone
                            f"{zone}.*{tech_type}"   # Any pattern with zone and tech
                        ]

                        matched = False
                        for col in thermal_df.columns:
                            if col == 'Time_Index':
                                continue

                            # Check if column matches any of our patterns
                            for pattern in possible_patterns:
                                if re.search(pattern, col, re.IGNORECASE):
                                    print(f"  Matched thermal resource {resource} with data column {col}")
                                    variability_df[resource] = thermal_df[col].values
                                    thermal_columns_processed += 1
                                    matched = True
                                    break

                            if matched:
                                break

                        if not matched:
                            print(f"  No matching thermal data found for {resource}")

            print(f"Processed {thermal_columns_processed} thermal resources")

        # Step 5: Process nuclear resources
        if os.path.exists(data_files['nuclear']):
            print(f"Reading nuclear data from {data_files['nuclear']}...")
            nuclear_df = pd.read_csv(data_files['nuclear'])

            # Identify which zones have nuclear data
            zones_with_data = []
            for col in nuclear_df.columns:
                if col == 'Time_Index':
                    continue

                # Try to extract zone from column name
                for zone in self.zones:
                    if zone in col:
                        zones_with_data.append(zone)
                        break

            print(f"  Found nuclear data for zones: {zones_with_data}")

            # Get nuclear resources
            nuclear_resources = [r for r in resource_names.get('Thermal', []) if 'Nuclear' in r]
            nuclear_columns_processed = 0

            # Process each nuclear resource
            for resource in nuclear_resources:
                # Extract zone from resource name
                zone = None
                for part in resource.split('_'):
                    if part in self.zones:
                        zone = part
                        break

                if not zone:
                    print(f"  Cannot identify zone for nuclear resource: {resource}")
                    continue

                # If we have data for this zone, use it
                if zone in zones_with_data:
                    # Find the column containing this zone
                    data_col = next((col for col in nuclear_df.columns if zone in col), None)
                    if data_col:
                        print(f"  Matched nuclear resource {resource} with data column {data_col}")
                        variability_df[resource] = nuclear_df[data_col].values
                        nuclear_columns_processed += 1
                else:
                    # Find closest zone with data
                    if zones_with_data:
                        # Get indices of current zone and zones with data
                        zone_idx = self.zones.index(zone)
                        closest_zone = min(zones_with_data, 
                                           key=lambda z: abs(self.zones.index(z) - zone_idx))

                        # Find the column containing the closest zone
                        data_col = next((col for col in nuclear_df.columns if closest_zone in col), None)
                        if data_col:
                            print(f"  No data for zone {zone}, using closest zone {closest_zone} for {resource}")
                            variability_df[resource] = nuclear_df[data_col].values
                            nuclear_columns_processed += 1
                    else:
                        print(f"  No nuclear data available for any zone - using default for {resource}")
                        variability_df[resource] = 1.0

            print(f"Processed {nuclear_columns_processed} nuclear resources")

        # Step 6: Process hydro resources
        if os.path.exists(data_files['hydro']):
            print(f"Reading hydro data from {data_files['hydro']}...")
            hydro_df = pd.read_csv(data_files['hydro'])

            # Identify which zones have hydro data
            zones_with_data = []
            for col in hydro_df.columns:
                if col == 'Time_Index':
                    continue

                # Try to extract zone from column name
                for zone in self.zones:
                    if zone in col:
                        zones_with_data.append(zone)
                        break

            print(f"  Found hydro data for zones: {zones_with_data}")

            # Get hydro resources
            hydro_resources = resource_names.get('Hydro', [])
            hydro_columns_processed = 0

            # Process each hydro resource
            for resource in hydro_resources:
                # Extract zone from resource name
                zone = None
                for part in resource.split('_'):
                    if part in self.zones:
                        zone = part
                        break

                if not zone:
                    print(f"  Cannot identify zone for hydro resource: {resource}")
                    continue

                # If we have data for this zone, use it
                if zone in zones_with_data:
                    # Find the column containing this zone
                    data_col = next((col for col in hydro_df.columns if zone in col), None)
                    if data_col:
                        print(f"  Matched hydro resource {resource} with data column {data_col}")
                        variability_df[resource] = hydro_df[data_col].values
                        hydro_columns_processed += 1
                else:
                    # Find closest zone with data
                    if zones_with_data:
                        # Get indices of current zone and zones with data
                        zone_idx = self.zones.index(zone)
                        closest_zone = min(zones_with_data, 
                                           key=lambda z: abs(self.zones.index(z) - zone_idx))

                        # Find the column containing the closest zone
                        data_col = next((col for col in hydro_df.columns if closest_zone in col), None)
                        if data_col:
                            print(f"  No data for zone {zone}, using closest zone {closest_zone} for {resource}")
                            variability_df[resource] = hydro_df[data_col].values
                            hydro_columns_processed += 1
                    else:
                        print(f"  No hydro data available for any zone - using default for {resource}")
                        variability_df[resource] = 1.0

            print(f"Processed {hydro_columns_processed} hydro resources")
        # Step 7: Process wind and solar data from merged file
        print("Processing wind and solar data from merged file...")

        # Check if merged file exists
        if os.path.exists(data_files['merged']):
            merged_df = pd.read_csv(data_files['merged'])

            # Convert date to datetime if it's not already
            if 'date' in merged_df.columns and merged_df['date'].dtype == 'object':
                merged_df['date'] = pd.to_datetime(merged_df['date'])

            # Sort by date and hour_index if those columns exist
            if 'date' in merged_df.columns and 'hour_index' in merged_df.columns:
                merged_df = merged_df.sort_values(['date', 'hour_index'])

            # Get all VRE resources
            vre_resources = resource_names.get('Vre', [])
            vre_columns_processed = 0

            for resource in vre_resources:
                # Extract resource type (Wind or Solar) and zone
                is_wind = any(wind_term in resource.lower() for wind_term in ['wind', 'offshore', 'onshore'])
                is_solar = any(solar_term in resource.lower() for solar_term in ['solar', 'pv', 'photovoltaic'])

                # Extract zone
                zone = None
                for part in resource.split('_'):
                    if part in self.zones:
                        zone = part
                        break

                if zone is None:
                    print(f"  Cannot identify zone for VRE resource: {resource}")
                    continue

                # Filter merged_df for this zone
                if 'zone' in merged_df.columns:
                    zone_data = merged_df[merged_df['zone'] == zone]

                    if len(zone_data) == 0:
                        print(f"  No data found for zone {zone} in merged file")
                        continue

                    # Ensure we have 8760 hours of data
                    if len(zone_data) != 8760:
                        print(f"  Warning: Zone {zone} has {len(zone_data)} hours instead of 8760")
                        # Create a proper index to ensure alignment
                        zone_data = zone_data.reset_index(drop=True)
                        if len(zone_data) < 8760:
                            # Pad with last values if needed
                            zone_data = pd.concat([zone_data] + [zone_data.iloc[-1:]] * (8760 - len(zone_data)))
                        zone_data = zone_data.iloc[:8760]

                    # Assign data based on resource type
                    if is_wind and 'Wind' in zone_data.columns:
                        print(f"  Assigning wind data to {resource} from zone {zone}")
                        variability_df[resource] = zone_data['Wind'].values
                        vre_columns_processed += 1
                    elif is_solar and 'Solar' in zone_data.columns:
                        print(f"  Assigning solar data to {resource} from zone {zone}")
                        variability_df[resource] = zone_data['Solar'].values
                        vre_columns_processed += 1
                    else:
                        print(f"  No matching data type found for {resource}")
                else:
                    print("  'zone' column not found in merged file")

            print(f"Processed {vre_columns_processed} VRE resources")
        else:
            print(f"Merged data file not found: {data_files['merged']}")

        # Check for missing resources and set them to 1.0 for all hours
        print("Checking for missing resources...")

        # Get all columns except Time_Index
        existing_resources = [col for col in variability_df.columns if col != 'Time_Index']

        # Identify missing resources
        missing_resources = [res for res in all_resources if res not in existing_resources]
        print(f"Found {len(missing_resources)} resources without variability data")

        # Add missing resources with value 1.0 (always available)
        for resource in missing_resources:
            variability_df[resource] = 1.0
            print(f"  Set resource {resource} to constant 1.0 (always available)")

        # Fill any NaN values with 1.0 (assume full availability if no data)
        variability_df = variability_df.fillna(1.0)

        # Cap values between 0 and 1
        for col in variability_df.columns:
            if col != 'Time_Index':
                variability_df[col] = variability_df[col].clip(0, 1)

        # Save the output
        system_folder = os.path.join(self.case_path, 'system')
        os.makedirs(system_folder, exist_ok=True)
        output_path = os.path.join(system_folder, 'Generators_variability.csv')

        variability_df.to_csv(output_path, index=False)
        print(f"Generators_variability.csv created at {output_path} with {len(variability_df.columns) - 1} resources")

        return variability_df
    def generate_settings_yaml(self):
        """Generate settings YAML files for GenX"""
        data_genx = {
            'OverwriteResults': 0,
            'PrintModel': 0,
            'NetworkExpansion': 0,
            'Trans_Loss_Segments': 0,
            'OperationalReserves': 0,
            'EnergyShareRequirement': 0,
            'CapacityReserveMargin': 0,
            'CO2Cap': 0,
            'StorageLosses': 0,
            'MinCapReq': 0,
            'MaxCapReq': 0,
            'Solver': 'Gurobi',
            'ParameterScale': 0,
            'WriteShadowPrices': 1,
            'UCommit': 2,
            'TimeDomainReductionFolder': 'TDR_Results',
            'TimeDomainReduction': 0,
            'ModelingToGenerateAlternatives': 0,
            'MethodofMorris': 0,
            'OutputFullTimeSeries': 1
        }
        
        data_gurobi = {
            'Feasib_Tol': 1.0e-05,          
            'Optimal_Tol': 1e-5,            
            'TimeLimit': 200000,            
            'Pre_Solve': 1,                
            'Method': -1,                    
            'MIPGap': 1e-3,                  
            'BarConvTol': 1.0e-05,         
            'NumericFocus': 0,              
            'Crossover': -1,                
#             'PreDual': 0,                    
            'AggFill': -1,                  
        }
        
        settings_path = os.path.join(self.case_path, 'settings')
        os.makedirs(settings_path, exist_ok=True)
        
        with open(os.path.join(settings_path, 'genx_settings.yml'), 'w') as file:
            yaml.dump(data_genx, file, default_flow_style=False)
        
        with open(os.path.join(settings_path, 'gurobi_settings.yml'), 'w') as file:
            yaml.dump(data_gurobi, file, default_flow_style=False)
        
        print("Successfully set GenX and Gurobi settings")
    
    def write_run_file(self):
        """Create the Run.jl file for executing the model"""
        julia_code = """
        using GenX
        using Gurobi

        run_genx_case!(dirname(@__FILE__), Gurobi.Optimizer)
        """
        
        with open(os.path.join(self.case_path, 'Run.jl'), 'w') as file:
            file.write(julia_code)
            
        print("Created Run.jl file")
        
    def generate_split_resources(self, atb_summary_file=None):
        """
        Generate resource files with existing and new build resources separated.
        Directly integrates ATB data if provided.

        For each resource:
        1. Create one version for existing capacity (with full O&M costs, no new build allowed)
        2. Create another version for new builds (with investment costs and reduced O&M)

        Args:
            atb_summary_file (str): Path to ATB summary CSV file

        Returns:
            tuple: Containing DataFrames for each resource type
        """
        # First load ATB data if provided
        if atb_summary_file and os.path.exists(atb_summary_file):
            print(f"Loading ATB data from {atb_summary_file}...")
            atb_summary_df = pd.read_csv(atb_summary_file)

            # Create a dictionary from the summary DataFrame for easy access
            for _, row in atb_summary_df.iterrows():
                tech = row['Technology']
                self.technology_params[tech] = {
                    'Ann_Inv_MWYr': row['Ann_Inv_MWYr'],
                    'Fix_OM_MW': row['Fix_OM_MW'],
                    'Var_OM': row['Var_OM']
                }

                # Add technology-specific parameters
                if tech == 'CombinedCycle':
                    self.technology_params[tech].update({
                        'Min_Power': 0.4,
                        'Ramp_Up_Percentage': 0.35,
                        'Ramp_Dn_Percentage': 0.35,
                        'Up_Time': 6,
                        'Down_Time': 4,
                        'Startup_Cost_per_MW': 80,
                        'Shutdown_Cost_per_MW': 40,
                        'Heat_Rate_MMBTU_per_MWh': 6.43
                    })
                elif tech == 'CombustionTurbine':
                    self.technology_params[tech].update({
                        'Min_Power': 0.3,
                        'Ramp_Up_Percentage': 1.0,
                        'Ramp_Dn_Percentage': 1.0,
                        'Up_Time': 1,
                        'Down_Time': 1,
                        'Startup_Cost_per_MW': 60,
                        'Shutdown_Cost_per_MW': 30,
                        'Heat_Rate_MMBTU_per_MWh': 9.5
                    })
                elif tech == 'InternalCombustion':
                    self.technology_params[tech].update({
                        'Min_Power': 0.2,
                        'Ramp_Up_Percentage': 1.0,
                        'Ramp_Dn_Percentage': 1.0,
                        'Up_Time': 1,
                        'Down_Time': 1,
                        'Startup_Cost_per_MW': 30,
                        'Shutdown_Cost_per_MW': 15,
                        'Heat_Rate_MMBTU_per_MWh': 9.75
                    })
                elif tech == 'SteamTurbine':
                    self.technology_params[tech].update({
                        'Min_Power': 0.3,
                        'Ramp_Up_Percentage': 0.25,
                        'Ramp_Dn_Percentage': 0.25,
                        'Up_Time': 12,
                        'Down_Time': 12,
                        'Startup_Cost_per_MW': 100,
                        'Shutdown_Cost_per_MW': 50,
                        'Heat_Rate_MMBTU_per_MWh': 10.0
                    })
                elif tech == 'Nuclear':
                    self.technology_params[tech].update({
                        'Min_Power': 0.4,
                        'Ramp_Up_Percentage': 0.2,
                        'Ramp_Dn_Percentage': 0.2,
                        'Up_Time': 48,
                        'Down_Time': 48,
                        'Startup_Cost_per_MW': 200,
                        'Shutdown_Cost_per_MW': 100,
                        'Heat_Rate_MMBTU_per_MWh': 10.46
                    })
                elif tech == 'Battery':
                    self.technology_params[tech].update({
                        'Eff_Up': 0.92,
                        'Eff_Down': 0.92,
                        'Self_Disch': 0.0004,
                        'Min_Duration': 2,
                        'Max_Duration': 8
                    })
                elif (tech == 'Hydro' ) |( tech == 'PumpedHydro'):
                    self.technology_params[tech].update({
                        'Min_Power': 0.1,
                        'Ramp_Up_Percentage': 0.5,
                        'Ramp_Dn_Percentage': 0.5,
                        'Hydro_Energy_to_Power_Ratio': 12,
                        'LDS': 1
                    })
                elif tech in ['WindLand', 'WindOffshore', 'SolarUtility', 'SolarBTM']:
                    self.technology_params[tech].update({
                        'Num_VRE_bins': 1
                    })

            print(f"Integrated cost data for {len(atb_summary_df)} technologies")
        else:
            print("No ATB data file provided or file not found. Using current technology parameters.")

        # Lists to hold resource data
        thermal_data = []
        vre_data = []
        hydro_data = []
        storage_data = []

        # Define technologies that can have new builds
        new_build_techs = {
            'CombinedCycle', 'CombustionTurbine', 'WindLand', 'WindOffshore', 
            'SolarUtility', 'SolarBTM', 'Battery', 'Nuclear'
        }

        # O&M cost reduction factor for new builds (percent of full O&M)
        # New units have lower O&M because they don't need major part replacements
        new_build_om_factor = 0.5

        # Process thermal resources
        print("Processing thermal resources...")
        for resource_name, resource_info in self.thermal_resources.items():
            zone = resource_info['zone']
            tech = resource_info['tech']
            fuel = resource_info['fuel']
            capacity = resource_info['capacity']

            # Skip entries with zero or missing capacity
            if capacity <= 0:
                continue

            # Map fuel type to GenX fuel name
            fuel_name = self.map_fuel_to_genx(fuel, zone)
            # Get technology parameters
            tech_key = f"{tech}_{fuel}"
            if tech_key in self.technology_params:
                tech_params = self.technology_params[tech_key]
            elif tech in self.technology_params:
                tech_params = self.technology_params[tech]
            else:
                print(f"Warning: No technology parameters for {tech} or {tech_key}. Using defaults.")
                tech_params = {
                    'Ann_Inv_MWYr': 100000,
                    'Fix_OM_MW': 30000,
                    'Var_OM': 5.0,
                    'Min_Power': 0.3,
                    'Ramp_Up_Percentage': 0.5,
                    'Ramp_Dn_Percentage': 0.5,
                    'Up_Time': 4,
                    'Down_Time': 4,
                    'Startup_Cost_per_MW': 50,
                    'Shutdown_Cost_per_MW': 25,
                    'Heat_Rate_MMBTU_per_MWh': 10.0
                }

            # Apply zonal scaling factor
            scale_factor = self.zonal_scale_factors[zone]

            # Create the existing capacity resource (no new build allowed)
            existing_resource = {
                'Resource': f"{resource_name}_Existing",
                'Zone': self.zone_to_number[zone],
                'region': zone,
                'cluster': 1,
                'Technology': tech,
                'New_Build': 0,  # No new build for existing
                'Can_Retire': 1,  # All existing thermal units can retire
                'Existing_Cap_MW': capacity,
                'Max_Cap_MW': capacity,  # No expansion
                'Min_Cap_MW': 0,  # Can be fully retired
                'Min_Power': tech_params.get('Min_Power', 0.3),
                'Ramp_Up_Percentage': tech_params.get('Ramp_Up_Percentage', 0.5),
                'Ramp_Dn_Percentage': tech_params.get('Ramp_Dn_Percentage', 0.5),
                'Shutdown_Cost_per_MW': tech_params.get('Shutdown_Cost_per_MW', 25),
                'Startup_Cost_per_MW': tech_params.get('Startup_Cost_per_MW', 50),
                'Up_Time': tech_params.get('Up_Time', 4),
                'Down_Time': tech_params.get('Down_Time', 4),
                'Inv_Cost_per_MWyr': 0,  # No investment cost for existing
                'Fixed_OM_Cost_per_MWyr': tech_params.get('Fix_OM_MW', 30000) * scale_factor,
                'Var_OM_Cost_per_MWh': tech_params.get('Var_OM', 5.0),
                'Heat_Rate_MMBTU_per_MWh': tech_params.get('Heat_Rate_MMBTU_per_MWh', 10.0),
                'Fuel': fuel_name,
                'Model': 1  # Default model
            }
            thermal_data.append(existing_resource)

            # Create new build version only for applicable technologies
            if tech in new_build_techs:
                new_resource = existing_resource.copy()
                new_resource['Resource'] = f"{resource_name}_New"
                new_resource['New_Build'] = 1
                new_resource['Existing_Cap_MW'] = 0
                new_resource['Max_Cap_MW'] = -1

                # Include investment cost for new builds
                new_resource['Inv_Cost_per_MWyr'] = tech_params.get('Ann_Inv_MWYr', 100000) * scale_factor

                # Reduce O&M cost for new builds
                new_resource['Fixed_OM_Cost_per_MWyr'] = tech_params.get('Fix_OM_MW', 30000) * scale_factor * new_build_om_factor

                thermal_data.append(new_resource)

        # Process nuclear plants if they exist in technology capacities
        if 'Nuclear' in self.technology_capacities:
            for zone in self.zones:
                capacity = self.technology_capacities['Nuclear'].get(zone, 0)
#                 if 'Nuclear' in new_build_techs:
#                     new_nuclear = {
#                         'Resource': f'Nuclear_{zone}_New',
#                         'Zone': self.zone_to_number[zone],
#                         'region': zone,
#                         'Model': 1,
#                         'cluster': 1,
#                         'Technology': 'Nuclear',
#                         'New_Build': 1,
#                         'Can_Retire': 1, 
#                         'Existing_Cap_MW': 0,
#                         'Max_Cap_MW': -1, 
#                         'Min_Cap_MW': 0,  
#                         'Min_Power': tech_params.get('Min_Power', 0.4),
#                         'Ramp_Up_Percentage': tech_params.get('Ramp_Up_Percentage', 0.2),
#                         'Ramp_Dn_Percentage': tech_params.get('Ramp_Dn_Percentage', 0.2),
#                         'Shutdown_Cost_per_MW': tech_params.get('Shutdown_Cost_per_MW', 100),
#                         'Startup_Cost_per_MW': tech_params.get('Startup_Cost_per_MW', 200),
#                         'Up_Time': tech_params.get('Up_Time', 24),
#                         'Down_Time': tech_params.get('Down_Time', 24),
#                         'Inv_Cost_per_MWyr': tech_params.get('Ann_Inv_MWYr', 600000) * scale_factor,
#                         'Fixed_OM_Cost_per_MWyr': tech_params.get('Fix_OM_MW', 120000) * scale_factor * new_build_om_factor,
#                         'Var_OM_Cost_per_MWh': tech_params.get('Var_OM', 2.3),
#                         'Heat_Rate_MMBTU_per_MWh': tech_params.get('Heat_Rate_MMBTU_per_MWh', 10.46),
#                         'Fuel': 'None'
#                         }
#                     thermal_data.append(new_nuclear)
                if capacity > 0:

                    nuclear_resource = {
                        'Resource': f'Nuclear_{zone}_Existing',
                        'Zone': self.zone_to_number[zone],
                        'region': zone,
                        'Model': 1,
                        'cluster': 1,
                        'Technology': 'Nuclear',
                        'New_Build': 0,  # No new nuclear builds for existing
                        'Can_Retire': 1,  # Can retire existing nuclear
                        'Existing_Cap_MW': capacity,
                        'Max_Cap_MW': capacity,  # No expansion
                        'Min_Cap_MW': 0,  # Can be fully retired
                        'Min_Power': tech_params.get('Min_Power', 0.4),
                        'Ramp_Up_Percentage': tech_params.get('Ramp_Up_Percentage', 0.2),
                        'Ramp_Dn_Percentage': tech_params.get('Ramp_Dn_Percentage', 0.2),
                        'Shutdown_Cost_per_MW': tech_params.get('Shutdown_Cost_per_MW', 100),
                        'Startup_Cost_per_MW': tech_params.get('Startup_Cost_per_MW', 200),
                        'Up_Time': tech_params.get('Up_Time', 24),
                        'Down_Time': tech_params.get('Down_Time', 24),
                        'Inv_Cost_per_MWyr': 0,  # No investment cost for existing
                        'Fixed_OM_Cost_per_MWyr': tech_params.get('Fix_OM_MW', 120000) * scale_factor,
                        'Var_OM_Cost_per_MWh': tech_params.get('Var_OM', 2.3),
                        'Heat_Rate_MMBTU_per_MWh': tech_params.get('Heat_Rate_MMBTU_per_MWh', 10.46),
                        'Fuel': 'None'
                    }
                    thermal_data.append(nuclear_resource)

                    # Add new nuclear option if applicable
                    if 'Nuclear' in new_build_techs:
                        new_nuclear = nuclear_resource.copy()
                        new_nuclear['Resource'] = f'Nuclear_{zone}_New'
                        new_nuclear['New_Build'] = 1
                        new_nuclear['Existing_Cap_MW'] = 0
                        new_nuclear['Max_Cap_MW'] = -1
                        new_nuclear['Inv_Cost_per_MWyr'] = tech_params.get('Ann_Inv_MWYr', 600000) * scale_factor
                        new_nuclear['Fixed_OM_Cost_per_MWyr'] = tech_params.get('Fix_OM_MW', 120000) * scale_factor * new_build_om_factor
                        thermal_data.append(new_nuclear)

        # Process VRE technologies
        print("Processing VRE resources...")
        vre_techs = {
            'WindLand', 'WindOffshore', 'SolarUtility', 'SolarBTM'
        }

        for tech in vre_techs:
            for zone in self.zones:
                    if tech in self.technology_capacities:
                        capacity = self.technology_capacities[tech].get(zone, 0)
                    else:
                        capacity = 0 
                    scale_factor = self.zonal_scale_factors[zone]
                    tech_params = self.technology_params.get(tech, {})
                        # Use defaults if parameters not found
                    ann_inv = tech_params.get('Ann_Inv_MWYr', 100000) * scale_factor
                    fix_om = tech_params.get('Fix_OM_MW', 30000) * scale_factor
                    var_om = tech_params.get('Var_OM', 0)
                    if capacity > 0:
                        # Existing capacity resource
                        existing_resource = {
                            'Resource': f'{tech}_{zone}_Existing',
                            'Zone': self.zone_to_number[zone],
                            'region': zone,
                            'cluster': 1,
                            'Technology': tech,
                            'New_Build': 0,
                            'Can_Retire': 1,
                            'Existing_Cap_MW': capacity,
                            'Max_Cap_MW': capacity,
                            'Min_Cap_MW': 0,
                            'Inv_Cost_per_MWyr': 0,  # No investment cost for existing
                            'Fixed_OM_Cost_per_MWyr': fix_om,
                            'Var_OM_Cost_per_MWh': var_om,
                            'Heat_Rate_MMBTU_per_MWh': 0,
                            'Fuel': 'None',
                            'Num_VRE_bins': 1
                        }
                        vre_data.append(existing_resource)
                    if (zone in ['J', 'K']) & (tech == 'WindOffshore'):
                        new_resource = {
                            'Resource': f'{tech}_{zone}_New',
                            'Zone': self.zone_to_number[zone],
                            'region': zone,
                            'cluster': 1,
                            'Technology': tech,
                            'New_Build': 1,
                            'Can_Retire': 1,
                            'Existing_Cap_MW':0,
                            'Max_Cap_MW': -1,
                            'Min_Cap_MW': 0,
                            'Inv_Cost_per_MWyr': ann_inv,  # No investment cost for existing
                            'Fixed_OM_Cost_per_MWyr': fix_om * new_build_om_factor,
                            'Var_OM_Cost_per_MWh': var_om,
                            'Heat_Rate_MMBTU_per_MWh': 0,
                            'Fuel': 'None',
                            'Num_VRE_bins': 1
                        }
                        vre_data.append(new_resource)
                    elif tech != 'WindOffshore':
                        new_resource = {
                            'Resource': f'{tech}_{zone}_New',
                            'Zone': self.zone_to_number[zone],
                            'region': zone,
                            'cluster': 1,
                            'Technology': tech,
                            'New_Build': 1,
                            'Can_Retire': 1,
                            'Existing_Cap_MW': 0,
                            'Max_Cap_MW': -1,
                            'Min_Cap_MW': 0,
                            'Inv_Cost_per_MWyr': ann_inv,  # No investment cost for existing
                            'Fixed_OM_Cost_per_MWyr': fix_om * new_build_om_factor,
                            'Var_OM_Cost_per_MWh': var_om,
                            'Heat_Rate_MMBTU_per_MWh': 0,
                            'Fuel': 'None',
                            'Num_VRE_bins': 1
                        }
                        vre_data.append(new_resource)


        # Process Hydro
        print("Processing hydro resources...")
        if 'Hydro' in self.technology_capacities:
            for zone in self.zones:
                capacity = self.technology_capacities['Hydro'].get(zone, 0)
                if capacity > 0:
                    scale_factor = self.zonal_scale_factors[zone]

                    # Get technology parameters
                    tech_params = self.technology_params.get('Hydro', {})

                    # Use defaults if parameters not found
                    fix_om = tech_params.get('Fix_OM_MW', 45000) * scale_factor
                    var_om = tech_params.get('Var_OM', 2.5)
                    min_power = tech_params.get('Min_Power', 0.1)
                    ramp_up = tech_params.get('Ramp_Up_Percentage', 0.5)
                    ramp_down = tech_params.get('Ramp_Dn_Percentage', 0.5)
                    hydro_e2p = tech_params.get('Hydro_Energy_to_Power_Ratio', 12)

                    # Hydro is typically only existing
                    hydro_data.append({
                        'Resource': f'Hydro_{zone}',
                        'Zone': self.zone_to_number[zone],
                        'region': zone,
                        'cluster': 1,
                        'Technology': 'Hydro',
                        'New_Build': 0,  # No new hydro builds
                        'Can_Retire': 0,  # Typically can't retire hydro
                        'Existing_Cap_MW': capacity,
                        'Max_Cap_MW': capacity,  # No expansion
                        'Min_Cap_MW': capacity,  # Can't retire, so min = existing
                        'Inv_Cost_per_MWyr': 0,  # No investment cost for existing
                        'Fixed_OM_Cost_per_MWyr': fix_om,
                        'Var_OM_Cost_per_MWh': var_om,
                        'Heat_Rate_MMBTU_per_MWh': 0,
                        'Fuel': 'None',
                        'Min_Power': min_power,
                        'Ramp_Up_Percentage': ramp_up,
                        'Ramp_Dn_Percentage': ramp_down,
                        'Hydro_Energy_to_Power_Ratio': hydro_e2p,
                        'LDS': 1  # Long duration storage
                    })

        # Process Pumped Hydro
        if 'PumpedHydro' in self.technology_capacities:
            for zone in self.zones:
                capacity = self.technology_capacities['PumpedHydro'].get(zone, 0)
                if capacity > 0:
                    scale_factor = self.zonal_scale_factors[zone]

                    # Get technology parameters
                    tech_params = self.technology_params.get('PumpedHydro', {})

                    # Use defaults if parameters not found
                    fix_om = tech_params.get('Fix_OM_MW', 50000) * scale_factor
                    var_om = tech_params.get('Var_OM', 2.0)
                    eff_up = tech_params.get('Eff_Up', 0.85)
                    eff_down = tech_params.get('Eff_Down', 0.85)
                    self_disch = tech_params.get('Self_Disch', 0.0001)
                    min_duration = tech_params.get('Min_Duration', 8)
                    max_duration = tech_params.get('Max_Duration', 12)

                    # Calculate energy capacity based on assumed duration
                    energy_capacity = capacity * 10  # Assuming 10 hours of storage

                    # Pumped hydro is typically only existing
                    hydro_data.append({
                        'Resource': f'PumpedHydro_{zone}',
                        'Zone': self.zone_to_number[zone],
                        'region': zone,
                        'cluster': 1,
                        'Technology': 'Hydro',
                        'New_Build': 0,  # No new hydro builds
                        'Can_Retire': 0,  # Typically can't retire hydro
                        'Existing_Cap_MW': capacity,
                        'Max_Cap_MW': capacity,  # No expansion
                        'Min_Cap_MW': capacity,  # Can't retire, so min = existing
                        'Inv_Cost_per_MWyr': 0,  # No investment cost for existing
                        'Fixed_OM_Cost_per_MWyr': fix_om,
                        'Var_OM_Cost_per_MWh': var_om,
                        'Heat_Rate_MMBTU_per_MWh': 0,
                        'Fuel': 'None',
                        'Min_Power': min_power,
                        'Ramp_Up_Percentage': ramp_up,
                        'Ramp_Dn_Percentage': ramp_down,
                        'Hydro_Energy_to_Power_Ratio': hydro_e2p,
                        'LDS': 1  # Long duration storage
                    })

        # Process Battery storage
        print("Processing battery resources...")
        if 'Battery' in self.technology_capacities:
            for zone in self.zones:
                capacity = self.technology_capacities['Battery'].get(zone, 0)
                
                scale_factor = self.zonal_scale_factors[zone]

                    # Get technology parameters
                tech_params = self.technology_params.get('Battery', {})

                # Calculate power and energy cost components
                # For 4-hour battery: Total cost = Power cost + (4 * Energy cost)
                # Assuming 60% of cost is power-related, 40% is energy-related for a 4-hour system
                power_cost_fraction = 0.6
                energy_cost_fraction = 0.4 / 4  # per MWh for a 4-hour system

                total_ann_inv = tech_params.get('Ann_Inv_MWYr', 70000)
                power_cost = total_ann_inv * power_cost_fraction
                energy_cost = total_ann_inv * energy_cost_fraction

                total_fixed_om = tech_params.get('Fix_OM_MW', 20000)
                power_fixed_om = total_fixed_om * power_cost_fraction
                energy_fixed_om = total_fixed_om * energy_cost_fraction

                # Get efficiency and other parameters
                eff_up = tech_params.get('Eff_Up', 0.92)
                eff_down = tech_params.get('Eff_Down', 0.92)
                self_disch = tech_params.get('Self_Disch', 0.0004)
                min_duration = tech_params.get('Min_Duration', 2)
                max_duration = tech_params.get('Max_Duration', 8)
                if capacity > 0:
                    # Existing capacity
                    existing_resource = {
                        'Resource': f'Battery_{zone}_Existing',
                        'Zone': self.zone_to_number[zone],
                        'region': zone,
                        'cluster': 1,
                        'Technology': 'Battery',
                        'New_Build': 0,
                        'Can_Retire': 1,
                        'Model': 1,  # Symmetric charge/discharge
                        'LDS': 0,  # Not long-duration storage
                        'Existing_Cap_MW': capacity,
                        'Existing_Cap_MWh': capacity * 4,  # Assuming 4-hour batteries
                        'Max_Cap_MW': capacity,
                        'Max_Cap_MWh': capacity * 4,
                        'Min_Cap_MW': 0,
                        'Min_Cap_MWh': 0,
                        'Inv_Cost_per_MWyr': 0,  # No investment cost for existing
                        'Inv_Cost_per_MWhyr': 0,  # No investment cost for existing
                        'Fixed_OM_Cost_per_MWyr': power_fixed_om * scale_factor,
                        'Fixed_OM_Cost_per_MWhyr': energy_fixed_om * scale_factor,
                        'Var_OM_Cost_per_MWh': 0,
                        'Var_OM_Cost_per_MWhIn': 0,
                        'Heat_Rate_MMBTU_per_MWh': 0,
                        'Fuel': 'None',
                        'Self_Disch': self_disch,
                        'Eff_Up': eff_up,
                        'Eff_Down': eff_down,
                        'Min_Duration': min_duration,
                        'Max_Duration': max_duration
                    }
                    storage_data.append(existing_resource)
                new_resource = {
                        'Resource': f'Battery_{zone}_New',
                        'Zone': self.zone_to_number[zone],
                        'region': zone,
                        'cluster': 1,
                        'Technology': 'Battery',
                        'New_Build': 1,
                        'Can_Retire': 1,
                        'Model': 1,  # Symmetric charge/discharge
                        'LDS': 0,  # Not long-duration storage
                        'Existing_Cap_MW': 0,
                        'Existing_Cap_MWh': 0,
                        'Max_Cap_MW': -1,
                        'Max_Cap_MWh': -1,
                        'Min_Cap_MW': 0,
                        'Min_Cap_MWh': 0,
                        'Inv_Cost_per_MWyr': power_cost * scale_factor,  
                        'Inv_Cost_per_MWhyr': energy_cost * scale_factor,  
                        'Fixed_OM_Cost_per_MWyr': power_fixed_om * scale_factor * new_build_om_factor,
                        'Fixed_OM_Cost_per_MWhyr': energy_fixed_om * scale_factor * new_build_om_factor,
                        'Var_OM_Cost_per_MWh': 0,
                        'Var_OM_Cost_per_MWhIn': 0,
                        'Heat_Rate_MMBTU_per_MWh': 0,
                        'Fuel': 'None',
                        'Self_Disch': self_disch,
                        'Eff_Up': eff_up,
                        'Eff_Down': eff_down,
                        'Min_Duration': min_duration,
                        'Max_Duration': max_duration
                    }
                storage_data.append(new_resource)

        # Create resource CSVs
        resources_folder = os.path.join(self.case_path, 'resources')
        os.makedirs(resources_folder, exist_ok=True)

        # Save resources to CSVs
        thermal_df = pd.DataFrame(thermal_data)
        vre_df = pd.DataFrame(vre_data)
        hydro_df = pd.DataFrame(hydro_data)
        storage_df = pd.DataFrame(storage_data)

        # Save to CSVs
        thermal_df.to_csv(os.path.join(resources_folder, "Thermal.csv"), index=False)
        vre_df.to_csv(os.path.join(resources_folder, "Vre.csv"), index=False)
        hydro_df.to_csv(os.path.join(resources_folder, "Hydro.csv"), index=False)
        storage_df.to_csv(os.path.join(resources_folder, "Storage.csv"), index=False)

        print(f"Created resource files with split existing/new resources:")
        print(f"  Thermal: {len(thermal_data)} resources")
        print(f"  VRE: {len(vre_data)} resources")
        print(f"  Hydro: {len(hydro_data)} resources")
        print(f"  Storage: {len(storage_data)} resources")

        return thermal_df, vre_df, hydro_df, storage_df
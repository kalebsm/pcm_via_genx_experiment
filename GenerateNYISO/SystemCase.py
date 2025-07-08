#!/usr/bin/env python3
"""
NYISO System-Level GenX Case Generator

This script generates a GenX case for the NYISO system at the system level (single zone),
aggregating data from the original 11-zone model.
"""

import os
import csv
import shutil
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

class NYISOSystemGenerator:
    """Class for generating a system-level GenX case for NYISO"""
    
    def __init__(self, case_name="NYISO_System", base_dir=None):
        """
        Initialize the NYISO GenX case generator for system-level aggregation
        
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
        
        # Define the NYISO zones (for reference)
        self.zones = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
        
        # Initialize technology capacities
        self.technology_capacities = {}
        
        # Keep track of all technologies and resource types
        self.all_technologies = set()
        self.thermal_technologies = {
            'CombinedCycle', 
            'CombustionTurbine', 
            'SteamTurbine'
        }
        
        # Technology parameters
        self.technology_params = {}
        
        # Thermal resources
        self.thermal_resources = {}
        
        # Fuel mapping - simplified for system level
        self.system_fuels = {
            'NG': 'NG_NY',       # Natural gas, system-wide
            'FO2': 'FO2_NY',     # Fuel oil 2, system-wide
            'FO6': 'FO6_NY',     # Fuel oil 6, system-wide
            'Coal': 'coal_NY',   # Coal, system-wide
#             'Kerosene': 'Kerosene_NY', # Kerosene, system-wide
            'None': 'None'       # For renewables
        }
        
        # NYCA-level cost scaling factor
        self.system_scale_factor = 1.0  # Using NYCA as reference
    
    def load_technology_capacities(self, data=None, from_11zone=None):
        """
        Load technology capacities from provided data or from 11-zone model
        
        Args:
            data (DataFrame or dict): Direct data containing technology capacities
            from_11zone (NYISOGenXGenerator): 11-zone model instance to extract data from
        """
        if from_11zone is not None:
            # Extract and aggregate capacities from 11-zone model
            print("Aggregating technology capacities from 11-zone model...")
            
            # Aggregate technology capacities across zones
            for tech, zone_caps in from_11zone.technology_capacities.items():
                total_capacity = sum(zone_caps.values())
                if total_capacity > 0:
                    if tech not in self.technology_capacities:
                        self.technology_capacities[tech] = total_capacity
                    else:
                        self.technology_capacities[tech] += total_capacity
                    
                    # Add to all technologies
                    self.all_technologies.add(tech)
            
            # Aggregate thermal resources by technology and fuel
            thermal_by_tech_fuel = {}
            for resource_name, resource_info in from_11zone.thermal_resources.items():
                tech = resource_info['tech']
                fuel = resource_info['fuel']
                capacity = resource_info['capacity']
                
                tech_fuel_key = f"{tech}_{fuel}"
                if tech_fuel_key not in thermal_by_tech_fuel:
                    thermal_by_tech_fuel[tech_fuel_key] = {
                        'tech': tech,
                        'fuel': fuel,
                        'capacity': 0
                    }
                
                thermal_by_tech_fuel[tech_fuel_key]['capacity'] += capacity
            
            # Convert aggregated thermal resources to our format
            for tech_fuel_key, info in thermal_by_tech_fuel.items():
                if info['capacity'] > 0:
                    resource_name = f"{info['tech']}_{info['fuel']}_NY"
                    self.thermal_resources[resource_name] = {
                        'zone': 'NY',  # System level
                        'tech': info['tech'],
                        'fuel': info['fuel'],
                        'capacity': info['capacity']
                    }
            
            print(f"Aggregated {len(self.technology_capacities)} technologies and {len(self.thermal_resources)} thermal resources")
            
        elif isinstance(data, dict) and 'capacity_by_tech' in data:
            # Direct system-level data
            capacity_by_tech = data['capacity_by_tech']
            
            # Extract technology capacities
            for tech_key, capacity in capacity_by_tech.items():
                # Determine if this is a thermal resource with fuel type
                if '_' in tech_key and any(thermal_tech in tech_key for thermal_tech in self.thermal_technologies):
                    tech, fuel = tech_key.split('_', 1)
                    resource_name = f"{tech}_{fuel}_NY"
                    
                    # Add to thermal resources
                    self.thermal_resources[resource_name] = {
                        'zone': 'NY', 
                        'tech': tech, 
                        'fuel': fuel, 
                        'capacity': capacity
                    }
                    
                    # Also add to general technology capacities
                    if tech not in self.technology_capacities:
                        self.technology_capacities[tech] = 0
                    
                    # Add capacity
                    self.technology_capacities[tech] = capacity + self.technology_capacities.get(tech, 0)
                    
                    # Add to all technologies
                    self.all_technologies.add(tech)
                else:
                    # Non-thermal resource
                    tech = tech_key
                    
                    # Add to technology capacities
                    self.technology_capacities[tech] = capacity
                    
                    # Add to all technologies
                    self.all_technologies.add(tech)
            
            print(f"Loaded system-level technology capacities for {len(self.technology_capacities)} technologies")
            
        elif isinstance(data, pd.DataFrame):
            # Expected columns: Technology, Fuel, Capacity_MW
            required_cols = ['Technology', 'Capacity_MW']
            if not all(col in data.columns for col in required_cols):
                print(f"Error: DataFrame must contain columns {required_cols}")
                return
            
            # Handle Fuel column if present
            has_fuel = 'Fuel' in data.columns
            
            # Process each row
            for _, row in data.iterrows():
                tech = row['Technology']
                capacity = row['Capacity_MW']
                
                # Add to all technologies
                self.all_technologies.add(tech)
                
                # Check if this is a thermal technology
                is_thermal = tech in self.thermal_technologies
                
                if is_thermal and has_fuel and row['Fuel'] != 'None':
                    # This is a thermal resource with fuel
                    fuel = row['Fuel']
                    resource_name = f"{tech}_{fuel}_NY"
                    
                    # Add to thermal resources
                    self.thermal_resources[resource_name] = {
                        'zone': 'NY', 
                        'tech': tech, 
                        'fuel': fuel, 
                        'capacity': capacity
                    }
                
                # Add to general technology capacities
                if tech not in self.technology_capacities:
                    self.technology_capacities[tech] = 0
                
                # Add capacity
                self.technology_capacities[tech] = capacity + self.technology_capacities.get(tech, 0)
            
            print(f"Loaded system-level technology capacities for {len(self.technology_capacities)} technologies")
        
        else:
            print("Error: Unsupported data format or no data provided.")
    
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
            readme.write(f"# {self.case_name}\n\nThis is a GenX case for NYISO aggregated at the system level (single zone).")
        
        # Create empty Run.jl file
        with open(f"{self.case_path}/Run.jl", "w") as run_file:
            run_file.write("# GenX Run file for NYISO system-level case\n")
        
        print(f"GenX case structure for {self.case_name} created successfully")
    
    def create_network_csv(self):
        """Create a Network.csv file for a single-zone system"""
        output_file = f"{self.case_path}/system/Network.csv"
        
        # Ensure system directory exists
        system_dir = os.path.dirname(output_file)
        os.makedirs(system_dir, exist_ok=True)
        
        # For a single zone, we create a minimal Network.csv with just the zone definition
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write the header row
            writer.writerow(["Network_zones", "Network_Lines", "Start_Zone", "End_Zone", "Line_Max_Flow_MW"])
            
            # Write a single zone (no network lines since it's a single zone)
            writer.writerow(["z1", "1", "1", "1", "-1"])
        
        print(f"Successfully created single-zone {output_file}")
    
    def create_demand_data_csv(self, year="2019", from_11zone=None):
        """
        Create Demand_data.csv file for GenX by aggregating 11-zone demand data
        
        Args:
            year (str): Year for data
            from_11zone (object): 11-zone model to extract data from
        """
        input_file = os.path.join(self.data_path, f"loadHourly_{year}.csv")
        output_file = f"{self.case_path}/system/Demand_data.csv"
        
        # Read the hourly load data
        print(f"Reading load data from {input_file}...")
        
        if os.path.exists(input_file):
            df = pd.read_csv(input_file)
            
            # Convert timestamp to datetime
            df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
            
            # Sum load across all zones for each timestamp
            system_load = df.groupby('TimeStamp')['Load'].sum().reset_index()
            
            # Check that we have the correct number of hours
            n_hours = len(system_load)
            print(f"Dataset contains {n_hours} hours of data")
            
            # Create the output dataframe
            metadata_rows = pd.DataFrame({
                'Voll': [10000],
                'Demand_Segment': [1],
                'Cost_of_Demand_Curtailment_per_MW': [1],
                'Max_Demand_Curtailment': [1],
                'Rep_Periods': [1],
                'Timesteps_per_Rep_Period': [n_hours],
                'Sub_Weights': [n_hours]
            })
            
            # Create time series data
            timeseries_rows = pd.DataFrame()
            timeseries_rows['Time_Index'] = list(range(1, n_hours + 1))
            timeseries_rows['Demand_MW_z1'] = system_load['Load'].values
            
            # Concatenate metadata and time series
            demand_data = pd.concat([metadata_rows, timeseries_rows], axis=1)
            demand_data.to_csv(output_file, index=False)
            
            print(f"Demand_data.csv created at {output_file}")
        
        elif from_11zone is not None:
            # Try to extract from 11-zone data if available
            eleven_zone_demand_file = os.path.join(from_11zone.case_path, "system/Demand_data.csv")
            
            if os.path.exists(eleven_zone_demand_file):
                print(f"Reading load data from 11-zone file: {eleven_zone_demand_file}")
                eleven_zone_demand = pd.read_csv(eleven_zone_demand_file)
                
                # Identify demand columns
                demand_cols = [col for col in eleven_zone_demand.columns if col.startswith('Demand_MW_z')]
                
                # First few rows are metadata
                metadata_rows = eleven_zone_demand.iloc[:1].copy()
                timeseries_rows = eleven_zone_demand.iloc[1:].copy()
                
                # Sum across all zones for each timestamp
                if 'Time_Index' in timeseries_rows.columns:
                    system_demand = timeseries_rows[demand_cols].sum(axis=1)
                    
                    # Create new dataframe with single zone
                    system_demand_data = metadata_rows.copy()
                    
                    # Replace multiple zone columns with a single zone column
                    for col in demand_cols:
                        if col in system_demand_data.columns:
                            system_demand_data.drop(columns=[col], inplace=True)
                    
                    system_demand_data['Demand_MW_z1'] = [0]  # Placeholder in metadata row
                    
                    # Create time series part
                    system_timeseries = timeseries_rows[['Time_Index']].copy()
                    system_timeseries['Demand_MW_z1'] = system_demand.values
                    
                    # Combine metadata and time series
                    system_demand_full = pd.concat([system_demand_data, system_timeseries], ignore_index=True)
                    system_demand_full.to_csv(output_file, index=False)
                    
                    print(f"Aggregated Demand_data.csv created at {output_file}")
                else:
                    print("Error: Time_Index column not found in 11-zone demand data")
            else:
                print(f"Error: 11-zone demand file not found at {eleven_zone_demand_file}")
        else:
            print(f"Error: Could not find input file {input_file} and no 11-zone data provided")
    
    def create_fuels_data_csv(self, year="2019", from_11zone=None):
        """
        Create Fuels_data.csv file for GenX at the system level
        
        Args:
            year (str): Year for data
            from_11zone (object): 11-zone model to extract data from
        """
        input_file = os.path.join(self.data_path, f"fuelPriceWeekly_{year}.csv")
        output_file = f"{self.case_path}/system/Fuels_data.csv"

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # If we have direct access to the fuel price data, use it
        if os.path.exists(input_file):
            print(f"Reading fuel price data from {input_file}...")
            df = pd.read_csv(input_file)

            # Convert timestamp to datetime
            df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])

            # Get the list of fuels (all columns except TimeStamp)
            original_fuel_columns = [col for col in df.columns if col != 'TimeStamp']
            
            # Map to the system-level fuels
            system_fuel_mapping = {
                'NG_A2E': 'NG_NY',
                'NG_F2I': 'NG_NY',
                'NG_J': 'NG_NY',
                'NG_K': 'NG_NY',
                'FO2_UPNY': 'FO2_NY',
                'FO2_DSNY': 'FO2_NY',
                'FO6_UPNY': 'FO6_NY',
                'FO6_DSNY': 'FO6_NY',
                'coal_NY': 'coal_NY',
#                 'Kerosene_NY': 'Kerosene_NY'
            }
            
            # Process to create system-wide fuel prices
            system_fuels = list(set(system_fuel_mapping.values())) + ['None']
            system_df = pd.DataFrame({'TimeStamp': df['TimeStamp']})
            
            # For each system fuel, calculate the average price across source fuels
            for system_fuel in system_fuels:
                if system_fuel == 'None':
                    system_df[system_fuel] = 0
                    continue
                    
                # Find all source fuels that map to this system fuel
                source_fuels = [fuel for fuel, mapped in system_fuel_mapping.items() 
                              if mapped == system_fuel and fuel in original_fuel_columns]
                
                if source_fuels:
                    # Average the prices
                    system_df[system_fuel] = df[source_fuels].mean(axis=1)
                else:
                    print(f"Warning: No source fuels found for {system_fuel}")
                    system_df[system_fuel] = 0
            
            # Define CO2 emissions intensity for system fuels (tons/MMBtu)
            co2_intensity = {
                'NG_NY': 0.05306,
                'FO2_NY': 0.07315,
                'FO6_NY': 0.07415,
                'coal_NY': 0.10316,
#                 'Kerosene_NY': 0.07315,
                'None': 0
            }

            # Create a fixed date range for exactly 8760 hours (full year)
            start_date = datetime(int(year), 1, 1)
            end_date = datetime(int(year), 12, 31, 23)
            hourly_dates = pd.date_range(start=start_date, end=end_date, freq='H')
            
            # Create hourly dataframe
            hourly_df = pd.DataFrame({'TimeStamp': hourly_dates})
            
            # For each weekly entry in the system data, find matching hours
            for idx, row in system_df.iterrows():
                week_start = row['TimeStamp']
                if idx < len(system_df) - 1:
                    week_end = system_df.iloc[idx + 1]['TimeStamp'] - timedelta(hours=1)
                else:
                    # For the last week, only go up to the end of year
                    week_end = min(week_start + timedelta(days=6, hours=23), end_date)

                # Get mask for this week's hours
                mask = (hourly_df['TimeStamp'] >= week_start) & (hourly_df['TimeStamp'] <= week_end)

                # Assign fuel prices to these hours
                for fuel in system_fuels:
                    hourly_df.loc[mask, fuel] = row[fuel]
            
            # Fill any NaN values
            for fuel in system_fuels:
                hourly_df[fuel] = hourly_df[fuel].ffill().bfill()

            # Create the final Fuels_data.csv structure
            fuel_names = ['Time_Index'] + system_fuels
            co2_row = [''] + [co2_intensity.get(fuel, 0) for fuel in system_fuels]

            # Create output dataframe
            fuels_data = pd.DataFrame(columns=fuel_names)
            fuels_data.loc[0] = co2_row

            # Add hourly fuel prices with proper Time_Index
            hourly_df['Time_Index'] = range(1, len(hourly_df) + 1)

            # Reorder columns
            cols_to_use = ['Time_Index'] + system_fuels
            hourly_df = hourly_df[cols_to_use]

            # Append the hourly data
            fuels_data = pd.concat([fuels_data, hourly_df], ignore_index=True)

            # Save the result
            fuels_data.to_csv(output_file, index=False)
            print(f"System-level Fuel_data.csv created at {output_file}")
            
        elif from_11zone is not None:
            # Try to extract from 11-zone data if available
            eleven_zone_fuel_file = os.path.join(from_11zone.case_path, "system/Fuels_data.csv")
            
            if os.path.exists(eleven_zone_fuel_file):
                print(f"Reading fuel data from 11-zone file: {eleven_zone_fuel_file}")
                eleven_zone_fuels = pd.read_csv(eleven_zone_fuel_file)
                
                # Map zone-specific fuels to system-level fuels
                system_fuel_mapping = {
                    'NG_A2E': 'NG_NY',
                    'NG_F2I': 'NG_NY',
                    'NG_J': 'NG_NY',
                    'NG_K': 'NG_NY',
                    'FO2_UPNY': 'FO2_NY',
                    'FO2_DSNY': 'FO2_NY',
                    'FO6_UPNY': 'FO6_NY',
                    'FO6_DSNY': 'FO6_NY',
                    'coal_NY': 'coal_NY',
#                     'Kerosene_NY': 'Kerosene_NY',
                    'None': 'None'
                }
                
                # Identify unique system fuels we need
                system_fuels = list(set(system_fuel_mapping.values()))
                
                # Create a new dataframe for system fuels
                # First row contains CO2 intensity
                system_fuels_df = pd.DataFrame(columns=['Time_Index'] + system_fuels)
                
                # Extract the first row (CO2 intensity)
                first_row = ['']  # Time_Index is empty for first row
                
                for fuel in system_fuels:
                    if fuel == 'None':
                        first_row.append(0)  # Zero CO2 for None
                        continue
                        
                    # Find corresponding 11-zone fuels
                    corresponding_fuels = [z_fuel for z_fuel, sys_fuel in system_fuel_mapping.items() 
                                         if sys_fuel == fuel and z_fuel in eleven_zone_fuels.columns]
                    
                    if corresponding_fuels:
                        # Use the first fuel's CO2 intensity as they should be the same for the same fuel type
                        first_row.append(eleven_zone_fuels.iloc[0][corresponding_fuels[0]])
                    else:
                        print(f"Warning: No corresponding 11-zone fuel found for {fuel}")
                        first_row.append(0)
                
                # Add the CO2 intensity row
                system_fuels_df.loc[0] = first_row
                
                # Process hourly data (rows 1 onward)
                hourly_data = []
                
                for i in range(1, len(eleven_zone_fuels)):
                    row = eleven_zone_fuels.iloc[i]
                    time_index = row['Time_Index']
                    
                    new_row = {'Time_Index': time_index}
                    
                    for fuel in system_fuels:
                        if fuel == 'None':
                            new_row[fuel] = 0
                            continue
                            
                        # Find corresponding 11-zone fuels
                        corresponding_fuels = [z_fuel for z_fuel, sys_fuel in system_fuel_mapping.items() 
                                             if sys_fuel == fuel and z_fuel in eleven_zone_fuels.columns]
                        
                        if corresponding_fuels:
                            # Average the prices of corresponding fuels
                            fuel_prices = [row[z_fuel] for z_fuel in corresponding_fuels]
                            new_row[fuel] = sum(fuel_prices) / len(fuel_prices)
                        else:
                            new_row[fuel] = 0
                    
                    hourly_data.append(new_row)
                
                # Convert to DataFrame and combine with CO2 intensity row
                hourly_df = pd.DataFrame(hourly_data)
                system_fuels_df = pd.concat([system_fuels_df, hourly_df], ignore_index=True)
                
                # Save the result
                system_fuels_df.to_csv(output_file, index=False)
                print(f"System-level Fuel_data.csv created at {output_file}")
            else:
                print(f"Error: 11-zone fuel file not found at {eleven_zone_fuel_file}")
        else:
            print(f"Error: Could not find input file {input_file} and no 11-zone data provided")
    
    def map_fuel_to_genx(self, fuel):
        """Map a fuel type to the appropriate GenX fuel name at system level"""
        if isinstance(fuel, float):
            return 'None'
            
        fuel_lower = fuel.lower()
        
        if fuel_lower in ['natural_gas', 'naturalgas', 'natural gas', 'ng']:
            return 'NG_NY'
        elif fuel_lower in ['fuel_oil_2', 'fueloil2', 'fuel oil 2', 'fo2']:
            return 'FO2_NY'
        elif fuel_lower in ['fuel_oil_6', 'fueloil6', 'fuel oil 6', 'fo6']:
            return 'FO6_NY'
        elif fuel_lower in ['coal']:
            return 'coal_NY'
#         elif fuel_lower in ['kerosene']:
#             return 'Kerosene_NY'
            
        return 'None'

        def aggregate_generators_variability(self, from_11zone=None):
            """
            Create system-level Generators_variability.csv by directly aggregating 
            capacity-weighted averages from input data files.
            """
            if from_11zone is None:
                print("Error: Need 11-zone model to extract data paths")
                return

            # Set up data sources
            data_path = from_11zone.data_path
            data_files = {
                "merged": os.path.join(data_path, "merged_2019.csv"),
                "thermal": os.path.join(data_path, "thermalHourlyCF_2019.csv"),
                "nuclear": os.path.join(data_path, "nuclearHourlyCF_2019.csv"),
                "hydro": os.path.join(data_path, "hydroHourlyCF_2019.csv")
            }

            # Create empty dataframe with Time_Index
            num_hours = 8760
            system_variability = pd.DataFrame({'Time_Index': range(1, num_hours + 1)})

            # Get all resources from resource files
            resource_names = self.extract_resource_names()
            all_resources = []
            for resource_list in resource_names.values():
                all_resources.extend(resource_list)

            # Calculate system-level aggregated profiles

            # 1. Thermal resources (aggregated by technology type)
            if os.path.exists(data_files['thermal']):
                print("Aggregating thermal capacity factors...")
                thermal_df = pd.read_csv(data_files['thermal'])

                # Create aggregated profiles for each thermal technology
                for tech in ['CombinedCycle', 'CombustionTurbine', 'SteamTurbine']:
                    # For each fuel type
                    for fuel in ['NaturalGas', 'FuelOil2', 'FuelOil6']:
                        # Get columns matching this tech/fuel combination
                        tech_cols = [col for col in thermal_df.columns if col != 'Time_Index' and tech in col]

                        if tech_cols:
                            # Simple average across all zones
                            avg_profile = thermal_df[tech_cols].mean(axis=1).values

                            # Apply to both existing and new resources
                            system_variability[f"{tech}_{fuel}_NY_Existing"] = avg_profile
                            system_variability[f"{tech}_{fuel}_NY_New"] = avg_profile

            # 2. Nuclear resources
            if os.path.exists(data_files['nuclear']):
                print("Aggregating nuclear capacity factors...")
                nuclear_df = pd.read_csv(data_files['nuclear'])

                # Get all nuclear data columns
                nuclear_cols = [col for col in nuclear_df.columns if col != 'Time_Index']

                if nuclear_cols:
                    # Simple average across all zones
                    avg_profile = nuclear_df[nuclear_cols].mean(axis=1).values

                    # Apply to both existing and new resources
                    system_variability["Nuclear_NY_Existing"] = avg_profile
                    system_variability["Nuclear_NY_New"] = avg_profile

            # 3. Hydro resources
            if os.path.exists(data_files['hydro']):
                print("Aggregating hydro capacity factors...")
                hydro_df = pd.read_csv(data_files['hydro'])

                # Get all hydro data columns
                hydro_cols = [col for col in hydro_df.columns if col != 'Time_Index']

                if hydro_cols:
                    # Simple average across all zones
                    avg_profile = hydro_df[hydro_cols].mean(axis=1).values

                    # Apply to Hydro and PumpedHydro
                    system_variability["Hydro_NY"] = avg_profile
                    system_variability["PumpedHydro_NY"] = avg_profile

            # 4. VRE resources (wind and solar from merged file)
            if os.path.exists(data_files['merged']):
                print("Aggregating wind and solar capacity factors...")
                merged_df = pd.read_csv(data_files['merged'])

                # Process each zone's data and aggregate
                wind_data = []
                solar_data = []

                # Get unique zones
                if 'zone' in merged_df.columns:
                    for zone in merged_df['zone'].unique():
                        zone_data = merged_df[merged_df['zone'] == zone]

                        # Ensure 8760 hours of data
                        if len(zone_data) != 8760:
                            continue

                        # Add to aggregation arrays
                        if 'Wind' in zone_data.columns:
                            wind_data.append(zone_data['Wind'].values)
                        if 'Solar' in zone_data.columns:
                            solar_data.append(zone_data['Solar'].values)

                # Calculate averages
                if wind_data:
                    avg_wind = np.mean(wind_data, axis=0)
                    system_variability["WindLand_NY_Existing"] = avg_wind
                    system_variability["WindLand_NY_New"] = avg_wind
                    system_variability["WindOffshore_NY_New"] = avg_wind

                if solar_data:
                    avg_solar = np.mean(solar_data, axis=0)
                    system_variability["SolarUtility_NY_Existing"] = avg_solar
                    system_variability["SolarUtility_NY_New"] = avg_solar
                    system_variability["SolarBTM_NY_New"] = avg_solar

            # 5. Battery (always 1.0)
            for resource in all_resources:
                if 'Battery' in resource:
                    system_variability[resource] = 1.0

            # Check for any missing resources and set to 1.0
            for resource in all_resources:
                if resource not in system_variability.columns:
                    system_variability[resource] = 1.0

            # Ensure all values are between 0 and 1
            for col in system_variability.columns:
                if col != 'Time_Index':
                    system_variability[col] = system_variability[col].clip(0, 1)

            # Save output
            output_file = os.path.join(self.case_path, "system/Generators_variability.csv")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            system_variability.to_csv(output_file, index=False)

            print(f"Created system-level variability profiles for {len(system_variability.columns)-1} resources")
            return system_variability


    def generate_split_resources(self, atb_summary_file=None):
        """
        Generate resource files with existing and new build resources separated.
        Creates new build options for everything except hydro by default.
        Only creates existing resources if they have capacity > 0.

        Args:
            atb_summary_file (str): Path to ATB summary CSV file

        Returns:
            tuple: Containing DataFrames for each resource type
        """
        # Load ATB data if provided
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
                elif (tech == 'Hydro') | (tech == 'PumpedHydro'):
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
        
        # Lists to hold resource data
        thermal_data = []
        vre_data = []
        hydro_data = []
        storage_data = []
        
        # O&M cost reduction factor for new builds
        new_build_om_factor = 0.5
        
        # Process thermal resources
        for resource_name, resource_info in self.thermal_resources.items():
            tech = resource_info['tech']
            fuel = resource_info['fuel']
            capacity = resource_info['capacity']
            
            # Get fuel name and tech parameters
            fuel_name = self.map_fuel_to_genx(fuel)
            tech_params = self.technology_params.get(tech, {})
            
            # Add existing resource if capacity > 0
            if capacity > 0:
                existing_resource = {
                    'Resource': f"{resource_name}_Existing",
                    'Zone': 1,
                    'region': 'NY',
                    'cluster': 1,
                    'Technology': tech,
                    'New_Build': 0,
                    'Can_Retire': 1,
                    'Existing_Cap_MW': capacity,
                    'Max_Cap_MW': capacity,
                    'Min_Cap_MW': 0,
                    'Min_Power': tech_params.get('Min_Power', 0.3),
                    'Ramp_Up_Percentage': tech_params.get('Ramp_Up_Percentage', 0.5),
                    'Ramp_Dn_Percentage': tech_params.get('Ramp_Dn_Percentage', 0.5),
                    'Shutdown_Cost_per_MW': tech_params.get('Shutdown_Cost_per_MW', 25),
                    'Startup_Cost_per_MW': tech_params.get('Startup_Cost_per_MW', 50),
                    'Up_Time': tech_params.get('Up_Time', 4),
                    'Down_Time': tech_params.get('Down_Time', 4),
                    'Inv_Cost_per_MWyr': 0,
                    'Fixed_OM_Cost_per_MWyr': tech_params.get('Fix_OM_MW', 30000) * self.system_scale_factor,
                    'Var_OM_Cost_per_MWh': tech_params.get('Var_OM', 5.0),
                    'Heat_Rate_MMBTU_per_MWh': tech_params.get('Heat_Rate_MMBTU_per_MWh', 10.0),
                    'Fuel': fuel_name,
                    'Model': 1
                }
                thermal_data.append(existing_resource)
            
            # Always add new build option
            new_resource = {
                'Resource': f"{resource_name}_New",
                'Zone': 1,
                'region': 'NY',
                'cluster': 1,
                'Technology': tech,
                'New_Build': 1,
                'Can_Retire': 1,
                'Existing_Cap_MW': 0,
                'Max_Cap_MW': -1,
                'Min_Cap_MW': 0,
                'Min_Power': tech_params.get('Min_Power', 0.3),
                'Ramp_Up_Percentage': tech_params.get('Ramp_Up_Percentage', 0.5),
                'Ramp_Dn_Percentage': tech_params.get('Ramp_Dn_Percentage', 0.5),
                'Shutdown_Cost_per_MW': tech_params.get('Shutdown_Cost_per_MW', 25),
                'Startup_Cost_per_MW': tech_params.get('Startup_Cost_per_MW', 50),
                'Up_Time': tech_params.get('Up_Time', 4),
                'Down_Time': tech_params.get('Down_Time', 4),
                'Inv_Cost_per_MWyr': tech_params.get('Ann_Inv_MWYr', 100000) * self.system_scale_factor,
                'Fixed_OM_Cost_per_MWyr': tech_params.get('Fix_OM_MW', 30000) * self.system_scale_factor * new_build_om_factor,
                'Var_OM_Cost_per_MWh': tech_params.get('Var_OM', 5.0),
                'Heat_Rate_MMBTU_per_MWh': tech_params.get('Heat_Rate_MMBTU_per_MWh', 10.0),
                'Fuel': fuel_name,
                'Model': 1
            }
            thermal_data.append(new_resource)
        
        # Process nuclear if available
        if 'Nuclear' in self.technology_capacities:
            capacity = self.technology_capacities['Nuclear']
            tech_params = self.technology_params.get('Nuclear', {})
            
            if capacity > 0:
                # Add existing nuclear
                nuclear_resource = {
                    'Resource': 'Nuclear_NY_Existing',
                    'Zone': 1,
                    'region': 'NY',
                    'Model': 1,
                    'cluster': 1,
                    'Technology': 'Nuclear',
                    'New_Build': 0,
                    'Can_Retire': 1,
                    'Existing_Cap_MW': capacity,
                    'Max_Cap_MW': capacity,
                    'Min_Cap_MW': 0,
                    'Min_Power': tech_params.get('Min_Power', 0.4),
                    'Ramp_Up_Percentage': tech_params.get('Ramp_Up_Percentage', 0.2),
                    'Ramp_Dn_Percentage': tech_params.get('Ramp_Dn_Percentage', 0.2),
                    'Shutdown_Cost_per_MW': tech_params.get('Shutdown_Cost_per_MW', 100),
                    'Startup_Cost_per_MW': tech_params.get('Startup_Cost_per_MW', 200),
                    'Up_Time': tech_params.get('Up_Time', 48),
                    'Down_Time': tech_params.get('Down_Time', 48),
                    'Inv_Cost_per_MWyr': 0,
                    'Fixed_OM_Cost_per_MWyr': tech_params.get('Fix_OM_MW', 120000) * self.system_scale_factor,
                    'Var_OM_Cost_per_MWh': tech_params.get('Var_OM', 2.3),
                    'Heat_Rate_MMBTU_per_MWh': tech_params.get('Heat_Rate_MMBTU_per_MWh', 10.46),
                    'Fuel': 'None'
                }
                thermal_data.append(nuclear_resource)
            
            # Always add new nuclear option
            new_nuclear = {
                'Resource': 'Nuclear_NY_New',
                'Zone': 1,
                'region': 'NY',
                'Model': 1,
                'cluster': 1,
                'Technology': 'Nuclear',
                'New_Build': 1,
                'Can_Retire': 1,
                'Existing_Cap_MW': 0,
                'Max_Cap_MW': -1,
                'Min_Cap_MW': 0,
                'Min_Power': tech_params.get('Min_Power', 0.4),
                'Ramp_Up_Percentage': tech_params.get('Ramp_Up_Percentage', 0.2),
                'Ramp_Dn_Percentage': tech_params.get('Ramp_Dn_Percentage', 0.2),
                'Shutdown_Cost_per_MW': tech_params.get('Shutdown_Cost_per_MW', 100),
                'Startup_Cost_per_MW': tech_params.get('Startup_Cost_per_MW', 200),
                'Up_Time': tech_params.get('Up_Time', 48),
                'Down_Time': tech_params.get('Down_Time', 48),
                'Inv_Cost_per_MWyr': tech_params.get('Ann_Inv_MWYr', 600000) * self.system_scale_factor,
                'Fixed_OM_Cost_per_MWyr': tech_params.get('Fix_OM_MW', 120000) * self.system_scale_factor * new_build_om_factor,
                'Var_OM_Cost_per_MWh': tech_params.get('Var_OM', 2.3),
                'Heat_Rate_MMBTU_per_MWh': tech_params.get('Heat_Rate_MMBTU_per_MWh', 10.46),
                'Fuel': 'None'
            }
            thermal_data.append(new_nuclear)
        
        # Process VRE technologies
        vre_techs = {'WindLand', 'WindOffshore', 'SolarUtility', 'SolarBTM'}
        
        for tech in vre_techs:
            if tech in self.technology_capacities:
                capacity = self.technology_capacities[tech]
            else: 
                capacity = 0 
            tech_params = self.technology_params.get(tech, {})
                
            # Use defaults if parameters not found
            ann_inv = tech_params.get('Ann_Inv_MWYr', 100000) * self.system_scale_factor
            fix_om = tech_params.get('Fix_OM_MW', 30000) * self.system_scale_factor
            var_om = tech_params.get('Var_OM', 0)
                
            # Add existing if capacity > 0
            if capacity > 0:
                vre_data.append({
                        'Resource': f'{tech}_NY_Existing',
                        'Zone': 1,
                        'region': 'NY',
                        'cluster': 1,
                        'Technology': tech,
                        'New_Build': 0,
                        'Can_Retire': 1,
                        'Existing_Cap_MW': capacity,
                        'Max_Cap_MW': capacity,
                        'Min_Cap_MW': 0,
                        'Inv_Cost_per_MWyr': 0,
                        'Fixed_OM_Cost_per_MWyr': fix_om,
                        'Var_OM_Cost_per_MWh': var_om,
                        'Heat_Rate_MMBTU_per_MWh': 0,
                        'Fuel': 'None',
                        'Num_VRE_bins': 1
                })
                
                # Always add new build option
            vre_data.append({
                    'Resource': f'{tech}_NY_New',
                    'Zone': 1,
                    'region': 'NY',
                    'cluster': 1,
                    'Technology': tech,
                    'New_Build': 1,
                    'Can_Retire': 1,
                    'Existing_Cap_MW': 0,
                    'Max_Cap_MW': -1,
                    'Min_Cap_MW': 0,
                    'Inv_Cost_per_MWyr': ann_inv,
                    'Fixed_OM_Cost_per_MWyr': fix_om * new_build_om_factor,
                    'Var_OM_Cost_per_MWh': var_om,
                    'Heat_Rate_MMBTU_per_MWh': 0,
                    'Fuel': 'None',
                    'Num_VRE_bins': 1
            })
        
        # Process Hydro (only existing, no new build)
        if 'Hydro' in self.technology_capacities:
            capacity = self.technology_capacities['Hydro']
            if capacity > 0:
                tech_params = self.technology_params.get('Hydro', {})
                
                hydro_data.append({
                    'Resource': 'Hydro_NY',
                    'Zone': 1,
                    'region': 'NY',
                    'cluster': 1,
                    'Technology': 'Hydro',
                    'New_Build': 0,
                    'Can_Retire': 0,
                    'Existing_Cap_MW': capacity,
                    'Max_Cap_MW': capacity,
                    'Min_Cap_MW': capacity,
                    'Inv_Cost_per_MWyr': 0,
                    'Fixed_OM_Cost_per_MWyr': tech_params.get('Fix_OM_MW', 45000) * self.system_scale_factor,
                    'Var_OM_Cost_per_MWh': tech_params.get('Var_OM', 2.5),
                    'Heat_Rate_MMBTU_per_MWh': 0,
                    'Fuel': 'None',
                    'Min_Power': tech_params.get('Min_Power', 0.1),
                    'Ramp_Up_Percentage': tech_params.get('Ramp_Up_Percentage', 0.5),
                    'Ramp_Dn_Percentage': tech_params.get('Ramp_Dn_Percentage', 0.5),
                    'Hydro_Energy_to_Power_Ratio': tech_params.get('Hydro_Energy_to_Power_Ratio', 12),
                    'LDS': 1
                })
        
        # Process Pumped Hydro (only existing, no new build)
        if 'PumpedHydro' in self.technology_capacities:
            capacity = self.technology_capacities['PumpedHydro']
            if capacity > 0:
                tech_params = self.technology_params.get('PumpedHydro', {})
                
                # Use defaults if parameters not found
                min_power = tech_params.get('Min_Power', 0.1)
                ramp_up = tech_params.get('Ramp_Up_Percentage', 0.5)
                ramp_down = tech_params.get('Ramp_Dn_Percentage', 0.5)
                hydro_e2p = tech_params.get('Hydro_Energy_to_Power_Ratio', 12)
                
                hydro_data.append({
                    'Resource': 'PumpedHydro_NY',
                    'Zone': 1,
                    'region': 'NY',
                    'cluster': 1,
                    'Technology': 'Hydro',
                    'New_Build': 0,
                    'Can_Retire': 0,
                    'Existing_Cap_MW': capacity,
                    'Max_Cap_MW': capacity,
                    'Min_Cap_MW': capacity,
                    'Inv_Cost_per_MWyr': 0,
                    'Fixed_OM_Cost_per_MWyr': tech_params.get('Fix_OM_MW', 50000) * self.system_scale_factor,
                    'Var_OM_Cost_per_MWh': tech_params.get('Var_OM', 2.0),
                    'Heat_Rate_MMBTU_per_MWh': 0,
                    'Fuel': 'None',
                    'Min_Power': min_power,
                    'Ramp_Up_Percentage': ramp_up,
                    'Ramp_Dn_Percentage': ramp_down,
                    'Hydro_Energy_to_Power_Ratio': hydro_e2p,
                    'LDS': 1
                })
        
        # Process Battery storage
        if 'Battery' in self.technology_capacities:
            capacity = self.technology_capacities['Battery']
            tech_params = self.technology_params.get('Battery', {})
            
            # Calculate power and energy costs
            power_cost_fraction = 0.6
            energy_cost_fraction = 0.4 / 4  # per MWh for 4-hour system
            
            total_ann_inv = tech_params.get('Ann_Inv_MWYr', 70000)
            power_cost = total_ann_inv * power_cost_fraction
            energy_cost = total_ann_inv * energy_cost_fraction
            
            total_fixed_om = tech_params.get('Fix_OM_MW', 20000)
            power_fixed_om = total_fixed_om * power_cost_fraction
            energy_fixed_om = total_fixed_om * energy_cost_fraction
            
            # Add existing if capacity > 0
            if capacity > 0:
                storage_data.append({
                    'Resource': 'Battery_NY_Existing',
                    'Zone': 1,
                    'region': 'NY',
                    'cluster': 1,
                    'Technology': 'Battery',
                    'New_Build': 0,
                    'Can_Retire': 1,
                    'Model': 1,
                    'LDS': 0,
                    'Existing_Cap_MW': capacity,
                    'Existing_Cap_MWh': capacity * 4,
                    'Max_Cap_MW': capacity,
                    'Max_Cap_MWh': capacity * 4,
                    'Min_Cap_MW': 0,
                    'Min_Cap_MWh': 0,
                    'Inv_Cost_per_MWyr': 0,
                    'Inv_Cost_per_MWhyr': 0,
                    'Fixed_OM_Cost_per_MWyr': power_fixed_om * self.system_scale_factor,
                    'Fixed_OM_Cost_per_MWhyr': energy_fixed_om * self.system_scale_factor,
                    'Var_OM_Cost_per_MWh': 0,
                    'Var_OM_Cost_per_MWhIn': 0,
                    'Heat_Rate_MMBTU_per_MWh': 0,
                    'Fuel': 'None',
                    'Self_Disch': tech_params.get('Self_Disch', 0.0004),
                    'Eff_Up': tech_params.get('Eff_Up', 0.92),
                    'Eff_Down': tech_params.get('Eff_Down', 0.92),
                    'Min_Duration': tech_params.get('Min_Duration', 2),
                    'Max_Duration': tech_params.get('Max_Duration', 8)
                })
            
            # Always add new battery option
            storage_data.append({
                'Resource': 'Battery_NY_New',
                'Zone': 1,
                'region': 'NY',
                'cluster': 1,
                'Technology': 'Battery',
                'New_Build': 1,
                'Can_Retire': 1,
                'Model': 1,
                'LDS': 0,
                'Existing_Cap_MW': 0,
                'Existing_Cap_MWh': 0,
                'Max_Cap_MW': -1,
                'Max_Cap_MWh': -1,
                'Min_Cap_MW': 0,
                'Min_Cap_MWh': 0,
                'Inv_Cost_per_MWyr': power_cost * self.system_scale_factor,
                'Inv_Cost_per_MWhyr': energy_cost * self.system_scale_factor,
                'Fixed_OM_Cost_per_MWyr': power_fixed_om * self.system_scale_factor * new_build_om_factor,
                'Fixed_OM_Cost_per_MWhyr': energy_fixed_om * self.system_scale_factor * new_build_om_factor,
                'Var_OM_Cost_per_MWh': 0,
                'Var_OM_Cost_per_MWhIn': 0,
                'Heat_Rate_MMBTU_per_MWh': 0,
                'Fuel': 'None',
                'Self_Disch': tech_params.get('Self_Disch', 0.0004),
                'Eff_Up': tech_params.get('Eff_Up', 0.92),
                'Eff_Down': tech_params.get('Eff_Down', 0.92),
                'Min_Duration': tech_params.get('Min_Duration', 2),
                'Max_Duration': tech_params.get('Max_Duration', 8)
            })
        
        # Create resource CSVs
        resources_folder = os.path.join(self.case_path, 'resources')
        os.makedirs(resources_folder, exist_ok=True)
        
        # Convert to DataFrames
        thermal_df = pd.DataFrame(thermal_data)
        vre_df = pd.DataFrame(vre_data)
        hydro_df = pd.DataFrame(hydro_data)
        storage_df = pd.DataFrame(storage_data)
        
        # Save to CSVs
        thermal_df.to_csv(os.path.join(resources_folder, "Thermal.csv"), index=False)
        vre_df.to_csv(os.path.join(resources_folder, "Vre.csv"), index=False)
        hydro_df.to_csv(os.path.join(resources_folder, "Hydro.csv"), index=False)
        storage_df.to_csv(os.path.join(resources_folder, "Storage.csv"), index=False)
        
        print(f"Created system-level resource files:")
        print(f"  Thermal: {len(thermal_data)} resources")
        print(f"  VRE: {len(vre_data)} resources")
        print(f"  Hydro: {len(hydro_data)} resources")
        print(f"  Storage: {len(storage_data)} resources")
        
        return thermal_df, vre_df, hydro_df, storage_df 
    
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
            'ModelingtoGenerateAlternativeSlack': 0.1,
            'ModelingToGenerateAlternativeIterations': 3,
            'MethodofMorris': 0,
            'OutputFullTimeSeries': 1
        }
        
        data_gurobi = {
            'Feasib_Tol': 1.0e-05,          
            'Optimal_Tol': 1e-5,            
            'TimeLimit': 200000,            
            'Pre_Solve': 1,                
            'Method': 1,                    
            'MIPGap': 1e-3,                  
            'BarConvTol': 1.0e-08,         
            'NumericFocus': 0,  
            'PreDual': 0,
            'Crossover': -1,                
            'AggFill': -1,                  
        }
        settings_path = os.path.join(self.case_path, 'settings')
        os.makedirs(settings_path, exist_ok=True)
        
        with open(os.path.join(settings_path, 'genx_settings.yml'), 'w') as file:
            yaml.dump(data_genx, file, default_flow_style=False)
        
        with open(os.path.join(settings_path, 'gurobi_settings.yml'), 'w') as file:
            yaml.dump(data_gurobi, file, default_flow_style=False)
        
        print("Successfully set GenX and Gurobi settings")

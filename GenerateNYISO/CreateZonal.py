#!/usr/bin/env python3
"""
NYISO GenX Case Creator

Main script to create a complete GenX case at the zonal level
Handles data loading, directory creation, and file generation in the proper sequence.
"""

import os
import sys
import pandas as pd
import argparse
from datetime import datetime
from ZonalCase import * 

def main():
    # Configuration parameters
    case_name = "NYISO_Case"
    data_dir = "../NYISO_Data"  # Directory containing NYISO data files
    output_dir = "../SPCM/research_systems"        # Use None for current directory or specify a path
    year = "2019"            # Year for load and fuel data
    atb_file = "../NYISO_Data/ATB_summary.csv"  # Path to ATB cost summary file (set to None if not available)
    gb_file = os.path.join(data_dir, "NYCA2024_Summary.csv")
    print(f"=== Creating NYISO GenX Case: {case_name} ===")

    # Initialize the generator
    generator = NYISOGenXGenerator_Zonal(case_name=case_name, base_dir=output_dir)

    # Step 1: Set up the directory structure
    print("\n--- Step 1: Creating directory structure ---")
    generator.create_case_structure()

    # Step 2: Load technology capacities
    print("\n--- Step 2: Loading technology capacities ---")

    # Alternative: Load from CSV if available
    if os.path.exists(gb_file):
        capacity_data = pd.read_csv(gb_file)
        print(capacity_data)
    generator.load_technology_capacities(capacity_data)

    # Step 3: Create network topology file
    print("\n--- Step 3: Creating network topology ---")
    generator.create_nyiso_network_csv()

    # Step 4: Create demand data file
    print("\n--- Step 4: Processing load data ---")
    generator.create_demand_data_csv(year=year)

    # Step 5: Create fuel data file
    print("\n--- Step 5: Processing fuel price data ---")
    generator.create_fuels_data_csv(year=year)

    # Step 6: Generate resource files with existing and new-build options
    print("\n--- Step 7: Creating resource files ---")
    generator.generate_split_resources(atb_summary_file=atb_file)

    # Step 7: Create generators variability file
    print("\n--- Step 6: Creating generator variability profiles ---")
    generator.create_generators_variability()

    # Step 8: Generate settings files
    print("\n--- Step 8: Creating GenX settings files ---")
    generator.generate_settings_yaml()

    # Step 9: Create the Run.jl file
    print("\n--- Step 9: Creating Run.jl file ---")
    generator.write_run_file()

    print(f"\n=== GenX case {case_name} created successfully! ===")
    print(f"Case location: {generator.case_path}")

    # Display information about the created resources
    print("\nResource Summary:")
    resource_names = generator.extract_resource_names()
    for resource_type, resources in resource_names.items():
        print(f"  {resource_type}: {len(resources)} resources")
        
    # Optional: Show the first few resources of each type
    for resource_type, resources in resource_names.items():
        if resources:
            print(f"\nExample {resource_type} resources:")
            for resource in resources[:3]:  # Show first 3
                print(f"  - {resource}")
            if len(resources) > 3:
                print(f"  ... and {len(resources) - 3} more")
if __name__ == "__main__":
    main()
import os
import pandas as pd
from ZonalCase import *  # Original 11-zone generator
from SystemCase import *
def main():
    # Configuration parameters
    system_case_name = "NYISO_System"
    zonal_case_name = "NYISO_Case"  # Original 11-zone case for reference
    data_dir = "../NYISO_Data"  # Directory containing NYISO data files
    output_dir = "../SPCM/research_systems"  # Use None for current directory or specify
    year = "2019"  # Year for load and fuel data
    atb_file = "../NYISO_Data/ATB_summary.csv"  # ATB cost summary file (None if unavailable)

    print(f"=== Creating NYISO System-Level GenX Case: {system_case_name} ===")

    # First, load or create the 11-zone model to extract data from
    if os.path.exists(os.path.join(output_dir, zonal_case_name)):
        print("\n--- Loading existing 11-zone model for reference ---")
        zonal_generator = NYISOGenXGenerator_Zonal(case_name=zonal_case_name, base_dir=output_dir)
        
        # Load capacities if we have the file
        if os.path.exists(os.path.join(data_dir, "NYCA2024_Summary.csv")):
            capacity_data = pd.read_csv(os.path.join(data_dir, "NYCA2024_Summary.csv"))
            zonal_generator.load_technology_capacities(capacity_data)
    else:
        print("\n--- Creating new 11-zone model for reference ---")
        zonal_generator = NYISOGenXGenerator_Zonal(case_name=zonal_case_name, base_dir=output_dir)
        
        # Load capacities
        if os.path.exists(os.path.join(data_dir, "NYCA2024_Summary.csv")):
            capacity_data = pd.read_csv(os.path.join(data_dir, "NYCA2024_Summary.csv"))
            zonal_generator.load_technology_capacities(capacity_data)
        
        # Create the case structure (minimal, just need the data)
        zonal_generator.create_case_structure()
        zonal_generator.create_nyiso_network_csv()
        zonal_generator.create_demand_data_csv(year=year)
        zonal_generator.create_fuels_data_csv(year=year)
        zonal_generator.generate_split_resources(atb_summary_file=atb_file)
        zonal_generator.create_generators_variability()

    # Now initialize the system-level generator
    print("\n--- Initializing system-level generator ---")
    system_generator = NYISOSystemGenerator(case_name=system_case_name, base_dir=output_dir)

    # Step 1: Set up the directory structure
    print("\n--- Step 1: Creating directory structure ---")
    system_generator.create_case_structure()

    # Step 2: Load technology capacities by aggregating from zonal model
    print("\n--- Step 2: Aggregating technology capacities ---")
    system_generator.load_technology_capacities(from_11zone=zonal_generator)

    # Step 3: Create simplified network topology file (single zone)
    print("\n--- Step 3: Creating system-level network topology ---")
    system_generator.create_network_csv()

    # Step 4: Create demand data file by aggregating zonal demand
    print("\n--- Step 4: Aggregating load data ---")
    system_generator.create_demand_data_csv(year=year, from_11zone=zonal_generator)

    # Step 5: Create fuel data file
    print("\n--- Step 5: Creating system-level fuel price data ---")
    system_generator.create_fuels_data_csv(year=year, from_11zone=zonal_generator)

    # Step 6: Generate resource files
    print("\n--- Step 6: Creating system-level resource files ---")
    system_generator.generate_split_resources(atb_summary_file=atb_file)

    # Step 7: Create generators variability file
    print("\n--- Step 7: Creating system-level generator variability profiles ---")
    system_generator.aggregate_generators_variability(from_11zone=zonal_generator)

    # Step 8: Generate settings files
    print("\n--- Step 8: Creating GenX settings files ---")
    system_generator.generate_settings_yaml()

    # Step 9: Create the Run.jl file
    print("\n--- Step 9: Creating Run.jl file ---")
    # Implement or call equivalent of write_run_file for system level
    system_generator.write_run_file()

    print(f"\n=== GenX case {system_case_name} created successfully! ===")
    print(f"Case location: {system_generator.case_path}")

    # Display information about the created resources
    print("\nResource Summary:")
    # Implement or adapt extract_resource_names for system level
    resource_names = system_generator.extract_resource_names()
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
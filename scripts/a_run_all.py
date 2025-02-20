import subprocess
import os

# Change the current working directory to the scripts folder
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__))))

# List of Python files to run
python_files = [
                "generate_case_folders.py", 
                "initialize_generator_data.py", 
                "read_ATB_data.py",
                "read_FERC_data.py",
                "upd_utility_storage_assumptions.py",
                "upd_RICE_assumptions.py",
                "upd_SMR_assumptions.py",
                "set_model_params.py",
                "generate_load_input.py",
                "generate_thermal_input.py",
                "generate_storage_input.py",
                "generate_vre_input.py",
                "generate_policy_assignments.py",
                "generate_co2_cap.py",
                "generate_fuels_input.py",
                "generate_generators_variability.py",
                "generate_network_input.py",
                "generate_op_res_input.py",
                "generate_genx_settings.py",
                "generate_gurobi_settings.py",
                "generate_run_file.py",
                ]

# Loop through each file and run it
for file in python_files:
        print(f"Ran {file}...")
        result = subprocess.run(["python", file])
        if result.returncode != 0:
            print(f"Error running {file}. Stopping execution.")
            break

print("All scripts have been executed.")
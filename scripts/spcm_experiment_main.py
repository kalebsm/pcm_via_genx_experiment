import subprocess
import sys
import os

# Add the root directory to the Python path (from within the scripts folder)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import utils or any module that requires data folder access
from utils.spcm_experiment_utils import get_paths

# from model_setup import model_setup

# # Run the model_setup module
# model_setup.main()

# Change the current working directory to the root folder
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(root_path, "scripts"))

print("Current working directory:", os.getcwd())

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
    file_path = os.path.join("model_setup", file)
    print(f"Running {file_path}...")
    result = subprocess.run(["python", file_path])
    if result.returncode != 0:
        print(f"Error running {file_path}. Stopping execution.")
        break
    else:
        print("All scripts have been executed successfully.")
import sys
import os

# Add the root directory (my_package) to sys.path so Python can find 'utils'
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # One level up from 'scripts'
sys.path.append(root_path)

# Now import from utils
from utils.spcm_experiment_utils import get_paths

# Example usage
scripts_path = get_paths('scripts')
data_path = get_paths('data')
genx_research_path = get_paths('genx_research')
spcm_research_path = get_paths('spcm_research')
figures_path = get_paths('figures')


def run_all_scripts():
    # List of python files to run
    python_files = [
                "generate_case_folders.py", 
                # "initialize_generator_data.py", 
                # "read_ATB_data.py",
                # "read_FERC_data.py",
                # "upd_utility_storage_assumptions.py",
                # "upd_RICE_assumptions.py",
                # "upd_SMR_assumptions.py",
                # "set_model_params.py",
                # "generate_load_input.py",
                # "generate_thermal_input.py",
                # "generate_storage_input.py",
                # "generate_vre_input.py",
                # "generate_policy_assignments.py",
                # "generate_co2_cap.py",
                # "generate_fuels_input.py",
                # "generate_generators_variability.py",
                # "generate_network_input.py",
                # "generate_op_res_input.py",
                # "generate_genx_settings.py",
                # "generate_gurobi_settings.py",
                # "generate_run_file.py",
                ]

    for script in python_files:
        script_path = os.path.join(os.path.dirname(__file__), script)
        if os.path.isfile(script_path):
            print(f"Running {script}...")
            exec(open(script_path).read())
        else:
            print(f"Script {script} not found.")

if __name__ == "__main__":
    run_all_scripts()
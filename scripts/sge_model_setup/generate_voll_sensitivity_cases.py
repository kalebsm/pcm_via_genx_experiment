import os
import sys
import shutil
import pandas as pd

# filepath: c:\Users\ks885\Documents\aa_research\Modeling\spcm_check\pcm_via_genx_experiment\scripts\sge_model_setup\generate_voll_sensitivity_cases.py

# add project root so utils can be imported (one level up from scripts/)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from utils.sge_utils import get_paths


def find_case_dir(research_path, base_case_name):
    # exact match first
    candidate = os.path.join(research_path, base_case_name)
    if os.path.isdir(candidate):
        return candidate
    # fallback: case folder that contains the base name (case-insensitive)
    for name in os.listdir(research_path):
        if base_case_name.lower() in name.lower():
            path = os.path.join(research_path, name)
            if os.path.isdir(path):
                return path
    return None

def update_demand_voll(case_dir, voll_value):
    demand_csv = os.path.join(case_dir, 'system', 'Demand_data.csv')
    if not os.path.isfile(demand_csv):
        return False
    df = pd.read_csv(demand_csv)
    if 'Voll' not in df.columns or df.shape[0] == 0:
        return False
    # set first row Voll (matches generate_load_input pattern)
    df.at[0, 'Voll'] = voll_value
    df.to_csv(demand_csv, index=False)
    return True

def generate_voll_sensitivity_cases(base_case_name="4_Hr_BESS", voll_values=None):
    if voll_values is None:
        voll_values = [2500, 10000, 15000, 20000]

    data_path = get_paths('data')
    genx_research_path = get_paths('genx_research')
    spcm_research_path = get_paths('spcm_research')

    created = []
    for research_path in (genx_research_path, spcm_research_path):
        src = find_case_dir(research_path, base_case_name)
        if src is None:
            print(f"Source case not found in {research_path}: {base_case_name}")
            continue

        for v in voll_values:
            dest_name = f"{os.path.basename(src)}_Voll{v}"
            dest = os.path.join(research_path, dest_name)
            if os.path.exists(dest):
                print(f"Destination exists, skipping: {dest}")
                continue
            shutil.copytree(src, dest)
            ok = update_demand_voll(dest, v)
            if not ok:
                print(f"Warning: couldn't update Demand_data.csv for {dest}")
            created.append(dest)
            print(f"Created {dest}")

    return created

if __name__ == "__main__":
    generate_voll_sensitivity_cases()
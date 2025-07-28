import os
import subprocess
from glob import glob

def run_notebooks_in_folder(folder):
    # Find all .ipynb files containing 'fig' or 'table' in the filename
    notebook_files = glob(os.path.join(folder, '*fig*.ipynb')) + glob(os.path.join(folder, '*table*.ipynb'))
    for nb in notebook_files:
        print(f"Running {nb}...")
        subprocess.run([
            "jupyter", "nbconvert", "--to", "notebook", "--execute",
            "--inplace", nb
        ], check=True)

if __name__ == "__main__":
    folder = os.path.dirname(os.path.abspath(__file__))
    run_notebooks_in_folder(folder)
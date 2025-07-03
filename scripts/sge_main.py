import subprocess
import sys
import os

# Add the root directory to the Python path (from within the scripts folder)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import utils or any module that requires data folder access
from utils.sge_utils import get_paths

# Run the sge_model_setup.py script in the sge_model_setup directory
sge_model_setup_path = get_paths('scripts') / 'sge_model_setup' / 'sge_model_setup.py'

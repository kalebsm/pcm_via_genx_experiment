# import Pkg
# # Pkg.add("PyCall")
ENV["PYTHON"] = "C:\\Users\\ks885\\AppData\\Local\\anaconda3\\envs\\master-env\\python.exe"
# Sys.which("python")
using Pkg
Pkg.build("PyCall")
using PyCall
# show python interpretor to ensure environment is correct
println(PyCall.python)
# process_data.jl


# println("Path to sge_utils.jl: ", joinpath(@__DIR__, "utils", "sge_utils.jl"))
# Include the sge_utils.jl file
include(joinpath(@__DIR__, "utils", "sge_utils.jl"))  # Correct path to sge_utils.jl
# println("sge_utils.jl included.")

# Use the Utils module after it's included
using .Utils  # This will make the 'Utils' module and its functions available

# Add the root directory to the Julia LOAD_PATH (from within the scripts folder)
push!(LOAD_PATH, joinpath(@__DIR__, ".."))
# println("Current LOAD_PATH: ", LOAD_PATH)

# names(Main)

# # Now you can use get_paths because it is part of the Utils module
script_path = Utils.get_paths("scripts")  # Using Utils to get the 'scripts' path
# println("Scripts path: ", script_path)

python_script_path = joinpath(script_path, "sge_model_setup", "sge_model_setup.py")

# Ensure the Python environment is correctly set up to find the script
# Add the directory containing the Python script to the Python path

print(python_script_path)

# py"""
# import sys
# # print(sys.executable)
# # print(sys.path)

# try:
#     import win32api as pd
#     print("Package successfully imported!")
# except ModuleNotFoundError as e:
#     print("ERROR:", e)
# """


py"""
import os
import sys
import subprocess

python_executable = sys.executable
env = os.environ.copy()
env["PYTHONPATH"] = "C:\\Users\\ks885\\AppData\\Local\\anaconda3\\envs\\master-env\\Lib\\site-packages"

print("Running script with PYTHONPATH:", env["PYTHONPATH"])
subprocess.run([python_executable, $python_script_path], check=True, env=env)
"""



# py"""
# import sys
# import os
# import pandas as pd
# import numpy as np

# print("Current working directory: ", os.getcwd())
# print("Provided script path:", $python_script_path)
# # Add the directory of sge_model_setup.py to sys.path
# sys.path.append(os.path.dirname($python_script_path))
# # run the Python script
# # Run the script using subprocess (ensures proper environment)
# import subprocess
# subprocess.run(["python", $python_script_path])

# """




println("Scripts path: ", script_path)

python_script_dir = joinpath(script_path, "sge_model_setup")

# Ensure the Python environment is correctly set up to find the script
# Add the directory containing the Python script to the Python path

print(python_script_dir)
py"""
import sys
# Add the directory of sge_model_setup.py to sys.path
sys.path.append("$python_script_dir")
# run the Python script
import sge_model_setup
sge_model_setup.run()

"""

# import sge_model_setup
# sge_model_setup.run()
# # Now import and run the script
# from sge_model_setup import sge_model_setup  # Assuming sge_model_setup.py has a module named sge_model_setup

# Run a function from sge_model_setup.py, for example, a 'run' function
# sge_model_setup.run()  # This assumes your Python script has a function 'run' to execute

println("Python script has been executed successfully!")



### run CEM


### copy capacities from CEM to LACs


### run LACs


### print figures



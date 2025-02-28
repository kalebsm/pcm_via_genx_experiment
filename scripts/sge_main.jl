import Pkg
Pkg.add("PyCall")

using PyCall

# Add the root directory to the Julia LOAD_PATH (from within the scripts folder)
push!(LOAD_PATH, joinpath(@__DIR__, ".."))

# Now you can import utils or any module that requires data folder access
@pyimport utils.sge_utils as sge_utils

# Run the sge_model_setup.py script in the sge_model_setup directory
sge_model_setup_path = joinpath(sge_utils.get_paths("scripts"), "sge_model_setup", "sge_model_setup.py")
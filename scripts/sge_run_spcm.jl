# Include the sge_utils.jl file
include(joinpath(@__DIR__, "utils", "sge_utils.jl"))  # Correct path to sge_utils.jl

# Use the Utils module after it's included
using .Utils  # This will make the 'Utils' module and its functions available

# Activate the project in the root folder
using Pkg
# maybe replace with dynamic path, Dr.Watson, or utils
Pkg.activate(joinpath(@__DIR__, "..")) 

push!(Base.LOAD_PATH, "SPCM")
Pkg.activate("SPCM")
Pkg.instantiate()

genx_path = Utils.get_paths("spcm_research")

# get case list
data_path = Utils.get_paths("data")
case_list = readdir(data_path * "/cases")
case_list = [replace(case, ".csv" => "") for case in case_list]

# loop through case list and include all run files

for case in case_list
    include(genx_path * "/" * case * "/Run.jl")
end

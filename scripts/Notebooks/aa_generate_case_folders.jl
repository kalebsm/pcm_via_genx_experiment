using Dates
using FilePathsBase

# create a function that takes the current case and loc and generates folders: 
# policies, resources, settings, system
function create_case_folders(case_name, base_loc)
    folders = ["policies", "resources", "settings", "system"]
    for folder in folders
        folder_path = joinpath(base_loc, folder)
        if !isdir(folder_path)
            mkpath(folder_path)
        end
        if folder == "resources"
            policy_assignments_path = joinpath(folder_path, "policy_assignments")
            if !isdir(policy_assignments_path)
                mkpath(policy_assignments_path)
            end
        end
    end
end

# define location of cost assumptions
script_dir = @__DIR__
generator_assumptions_path = joinpath(script_dir, "..", "data", "cases")
# define path locations for CEM and LACs where inputs are going
genx_cem_loc = joinpath(script_dir, "..", "GenX.jl", "research_systems")
spcm_lac_loc = joinpath(script_dir, "..", "SPCM", "research_systems")

# Get the list of all files in the generator_assumptions_path directory
case_names_list = []
for xlsx_name in readdir(generator_assumptions_path)
    if isfile(joinpath(generator_assumptions_path, xlsx_name))
        case_name = replace(xlsx_name, ".xlsx" => "")
        push!(case_names_list, case_name)
    end
end
println(case_names_list)



# create folders in GenX and SPCM research systems for each case
for case_name in case_names_list
    genx_cem_unabr_case_loc = joinpath(genx_cem_loc, case_name)
    # genx_cem_abbr_case_loc = joinpath(genx_cem_loc, case_name * "_abbr")
    spcm_lac_case_loc = joinpath(spcm_lac_loc, case_name)

    if !isdir(genx_cem_unabr_case_loc)
        mkpath(genx_cem_unabr_case_loc)
    end
    create_case_folders(case_name, genx_cem_unabr_case_loc)
    # if !isdir(genx_cem_abbr_case_loc)
    #     mkpath(genx_cem_abbr_case_loc)
    # end
    # create_case_folders(case_name, genx_cem_abbr_case_loc)
    if !isdir(spcm_lac_case_loc)
        mkpath(spcm_lac_case_loc)
    end
    create_case_folders(case_name, spcm_lac_case_loc)
end

module Utils

using Printf

function get_paths(path_type::String)
    # Define the mapping for environment variables
    env_var_map = Dict(
        "scripts" => "SCRIPTS_PATH",
        "data" => "DATA_PATH",
        "genx_research" => "GENX_RESEARCH_PATH",
        "spcm" => "SPCM_PATH",
        "spcm_research" => "SPCM_RESEARCH_PATH",
        "figures" => "FIGURES_PATH",
        "scenario_generation" => "SCENARIO_GENERATION_PATH"
    )
    
    # Default path definitions, relative to the root directory
    default_path_map = Dict(
        "scripts" => joinpath(@__DIR__, "..", "..", "scripts"),  # Going up two levels to root
        "data" => joinpath(@__DIR__, "..", "..", "data"),        # Data folder in the root
        "genx_research" => joinpath(@__DIR__, "..", "..", "GenX.jl", "research_systems"),
        "spcm" => joinpath(@__DIR__, "..", "..", "SPCM"),
        "spcm_research" => joinpath(@__DIR__, "..", "..", "SPCM", "research_systems"),
        "figures" => joinpath(@__DIR__, "..", "..", "figures"),
        "scenario_generation"=>  joinpath(@__DIR__, "..", "..", "scenario_generation")
    )
    
    # Try to get the environment variable for the specified path type
    env_var = get(env_var_map, path_type, nothing)
    if env_var !== nothing && haskey(ENV, env_var)
        path = ENV[env_var]
        if !isempty(path)
            return path
        end
    end
    
    # Return the default path if the environment variable is not set
    return get(default_path_map, path_type, nothing)
end

end  # module

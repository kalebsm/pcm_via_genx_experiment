

module SPCMviaGenX

# thanks, ChatGPT
function include_all_in_folder(folder)
    base_path = joinpath(@__DIR__, folder)
    for (root, dirs, files) in Base.Filesystem.walkdir(base_path)
        for file in files
            if endswith(file, ".jl")
                include(joinpath(root, file))


            end
        end
    end
end

#export package_activate
export configure_settings
export configure_solver
export load_inputs
export load_dataframe
export generate_model
export solve_model
export write_outputs
export cluster_inputs
export mga
export morris
export choose_output_dir

export existing_cap_mw

# fuel
export start_fuel_mmbtu_per_mw
export fuel
export heat_rate_mmbtu_per_mwh



export ids_with_positive
export findall
export ids_with
export resources_in_zone_by_rid
export resource_id
export num_vre_bins
export existing_cap_mwh

export min_duration
export max_duration
export var_om_cost_per_mwh
export var_om_cost_per_mwh_in
export efficiency_down
export efficiency_up
export self_discharge
export hoursbefore

export reg_max
export rsv_max
export extract_time_series_to_expression

export cap_size
export ramp_up_fraction
export min_power
export ramp_down_fraction
export up_time
export down_time

export fixed_om_cost_per_mwyr
export inv_cost_per_mwyr
export inv_cost_per_mwhyr



# Multi-stage methods
export run_ddp
export configure_multi_stage_inputs
export load_inputs_multi_stage
export compute_cumulative_min_retirements!
export write_multi_stage_outputs
export run_genx_case!
export run_timedomainreduction!

# export scenario generation functions
export read_h5_file
export compute_hourly_average_actuals
export convert_hours_2018
export bind_historical_forecast
export compute_landing_probability
export transform_landing_probability
export generate_lower_triangular_correlation
export generate_norta_scenarios


using JuMP # used for mathematical programming
using DataFrames
using CSV
using StatsBase
using LinearAlgebra
using YAML
using Dates
using Clustering
using Distances
using Combinatorics
using Random
using RecursiveArrayTools
using Statistics
using HDF5

# sequential_norta packagees
using Dates
using DelimitedFiles
using Distributions
# using LaTeXStrings
using LinearSolve
using Plots
using Tables
import TSFrames: TSFrame  # Do not import TSFrames.nrow
import TSFrames: apply
using TimeZones

# Uncomment if Gurobi or CPLEX active license and installations are there and the user intends to use either of them
#using CPLEX
using Gurobi
#using CPLEX
#using MOI
#using SCIP
using HiGHS
using Logging

using PrecompileTools: @compile_workload

# Global scaling factor used when ParameterScale is on to shift values from MW to GW
# DO NOT CHANGE THIS (Unless you do so very carefully)
# To translate MW to GW, divide by ModelScalingFactor
# To translate $ to $M, multiply by ModelScalingFactor^2
# To translate $/MWh to $M/GWh, multiply by ModelScalingFactor
const ModelScalingFactor = 1e+3

"""
An abstract type that should be subtyped for users creating GenX resources.
"""
abstract type AbstractResource end


# using Pkg

# activate scenario generation
seq_norta_path = joinpath("src","scenario_generation", "sequential_norta")

# push!(LOAD_PATH, seq_norta_path)
# activate the project at the sequential_norta directory
using Pkg
# Pkg.add(path=seq_norta_path)
Pkg.develop(path=seq_norta_path)
# Pkg.instantiate()

# # include(joinpath(seq_norta_path,"src","sequential_norta.jl"))
# include_all_in_folder(joinpath(seq_norta_path,"src","sequential_norta.jl"))




# include_all_in_folder("case_runners")
include_all_in_folder("configure_settings")
include_all_in_folder("configure_solver")
include_all_in_folder("load_inputs")
include_all_in_folder("model")
include_all_in_folder("scenario_generation")
# include_all_in_folder("write_outputs")

# include("time_domain_reduction/time_domain_reduction.jl")
# include("time_domain_reduction/precluster.jl")
# include("time_domain_reduction/full_time_series_reconstruction.jl")

# include_all_in_folder("multi_stage")
include_all_in_folder("additional_tools")

# include("startup/genx_startup.jl")

end

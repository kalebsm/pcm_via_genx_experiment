import Pkg

# Activate the main project
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

# Include your core module
ENV["GENX_PRECOMPILE"] = "false"
include(joinpath(@__DIR__, "..", "..", "src", "SPCMviaGenX.jl"))
using .SPCMviaGenX

# Load the dev-tracked package
using SequentialNorta

using Gurobi
using YAML
using BenchmarkTools
using JuMP
using DataFrames
using CSV
using DelimitedFiles
using Distributions
using Random
using HDF5
using Plots # does not work for some reason
using TimeZones
using TSFrames
using Tables
using LinearSolve
using LinearAlgebra
using Dates
using DelimitedFiles
using StatsBase

case = dirname(@__FILE__)
folder_name_parts = split(case, "\\")
case_name = folder_name_parts[end]

# get model type XXX make automatic, update for PF vs LAC
# model_type = "dlac-p"
# determine model type from file name after Run_
model_type = split(split(basename(@__FILE__), ".jl")[1], "_")[3]

test_dictionary = Dict(
    "test_scenario_path" => 1,
    "test_scenario_lookahead_path" => 0,
    "test_prices_scen_path" => 1
)

model_context = initialize_policy_model(case)

model_ep, inputs, context = run_policy_instance(model_context, model_type, test_dictionary; write_results=true)


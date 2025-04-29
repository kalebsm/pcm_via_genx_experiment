# Add a small quadratic term to the objective function for smoother price formation
function add_quadratic_regularization(EP, generators, W, T; regularization_weight=1e-6)
    num_gen = length(generators)  # Number of scenarios
    # Create quadratic regularization expression
    @expression(EP, eQuadraticReg[w=1:W], 
        regularization_weight * sum(
            EP[:vP][y,t,w]^2 for y in 1:num_gen, t in 1:T
        )/2
    )
    
    # Add regularization to the objective
    EP[:eObj] += eQuadraticReg
    
    return EP
end

function initialize_policy_model(case::AbstractString; offset=true)
    # case = dirname(@__FILE__)
    optimizer = Gurobi.Optimizer

    # Print GenX version
    ascii_art = raw"""
    ______   _______     ______  ____    ____  
    .' ____ \ |_   __ \  .' ___  ||_   \  /   _| 
    | (___ \_|  | |__) |/ .'   \_|  |   \/   |   
    _.____`.   |  ___/ | |         | |\  /| |   
    | \____) | _| |_    \ `.___.'\ _| |_\/_| |_  
    \______.'|_____|    `.____ .'|_____||_____| 
                                            
    _   __  __   ,--.                           
    [ \ [  ][  | `'_\ :                          
    \ \/ /  | | // | |,                         
    \__/__[___]\'-;__/        ____  ____       
    .' ___  |                 |_  _||_  _|      
    / .'   \_|  .---.  _ .--.    \ \  / /        
    | |   ____ / /__\\[ `.-. |    > `' <         
    \ `.___]  || \__., | | | |  _/ /'`\ \_       
    `._____.'  '.__.'[___||__]|____||____|     
    """
    println(ascii_art)

    genx_settings = get_settings_path(case, "genx_settings.yml") # Settings YAML file path
    writeoutput_settings = get_settings_path(case, "output_settings.yml") # Write-output settings YAML file path
    setup = configure_settings(genx_settings, writeoutput_settings) # setup dictionary stores settings and GenX-specific parameters

    # Configure solver
    println("Configuring Solver")
    settings_path = get_settings_path(case)
    solver_name = lowercase(get(setup, "Solver", ""))
    OPTIMIZER = configure_solver(settings_path, optimizer; solver_name=solver_name)

    # Cluster time series inputs if necessary and if specified by the user
    if setup["TimeDomainReduction"] == 1
        TDRpath = joinpath(case, setup["TimeDomainReductionFolder"])
        system_path = joinpath(case, setup["SystemFolder"])
        prevent_doubled_timedomainreduction(system_path)
        if !time_domain_reduced_files_exist(TDRpath)
            println("Clustering Time Series Data (Grouped)...")
            cluster_inputs(case, settings_path, setup)
        else
            println("Time Series Data Already Clustered.")
        end
    end

    # Load inputs
    println("Loading Inputs")
    inputs = load_inputs(setup, case)

    # update omega and fuel_costs and c_start for lookahead length
    inputs["omega"] = vcat(inputs["omega"], ones(48))
    # for each fuel in fuel_costs, add 48 rows of last value
    for fuel in keys(inputs["fuel_costs"])
        inputs["fuel_costs"][fuel] = vcat(inputs["fuel_costs"][fuel], repeat([inputs["fuel_costs"][fuel][end]], 48))
    end
    # for each generator, add 48 rows of last value
    inputs["C_Start"] = hcat(inputs["C_Start"], repeat(inputs["C_Start"][:, end:end], 1, 48))
    # Should be defined in module, but doesn't get read in these run files???
    ModelScalingFactor = 1e+3
    
    #=======================================================================
    Set Case Specific Parameters
    =======================================================================#
    folder_name_parts = split(case, "\\")
    case_name = folder_name_parts[end]

    ### Load in Scenario Generation information
    scen_generator = scenario_generator_init()
    

    # define CEM path
    cem_path = joinpath(case, "..", "..", "..", "GenX.jl", "research_systems", case_name)
    cem_results_path = joinpath(cem_path, "results")
    if offset
        pf_results_path = joinpath(cem_path, "results_pf")
        pf_rev = CSV.read(joinpath(pf_results_path, "NetRevenue.csv"), DataFrame)
        offset_ = pf_rev.diff[:]
    else
        num_gen = inputs["G"]
        offset_ = zeros(num_gen)
    end    # load commit, commit, commit_dp
    cem_commit_raw = CSV.read(joinpath(cem_results_path, "commit.csv"), DataFrame)
    # Remove the first two rows and reset the index for `cem_commit`
    cem_commit = cem_commit_raw[3:end, :]

    # load startup, start, start_dp
    cem_start_raw = CSV.read(joinpath(cem_results_path, "start.csv"), DataFrame)
    # Remove the first two rows and reset the index for `cem_start`
    cem_start = cem_start_raw[3:end, :]

    # load shut down, shutdown, shut_dp
    cem_shut_raw = CSV.read(joinpath(cem_results_path, "shutdown.csv"), DataFrame)
    # Remove the first two rows and reset the index for `cem_shut`
    cem_shut = cem_shut_raw[3:end, :]

    # load state of charge, storage, s_dp
    cem_soc_raw = CSV.read(joinpath(cem_results_path, "storage.csv"), DataFrame)
    # Remove the first two rows and reset the index for `cem_soc`
    cem_soc = cem_soc_raw[3:end, :]

    # load dispatch, power, pgen_dp
    cem_dispatch_raw = CSV.read(joinpath(cem_results_path, "power.csv"), DataFrame)
    # Remove the first two rows and reset the index for `cem_dispatch`
    cem_dispatch = cem_dispatch_raw[3:end, :]
    # Create a context dictionary to store all the required data
    context = Dict(
        "setup" => setup,
        "inputs" => inputs,
        "OPTIMIZER" => OPTIMIZER,
        "ModelScalingFactor" => ModelScalingFactor,
        "case_name" => case_name,
        "scen_generator" => scen_generator,
        "cem_commit" => cem_commit,
        "cem_start" => cem_start,
        "cem_shut" => cem_shut,
        "cem_soc" => cem_soc,
        "cem_dispatch" => cem_dispatch,
        "case" => case,
        "offset" => offset_
    )
    
    return context
end

function run_policy_model_new(context::Dict, model_type::AbstractString, existing_capacities = []; 
                          write_results::Bool=false)
    start_time = time()
    # Unpack variables from context
    setup = context["setup"]
    inputs = deepcopy(context["inputs"])  # Deep copy to avoid modifying original
    OPTIMIZER = context["OPTIMIZER"]
    ModelScalingFactor = context["ModelScalingFactor"]
    case_name = context["case_name"]
    scen_generator = context["scen_generator"]
    cem_commit = context["cem_commit"]
    cem_start = context["cem_start"]
    cem_shut = context["cem_shut"]
    cem_soc = context["cem_soc"]
    cem_dispatch = context["cem_dispatch"]
    case = context["case"]
    offset = context["offset"]
    # Apply any new existing capacities if provided
    if !isempty(existing_capacities)
        for (idx, capacity) in enumerate(existing_capacities)
            inputs["RESOURCES"][idx].existing_cap_mw = capacity
            if idx in inputs["STOR_ALL"]
                inputs["RESOURCES"][idx].existing_cap_mwh = capacity.*2
            end
        end
    end
    
    # Unpack scenario generator variables
    unique_forecast_times = scen_generator["unique_forecast_times"]
    unique_issue_times = scen_generator["unique_issue_times"]
    start_date = scen_generator["start_date"]
    corr_forecast_issue_times = scen_generator["corr_forecast_issue_times"]
    forecast_scenario_length = scen_generator["forecast_scenario_length"]
    number_of_scenarios = scen_generator["number_of_scenarios"]
    solar_model_data = scen_generator["solar_model_data"]
    M_load = scen_generator["M_load"]
    M_solar = scen_generator["M_solar"]
    M_wind = scen_generator["M_wind"]
    lp_solar = scen_generator["lp_solar"]
    load_marginals_by_issue = scen_generator["load_marginals_by_issue"]
    solar_marginals_by_issue = scen_generator["solar_marginals_by_issue"]
    wind_marginals_by_issue = scen_generator["wind_marginals_by_issue"]
    load_landing_probabilities = scen_generator["load_landing_probabilities"]
    solar_landing_probabilities = scen_generator["solar_landing_probabilities"]
    wind_landing_probabilities = scen_generator["wind_landing_probabilities"]
    load_actual_avg = scen_generator["load_actual_avg"]
    solar_actual_avg = scen_generator["solar_actual_avg"]
    wind_actual_avg = scen_generator["wind_actual_avg"]
    solar_well_defined_cols = scen_generator["solar_well_defined_cols"]
    solar_issue_decn_time_matrix = scen_generator["solar_issue_decn_time_matrix"]
    load_actual_avg_GW = scen_generator["load_actual_avg_GW"]
    solar_actual_avg_cf = scen_generator["solar_actual_avg_cf"]
    wind_actual_avg_cf = scen_generator["wind_actual_avg_cf"]
    decision_mdl_lkd_length = scen_generator["decision_mdl_lkd_length"]
    max_solar_actual = scen_generator["max_solar_actual"];
    max_wind_actual = scen_generator["max_wind_actual"];
    start_date = scen_generator["start_date"];
    #=======================================================================
    DEFINE INDICES, DATETIMES, ISSUE SETS FOR NORTA SCENARIOS AND STOCASTIC SIM
    =======================================================================#
    # Set the date and time for the forecasts
    start_index = findfirst(isequal(start_date), unique_forecast_times)

    Tstart = start_index
    Tend = decision_mdl_lkd_length - 50 # should be -48 probably

    rh_len = forecast_scenario_length # scenario_length

    if model_type == "pf"
        R = 1
    elseif model_type == "dlac-p" || model_type == "dlac-i" || model_type == "slac"
        R = range(Tstart, Tend, step=1)
    else
        error("Model type not recognized")
    end

    # Rh as a dictionary
    Rh = Dict()

    # INTERIOR_SUBPERIODS
    for r in R
        Rh[r] = r:1:r+forecast_scenario_length # scenario_length
    end

    # initialize array for saving history of time indexes
    RhHistory = []

    # Extract resource information
    gen = inputs["RESOURCES"]
    zones = zone_id.(gen)
    regions = region.(gen)
    clusters = cluster.(gen)
    rid = resource_id.(gen)
    resource_names = inputs["RESOURCE_NAMES"]

    COMMIT = inputs["COMMIT"]
    THERM_COMMIT = inputs["THERM_COMMIT"]
    STOR_LIST = inputs["STOR_ALL"]
    STOR_ALL = inputs["STOR_ALL"]
    VRE_LIST = inputs["VRE"]
    if setup["OperationalReserves"] >= 1
        RSV = inputs["RSV"]
        REG = inputs["REG"]
    end

    num_gen = inputs["G"]
    G = inputs["G"] 
    Z = inputs["Z"]

    # Initialize dictionaries for saved variables
    var_strings = ["P", "RSV", "REG", "NSE", "COMMIT", "START", "SHUT", "CHARGE", "S"]
    pri_strings = ["PowerBalance", "Reg", "RsvReq"]

    var_dict = Dict(var_strings[i] => Dict() for i in 1:length(var_strings))
    pri_dict = Dict()

    # Initialize storage for results
    pgen_dp = zeros(num_gen, Tend)
    rsv_dp = zeros(num_gen, Tend)
    reg_dp = zeros(num_gen, Tend)
    nse_dp = zeros(Z, Tend)
    unmet_rsv_dp = zeros(Z, Tend)
    commit_dp = zeros(num_gen, Tend)
    start_dp = zeros(num_gen, Tend)
    shut_dp = zeros(num_gen, Tend)
    charge_dp = zeros(num_gen, Tend)
    s_dp = zeros(num_gen, Tend)

    # Initialize price storage
    elec_prices = zeros(Z, Tend)
    reg_prices = zeros(Z, Tend)
    rsv_prices = zeros(Z, Tend)

    for price_key in pri_strings
        pri_dict[price_key] = zeros(Z, Tend)
    end

    max_discharge_const_duals = Array{Any}(undef, Tend)
    max_charge_const_duals = Array{Any}(undef, Tend)
    soc_link_duals = Array{Any}(undef, Tend)
    soc_int_duals = Array{Any}(undef, Tend)

    # Initialize constant costs
    var_om_cost_per_gen = [var_om_cost_per_mwh(gen[y]) for y in 1:G]
    var_om_cost_in_per_gen = [y in STOR_ALL ? var_om_cost_per_mwh_in(gen[y]) : 0 for y in 1:G]
    fixed_om_cost_per_gen = [fixed_om_cost_per_mwyr(gen[y]) for y in 1:G]

    existing_cap_mw_per_gen = [existing_cap_mw(gen[y]) for y in 1:G]
    existing_cap_mwh_per_gen = [existing_cap_mwh(gen[y]) for y in 1:G]
    generator_name_per_gen = [gen[y].resource for y in 1:G]

    inv_cost_per_mwyr_per_gen = [inv_cost_per_mwyr(gen[y]) for y in 1:G]
    inv_cost_per_mwhyr_per_gen = [inv_cost_per_mwhyr(gen[y]) for y in 1:G]

    fuel_costs = inputs["fuel_costs"]

    fuel_cost_per_mmbtu = [fuel_costs[fuel(gen[y])][:] for y in 1:G]
    fuel_cost_per_mmbtu = transpose(hcat(fuel_cost_per_mmbtu...))

    # Initialize hourly components
    energy_revs_dp = zeros(num_gen, Tend)
    reg_revs_dp = zeros(num_gen, Tend)
    rsv_revs_dp = zeros(num_gen, Tend)
    var_om_costs_dp = zeros(num_gen, Tend)
    fuel_costs_dp = zeros(num_gen, Tend)
    start_costs_dp = zeros(num_gen, Tend)
    charge_costs_dp = zeros(num_gen, Tend)

    # Initialize welfare components
    nse_cost = zeros(Z, Tend)
    unmet_rsv_cost = zeros(Z, Tend)

    # Create a NetRevenue dataframe
    dfNetRevenue = DataFrame(region = regions,
                            Resource = inputs["RESOURCE_NAMES"],
                            zone = zones,
                            Cluster = clusters,
                            R_ID = rid)

    # Initialize scenario path information
    prices_scen_array = Array{Any}(undef, Tend)

    Random.seed!(12345)
    normal_dist = Normal(0, 1)
    date = deepcopy(start_date)

    println("Generating the SPCM Optimization")

    # Loop through time periods
    for r in R
        global decision_date = start_date + Dates.Hour(r - 1)

        #=======================================================================
        DEFINE DATETIMES AND FORECAST VS ACTUAL VS MODEL TIMES OR HOURS
        =======================================================================#
        horizon_start_index = findfirst(isequal(decision_date), unique_forecast_times)

        if model_type == "pf"
            current_issue = nothing
            issue_index = nothing
            active_issues = nothing
            current_forecast_indices = nothing
            current_forecast_times = nothing
            forecast_times_start_incl = nothing
            policy_model_length = Tend
            policy_forecast_length = nothing
            policy_actuals_length = Tend
            lookahead_decision_hours = nothing
        elseif model_type == "dlac-p" || model_type == "dlac-i" || model_type == "slac"
            current_issue = corr_forecast_issue_times[horizon_start_index, :issue_time]
            issue_index = findall(x -> x == current_issue, unique_issue_times)[1]
            active_issues = [current_issue]
            current_forecast_indices = findall(x -> x in active_issues, corr_forecast_issue_times[!,:issue_time])
            current_forecast_times = corr_forecast_issue_times[current_forecast_indices, :forecast_time]
            forecast_times_start_incl = filter(x -> x >= decision_date, current_forecast_times)
            policy_model_length = length(forecast_times_start_incl)
            policy_lookahead_length = policy_model_length - 1
            policy_actuals_length = forecast_scenario_length - policy_lookahead_length
            lookahead_decision_hours = collect(policy_actuals_length:forecast_scenario_length)
        else
            error("Model type not recognized")
        end

        # Define scenarios for different model types
        if model_type == "pf" || model_type == "dlac-p"
            scen_data = DataFrame()
            new_solar = solar_actual_avg_cf[r:r + policy_model_length - 1]
            new_wind = wind_actual_avg_cf[r:r + policy_model_length - 1]
            new_load = load_actual_avg_GW[r:r + policy_model_length - 1]
            new_scen = DataFrame(S = new_solar, W = new_wind, L = new_load)
            insertcols!(scen_data, :S => new_solar, :W => new_wind, :L => new_load, makeunique=true)
        elseif model_type == "dlac-i" || model_type == "slac"
            load_current_marginals = load_marginals_by_issue[issue_index]
            solar_current_marginals = solar_marginals_by_issue[issue_index]
            wind_current_marginals = wind_marginals_by_issue[issue_index]
            
            Y_load = generate_norta_scenarios(number_of_scenarios, forecast_scenario_length, decision_date, horizon_start_index, 
                                             active_issues, corr_forecast_issue_times, normal_dist,  
                                             M_load, load_current_marginals, load_landing_probabilities,
                                             solar_well_defined_cols, false)
            
            Y_solar = generate_norta_scenarios(number_of_scenarios, forecast_scenario_length, decision_date, horizon_start_index, 
                                              active_issues, corr_forecast_issue_times, normal_dist,  
                                              M_solar, solar_current_marginals, solar_landing_probabilities,
                                              solar_well_defined_cols, true)
            
            Y_wind = generate_norta_scenarios(number_of_scenarios, forecast_scenario_length, decision_date, horizon_start_index, 
                                             active_issues, corr_forecast_issue_times, normal_dist,  
                                             M_wind, wind_current_marginals, wind_landing_probabilities,
                                             solar_well_defined_cols, false)
            
            Y_load_GWh = Y_load ./ ModelScalingFactor
            Y_solar_cf = Y_solar ./ max_solar_actual
            Y_wind_cf = Y_wind ./ max_wind_actual

            scen_data = DataFrame()

            if model_type == "dlac-i"
                load_scenario_avg_GW = mean(Y_load_GWh, dims = 1)
                solar_scenario_avg_cf = mean(Y_solar_cf, dims = 1)
                wind_scenario_avg_cf = mean(Y_wind_cf, dims = 1)

                new_solar = vec(solar_scenario_avg_cf)
                new_wind = vec(wind_scenario_avg_cf)
                new_load = vec(load_scenario_avg_GW)
                new_scen = DataFrame(S = new_solar, W = new_wind, L = new_load)
                insertcols!(scen_data, :S => new_solar, :W => new_wind, :L => new_load, makeunique=true)
            elseif model_type == "slac"
                for i in 1:size(Y_load, 1)
                    new_solar = Y_solar_cf[i,:]
                    new_wind = Y_wind_cf[i,:]
                    new_load = Y_load_GWh[i,:]
                    new_scen = DataFrame(S = new_solar, W = new_wind, L = new_load)
                    insertcols!(scen_data, :S => new_solar, :W => new_wind, :L => new_load, makeunique=true)
                end
            end
        end

        # Process scenarios
        no_col = size(scen_data)[2]
        if mod(no_col, 3) == 0
            W = round(Int, size(scen_data)[2] / 3)
        else
            print("Scenario input does not have the required dimensions")
        end

        st = r
        en = r + policy_model_length - 1

        println()
        println("Begin Horizon ", st, " to ", en)

        # Initialize scenario dictionaries
        pd_dict = Dict()
        pp_max_dict = Dict()

        for scen_idx in 1:(W)
            col_idx = 1 + (scen_idx - 1) * 3
            
            sol_column = col_idx
            wind_column = col_idx + 1
            load_column = col_idx + 2

            load_df = scen_data[!, load_column]
            load_mt = reshape(load_df, (length(load_df), 1))

            pd_dict[scen_idx] = load_mt
            pp_max_dict[scen_idx] = zeros(num_gen, policy_model_length)

            pp_max_dict[scen_idx][inputs["THERM_ALL"],:] .= ones(1, policy_model_length)
            pp_max_dict[scen_idx][inputs["STOR_ALL"],:] .= ones(1, policy_model_length)
            pp_max_dict[scen_idx][inputs["SOLAR"],:] .= transpose(scen_data[!, sol_column])
            pp_max_dict[scen_idx][inputs["WIND"],:] .= transpose(scen_data[!, wind_column])
        end

        # Set up input data for this time period
        rhinputs = deepcopy(inputs)
        rhinputs["W"] = W
        rhinputs["pD"] = pd_dict
        rhinputs["pP_Max"] = pp_max_dict
        rhinputs["omega"] = inputs["omega"][st:en]
        rhinputs["C_Start"] = inputs["C_Start"][:, st:en]

        for k in keys(inputs["fuel_costs"])
            rhinputs["fuel_costs"][k] = inputs["fuel_costs"][k][st:en]
        end

        rhinputs["INTERIOR_SUBPERIODS"] = range(2, policy_model_length, step = 1)
        INTERIOR_SUBPERIODS = rhinputs["INTERIOR_SUBPERIODS"]
        START_SUBPERIODS = 1:1
        rhinputs["T"] = policy_model_length
        rhinputs["hours_per_subperiod"] = policy_model_length

        # Generate and solve model 
        println("Generating the Optimization Model")
        presolver_start_time = time()
        
        ### Stochastic Model ###
        # Generate Stochastic Energy Portfolio (EP) Model
        EP = Model(OPTIMIZER)

        ### Generate model without calling GenX
        G = rhinputs["G"]     # Number of resources (generators, storage, DR, and DERs)
        T = rhinputs["T"]     # Number of time EPs
        Z = rhinputs["Z"]     # Number of zones
        W = rhinputs["W"]		#Number of scenarios
        SEG = rhinputs["SEG"] # Number of load curtailment segments

        # Introduce dummy variable fixed to zero to ensure that expressions like eTotalCap,
        # eTotalCapCharge, eTotalCapEnergy and eAvail_Trans_Cap all have a JuMP variable
        @variable(EP, vZERO == 0);

        # # @expression(EP, ePowerBalance[t=1:T, z=1:Z, w=1:W], zeros((T,1,W))) # pretty sure this is wrong
        @expression(EP, ePowerBalance[t=1:T, z=1:Z, w=1:W], 0)

        # objective
        @expression(EP, eObj, zeros((W,1)))

        ### is this necessary?
        @expression(EP, eGenerationByZone[z=1:Z, t=1:T, w=1:W], 0) 

        ### add discharge.jl
        @variable(EP, vP[y=1:G,t=1:T,w=1:W] >=0);

        # add the here and now decision variable
        @variable(EP, hnP[y=1:G, t=[1]] >= 0);
        @constraint(EP, DISC_nonantic[y=1:G, t=[1], w=1:W], vP[y,t,w] == hnP[y,t])

        # Variable costs of "generation" for resource "y" during hour "t" = variable O&M plus fuel cost XXX gen[y].c_fuel_per_mwh
        @expression(EP, eCVar_out[y=1:G,t=1:T, w=1:W], (rhinputs["omega"][t]*(var_om_cost_per_mwh(gen[y]))*vP[y,t,w]))
        
        # Sum individual resource contributions to variable discharging costs to get total variable discharging costs
        @expression(EP, eTotalCVarOutT[t=1:T, w=1:W], sum(eCVar_out[y,t,w] for y in 1:G))
        @expression(EP, eTotalCVarOut[w=1:W], sum(eTotalCVarOutT[t,w] for t in 1:T))

        # # Add total variable discharging cost contribution to the objective function
        EP[:eObj] += eTotalCVarOut

        # constraints
        @constraint(EP, Max_vP[y=1:G, t=1:T, w=1:W], vP[y,t,w] <= gen[y].existing_cap_mw)

        ### non_served_energy.jl
        # variables
        # Non-served energy/curtailed demand in the segment "s" at hour "t" in zone "z"
        @variable(EP, vNSE[s=1:SEG,t=1:T,z=1:Z,w=1:W] >= 0);


        @variable(EP, hnNSE[s=1:SEG,t=[1],z=1:Z] >= 0);
        @constraint(EP, NSE_nonantic[s=1:SEG,t=[1],z=1:Z,w=1:W], vNSE[s,t,z,w] == hnNSE[s,t,z])

        #expressions

        # Objective Function Expressions

        # Cost of non-served energy/curtailed demand at hour "t" in zone "z"
        @expression(EP, eCNSE[s=1:SEG,t=1:T,z=1:Z, w=1:W], (rhinputs["omega"][t]*rhinputs["pC_D_Curtail"][s]*vNSE[s,t,z,w]))

        # Sum individual demand segment contributions to non-served energy costs to get total non-served energy costs
        # Julia is fastest when summing over one row one column at a time
        @expression(EP, eTotalCNSETS[t=1:T,z=1:Z, w=1:W], sum(eCNSE[s,t,z,w] for s in 1:SEG))
        @expression(EP, eTotalCNSET[t=1:T, w=1:W], sum(eTotalCNSETS[t,z,w] for z in 1:Z))
        @expression(EP, eTotalCNSE[w=1:W], sum(eTotalCNSET[t,w] for t in 1:T))

        # Add total cost contribution of non-served energy/curtailed demand to the objective function
        EP[:eObj] += eTotalCNSE

        ## Power Balance Expressions ##
        @expression(EP, ePowerBalanceNse[t=1:T, z=1:Z, w=1:W],
            sum(vNSE[s,t,z,w] for s=1:SEG))

        # Add non-served energy/curtailed demand contribution to power balance expression
        EP[:ePowerBalance] += ePowerBalanceNse

        # Constratints

        @constraint(EP, cNSEPerSeg[s=1:SEG, t=1:T, z=1:Z, w=1:W], vNSE[s,t,z,w] <= rhinputs["pMax_D_Curtail"][s]*rhinputs["pD"][w][t,z]) 

        @constraint(EP, cMaxNSE[t=1:T, z=1:Z, w=1:W], sum(vNSE[s,t,z,w] for s=1:SEG) <= rhinputs["pD"][w][t,z])

        ### investment_discharge
        @expression(EP, eExistingCap[y in 1:G], gen[y].existing_cap_mw)

        # is this needed?
        @expression(EP, eTotalCap[y in 1:G], eExistingCap[y] + EP[:vZERO])
        
        # individaul fixed costs
        @expression(EP, eCFix[y in 1:G],
                gen[y].fixed_om_cost_per_mwyr * eTotalCap[y]
        )
        # Sum individual resource contributions to fixed costs to get total fixed costs
        @expression(EP, eTotalCFix, sum(EP[:eCFix][y] for y in 1:G))
        # add fixed cost to objective
        EP[:eObj] += eTotalCFix .* ones(W,1)

        ### Update Intertemporal Constraints commented out in generate_model files

        ### UCommit
        COMMIT = rhinputs["COMMIT"] # For not, thermal resources are the only ones eligible for Unit Committment

        ### Variables ###

        ## Decision variables for unit commitment
        # commitment state variable
        @variable(EP, vCOMMIT[y in COMMIT, t=1:T, w=1:W] >= 0)
        # startup event variable
        @variable(EP, vSTART[y in COMMIT, t=1:T, w=1:W] >= 0)
        # shutdown event variable
        @variable(EP, vSHUT[y in COMMIT, t=1:T, w=1:W] >= 0)

        ### Here and Now Variables
        # commitment state variable
        @variable(EP, hnCOMMIT[y in COMMIT, t=[1]] >= 0)
        # startup event variable
        @variable(EP, hnSTART[y in COMMIT, t=[1]] >= 0)
        # shutdown event variable
        @variable(EP, hnSHUT[y in COMMIT, t=[1]] >= 0)

        ### Nonanticipativity constraints for each variable
        @constraint(EP, COMMIT_nonantic[y in COMMIT, t=[1], w=1:W], vCOMMIT[y,t,w] == hnCOMMIT[y,t])
        @constraint(EP, START_nonantic[y in COMMIT, t=[1], w=1:W], vSTART[y,t,w] == hnSTART[y,t])
        @constraint(EP, SHUT_nonantic[y in COMMIT, t=[1], w=1:W], vSHUT[y,t,w] == hnSHUT[y,t])

        ### Expressions ###

        ## Objective Function Expressions ##

        # Startup costs of "generation" for resource "y" during hour "t"
        @expression(EP, eCStart[y in COMMIT, t=1:T, w=1:W],(rhinputs["omega"][t]*rhinputs["C_Start"][y]*vSTART[y,t,w]))

        # Julia is fastest when summing over one row one column at a time
        @expression(EP, eTotalCStartT[t=1:T, w=1:W], sum(eCStart[y,t,w] for y in COMMIT))
        @expression(EP, eTotalCStart[w=1:W], sum(eTotalCStartT[t,w] for t=1:T))

        EP[:eObj] += eTotalCStart

        ### reserves

        # REG = rhinputs["REG"]
        # RSV = rhinputs["RSV"]

        ## Decision variables for reserves
        @variable(EP, vREG[y in REG, t=1:T, w=1:W] >= 0) # Contribution to regulation (primary reserves), assumed to be symmetric (up & down directions equal)
        @variable(EP, vRSV[y in RSV, t=1:T, w=1:W] >= 0) # Contribution to operating reserves (secondary reserves or contingency reserves); only model upward reserve requirements

        # Storage techs have two pairs of auxilary variables to reflect contributions to regulation and reserves
        # when charging and discharging (primary variable becomes equal to sum of these auxilary variables)
        @variable(EP, vREG_discharge[y in intersect(rhinputs["STOR_ALL"], REG), t=1:T, w=1:W] >= 0) # Contribution to regulation (primary reserves) (mirrored variable used for storage devices)
        @variable(EP, vRSV_discharge[y in intersect(rhinputs["STOR_ALL"], RSV), t=1:T, w=1:W] >= 0) # Contribution to operating reserves (secondary reserves) (mirrored variable used for storage devices)
        @variable(EP, vREG_charge[y in intersect(rhinputs["STOR_ALL"], REG), t=1:T, w=1:W] >= 0) # Contribution to regulation (primary reserves) (mirrored variable used for storage devices)
        @variable(EP, vRSV_charge[y in intersect(rhinputs["STOR_ALL"], RSV), t=1:T, w=1:W] >= 0) # Contribution to operating reserves (secondary reserves) (mirrored variable used for storage devices)

        @variable(EP, vUNMET_RSV[t=1:T, w=1:W] >= 0) # Unmet operating reserves penalty/cost


        # ## Decision variables for reserves
        @variable(EP, hnREG[y in REG, t=[1]] >= 0) # Contribution to regulation (primary reserves), assumed to be symmetric (up & down directions equal)
        @variable(EP, hnRSV[y in RSV, t=[1]] >= 0) # Contribution to operating reserves (secondary reserves or contingency reserves); only model upward reserve requirements

        # # Storage techs have two pairs of auxilary variables to reflect contributions to regulation and reserves
        # # when charging and discharging (primary variable becomes equal to sum of these auxilary variables)
        @variable(EP, hnREG_discharge[y in intersect(rhinputs["STOR_ALL"], REG), t=[1]] >= 0) # Contribution to regulation (primary reserves) (mirrored variable used for storage devices)
        @variable(EP, hnRSV_discharge[y in intersect(rhinputs["STOR_ALL"], RSV), t=[1]] >= 0) # Contribution to operating reserves (secondary reserves) (mirrored variable used for storage devices)
        @variable(EP, hnREG_charge[y in intersect(rhinputs["STOR_ALL"], REG), t=[1]] >= 0) # Contribution to regulation (primary reserves) (mirrored variable used for storage devices)
        @variable(EP, hnRSV_charge[y in intersect(rhinputs["STOR_ALL"], RSV), t=[1]] >= 0) # Contribution to operating reserves (secondary reserves) (mirrored variable used for storage devices)

        @variable(EP, hnUNMET_RSV[t=[1]] >= 0) # Unmet operating reserves penalty/cost

        # Nonanticipativity constraints for each variable type

        @constraint(EP, REG_nonantic[y in REG, t=[1], w=1:W], vREG[y,t,w] == hnREG[y,t])
        @constraint(EP, RSV_nonantic[y in REG, t=[1], w=1:W], vRSV[y,t,w] == hnRSV[y,t])

        @constraint(EP, REG_discharge_nonantic[y in intersect(rhinputs["STOR_ALL"], REG), t=[1], w=1:W], vREG_discharge[y,t,w] == hnREG_discharge[y,t])
        @constraint(EP, RSV_discharge_nonantic[y in intersect(rhinputs["STOR_ALL"], RSV), t=[1], w=1:W], vRSV_discharge[y,t,w] == hnRSV_discharge[y,t])
        @constraint(EP, REG_charge_nonantic[y in intersect(rhinputs["STOR_ALL"], REG), t=[1], w=1:W], vREG_charge[y,t,w] == hnREG_charge[y,t])
        @constraint(EP, RSV_charge_nonantic[y in intersect(rhinputs["STOR_ALL"], RSV), t=[1], w=1:W], vRSV_charge[y,t,w] == hnRSV_charge[y,t])

        @constraint(EP, UNMET_RSV_nonantic[t=[1], w=1:W], vUNMET_RSV[t,w] == hnUNMET_RSV[t])

        ### Expressions ###
        ## Total system reserve expressions
        # Regulation requirements as a percentage of load and scheduled variable renewable energy production in each hour
        # Reg up and down requirements are symmetric
        @expression(EP, eRegReq[t=1:T, w=1:W], rhinputs["pReg_Req_Demand"]*sum(rhinputs["pD"][w][t,z] for z=1:Z) +
        rhinputs["pReg_Req_VRE"]*sum(rhinputs["pP_Max"][w][y,t]*EP[:eTotalCap][y] for y in intersect(rhinputs["VRE"], rhinputs["MUST_RUN"])))
        # Operating reserve up / contingency reserve requirements as ˚a percentage of load and scheduled variable renewable energy production in each hour
        # and the largest single contingency (generator or transmission line outage)
        @expression(EP, eRsvReq[t=1:T, w=1:W], rhinputs["pRsv_Req_Demand"]*sum(rhinputs["pD"][w][t,z] for z=1:Z) +
        rhinputs["pRsv_Req_VRE"]*sum(rhinputs["pP_Max"][w][y,t]*EP[:eTotalCap][y] for y in intersect(rhinputs["VRE"], rhinputs["MUST_RUN"])))


        ## Objective Function Expressions ##

        # Penalty for unmet operating reserves
        @expression(EP, eCRsvPen[t=1:T, w=1:W], rhinputs["omega"][t]*rhinputs["pC_Rsv_Penalty"]*vUNMET_RSV[t,w])
        @expression(EP, eTotalCRsvPen[w=1:W], sum(eCRsvPen[t,w] for t=1:T) +
            sum(gen[y].reg_cost * vRSV[y,t,w] for y in RSV, t=1:T) +
            sum(gen[y].rsv_cost * vREG[y,t,w] for y in REG, t=1:T) )
        EP[:eObj] += eTotalCRsvPen

        ### Constraints ###

        ## Total system reserve constraints
        # Regulation requirements as a percentage of load and scheduled variable renewable energy production in each hour
        # Note: frequencty regulation up and down requirements are symmetric and all resources contributing to regulation are assumed to contribute equal capacity to both up and down directions
        @constraint(EP, cReg[t=1:T, w=1:W], sum(vREG[y,t,w] for y in REG) >= EP[:eRegReq][t,w])

        @constraint(EP, cRsvReq[t=1:T, w=1:W], sum(vRSV[y,t,w] for y in RSV) + vUNMET_RSV[t,w] >= EP[:eRsvReq][t,w])

        @constraint(EP, cUNMET_RSV_MAX[t=1:T,w=1:W], vUNMET_RSV[t,w]  <= EP[:eRsvReq][t,w])

        @constraint(EP, cREG_MAX[y in REG,t=1:T,w=1:W], vREG[y,t,w] <= EP[:eRegReq][t,w])

        @constraint(EP, cRSV_MAX[y in RSV,t=1:T,w=1:W], vRSV[y,t,w] <= EP[:eRsvReq][t,w])


        ### FUEL!!!!
        println("Fuel Module")
        G = rhinputs["G"]
        THERM_COMMIT = rhinputs["THERM_COMMIT"]
        HAS_FUEL = rhinputs["HAS_FUEL"]
        MULTI_FUELS = rhinputs["MULTI_FUELS"]
        SINGLE_FUEL = rhinputs["SINGLE_FUEL"]

        fuels = rhinputs["fuels"]
        fuel_costs = rhinputs["fuel_costs"]
        omega = rhinputs["omega"]

        NUM_FUEL = length(fuels)

        # variables
        # create variable for fuel consumption for output
        # for resources that only use a single fuel
        @variable(EP, vFuel[y in SINGLE_FUEL, t = 1:T, w=1:W]>=0)
        @variable(EP, vStartFuel[y in SINGLE_FUEL, t = 1:T, w=1:W]>=0)

        # non-anticipativity for fuel variables
        @variable(EP, hnFuel[y in SINGLE_FUEL, t=[1]] >= 0)
        @variable(EP, hnStartFuel[y in SINGLE_FUEL, t=[1]] >= 0)

        # nonanticp constraint
        @constraint(EP, FUEL_nonantic[y in SINGLE_FUEL, t=[1], w=1:W], vFuel[y,t,w] == hnFuel[y,t])
        @constraint(EP, StartFuel_nonantic[y in SINGLE_FUEL, t=[1], w=1:W], vStartFuel[y,t,w] == hnStartFuel[y,t])

        @expression(EP, eStartFuel[y in 1:G, t = 1:T, w=1:W],
            if y in THERM_COMMIT
                (cap_size(gen[y]) * EP[:vSTART][y, t, w] *
                start_fuel_mmbtu_per_mw(gen[y]))
            else
                0
            end)

        # time-series fuel consumption by plant 
        @expression(EP, ePlantFuel_generation[y in 1:G, t = 1:T, w=1:W],
            if y in SINGLE_FUEL   # for single fuel plants
                EP[:vFuel][y, t, w]
            else # for multi fuel plants
                sum(EP[:vMulFuels][y, i, t, w] for i in 1:max_fuels)
            end)
        @expression(EP, ePlantFuel_start[y in 1:G, t = 1:T, w=1:W],
            if y in SINGLE_FUEL   # for single fuel plants
                EP[:vStartFuel][y, t, w]
            else # for multi fuel plants
                sum(EP[:vMulStartFuels][y, i, t, w] for i in 1:max_fuels)
            end)

        @expression(EP, eCFuelStart[y in 1:G, t = 1:T, w = 1:W],
            if y in SINGLE_FUEL
                (fuel_costs[fuel(gen[y])][t] * EP[:vStartFuel][y, t, w])
            else
                sum(EP[:eCFuelOut_multi_start][y, i, t, w] for i in 1:max_fuels)
            end)


        # plant level start-up fuel cost for output
        @expression(EP, ePlantCFuelStart[y in 1:G, w=1:W],
            sum(rhinputs["omega"][t] * EP[:eCFuelStart][y, t, w] for t in 1:T))

        # zonal level total fuel cost for output
        @expression(EP, eZonalCFuelStart[z = 1:Z, w=1:W],
            sum(EP[:ePlantCFuelStart][y,w] for y in resources_in_zone_by_rid(gen, z)))

        @expression(EP, eCFuelOut[y in 1:G, t = 1:T, w=1:W],
        if y in SINGLE_FUEL
            (fuel_costs[fuel(gen[y])][t] * EP[:vFuel][y, t,w])
        else
            sum(EP[:eCFuelOut_multi][y, i, t,w] for i in 1:max_fuels)
        end)

        # plant level start-up fuel cost for output
        @expression(EP, ePlantCFuelOut[y in 1:G, w=1:W],
            sum(rhinputs["omega"][t] * EP[:eCFuelOut][y, t, w] for t in 1:T))

        # zonal level total fuel cost for output
        @expression(EP, eZonalCFuelOut[z = 1:Z, w=1:W],
            sum(EP[:ePlantCFuelOut][y,w] for y in resources_in_zone_by_rid(gen, z)))

        # system level total fuel cost for output
        @expression(EP, eTotalCFuelOut[w=1:W], sum(eZonalCFuelOut[z] for z in 1:Z))

        @expression(EP, eTotalCFuelStart[w=1:W], sum(eZonalCFuelStart[z] for z in 1:Z))

        EP[:eObj] += eTotalCFuelOut .+ eTotalCFuelStart

        @constraint(EP,
            cFuelCalculation_single[
                y in intersect(SINGLE_FUEL, setdiff(HAS_FUEL, THERM_COMMIT)),
                t = 1:T],
            EP[:vFuel][y, t, w] - EP[:vP][y, t, w] * heat_rate_mmbtu_per_mwh(gen[y]) ==0 )



        THERM_COMMIT_PWFU = rhinputs["THERM_COMMIT_PWFU"]

        @constraint(EP,
            FuelCalculationCommit_single[
                y in intersect(setdiff(THERM_COMMIT,
                        THERM_COMMIT_PWFU),
                    SINGLE_FUEL),
                t = 1:T, w=1:W],
            EP[:vFuel][y, t, w] - EP[:vP][y, t, w] * heat_rate_mmbtu_per_mwh(gen[y]) ==0)


        @constraint(EP, cStartFuel_single[y in intersect(THERM_COMMIT, SINGLE_FUEL), t = 1:T, w = 1:W],
            EP[:vStartFuel][y, t, w] -
            (cap_size(gen[y]) * EP[:vSTART][y, t, w] * heat_rate_mmbtu_per_mwh(gen[y])).==0)

        ### curtailable_variable_renewable
        println("Curtailable Variable Renewable Module")
        VRE = rhinputs["VRE"]

        VRE_POWER_OUT = intersect(VRE, ids_with_positive(gen, num_vre_bins))
        VRE_NO_POWER_OUT = setdiff(VRE, VRE_POWER_OUT)

        ## Power Balance Expressions ##

        @expression(EP, ePowerBalanceDisp[t=1:T, z=1:Z, w=1:W],
            sum(EP[:vP][y,t,w] for y in intersect(VRE, resources_in_zone_by_rid(gen, z))))



        EP[:ePowerBalance] += ePowerBalanceDisp

        # Constraints
        # For resource for which we are modeling hourly power output
        for y in VRE_POWER_OUT
            # Define the set of generator indices corresponding to the different sites (or bins) of a particular VRE technology (E.g. wind or solar) in a particular zone.
            # For example the wind resource in a particular region could be include three types of bins corresponding to different sites with unique interconnection, hourly capacity factor and maximim available capacity limits.
            VRE_BINS = intersect(resource_id.(gen[resource_id.(gen) .>= y]),
            resource_id.(gen[resource_id.(gen) .<= y + num_vre_bins(gen[y]) - 1]))

            # Maximum power generated per hour by renewable generators must be less than
            # sum of product of hourly capacity factor for each bin times its the bin installed capacity
            # Note: inequality constraint allows curtailment of output below maximum level.
            @constraint(EP, [t=1:T,w=1:W], EP[:vP][y,t,w] <= sum(rhinputs["pP_Max"][w][yy,t]*EP[:eTotalCap][yy] for yy in VRE_BINS))


        end


        ### storage.jl
        STOR_ALL = rhinputs["STOR_ALL"]

        println("Storage Module")
        ### investment_energy
        @expression(EP, eExistingCapEnergy[y in STOR_ALL], existing_cap_mwh(gen[y]))

        @expression(EP, eTotalCapEnergy[y in STOR_ALL], eExistingCapEnergy[y] + EP[:vZERO])

        # Max and min constraints on energy storage capacity built (as proportion to discharge power capacity)
        @constraint(EP,
            cMinCapEnergyDuration[y in STOR_ALL, w=1:W],
            EP[:eTotalCapEnergy][y]>=min_duration(gen[y]) * EP[:eTotalCap][y])
        @constraint(EP,
            cMaxCapEnergyDuration[y in STOR_ALL, w=1:W],
            EP[:eTotalCapEnergy][y]<=max_duration(gen[y]) * EP[:eTotalCap][y])


        ### storage_all.jl
        Reserves = setup["OperationalReserves"]
        STOR_SHORT_DURATION = rhinputs["STOR_SHORT_DURATION"]


        # Storage level of resource "y" at hour "t" [MWh] on zone "z" - unbounded
        @variable(EP, vS[y in STOR_ALL, t=1:T, w=1:W] >= 0);

        # Energy withdrawn from grid by resource "y" at hour "t" [MWh] on zone "z"
        @variable(EP, vCHARGE[y in STOR_ALL, t=1:T, w=1:W] >= 0);

        ## Here and Now Variables
        @variable(EP, hnS[y in STOR_ALL, t=[1]] >= 0);
        @variable(EP, hnCHARGE[y in STOR_ALL, t=[1]] >= 0);

        ## Nonanticipativity Constraints
        @constraint(EP, vS_nonantic[y in STOR_ALL, t=[1], w=1:W], vS[y,t,w] == hnS[y,t])
        @constraint(EP, vCHARGE_nonantic[y in STOR_ALL, t=[1], w=1:W], vCHARGE[y,t,w] == hnCHARGE[y,t])

        ### Expressions ###

        # Energy losses related to technologies (increase in effective demand)
        @expression(EP, eELOSS[y in STOR_ALL, w=1:W], sum(rhinputs["omega"][t]*EP[:vCHARGE][y,t,w] for t in 1:T) - sum(rhinputs["omega"][t]*EP[:vP][y,t,w] for t in 1:T))

        ## Objective Function Expressions ##

        #Variable costs of "charging" for technologies "y" during hour "t" in zone "z"
        @expression(EP, eCVar_in[y in STOR_ALL,t=1:T, w=1:W], rhinputs["omega"][t]*var_om_cost_per_mwh_in(gen[y])*vCHARGE[y,t,w])

        # Sum individual resource contributions to variable charging costs to get total variable charging costs
        @expression(EP, eTotalCVarInT[t=1:T, w=1:W], sum(eCVar_in[y,t,w] for y in STOR_ALL))
        @expression(EP, eTotalCVarIn[w=1:W], sum(eTotalCVarInT[t,w] for t in 1:T))
        EP[:eObj] += eTotalCVarIn

        ## Power Balance Expressions ##

        # Term to represent net dispatch from storage in any period
        @expression(EP, ePowerBalanceStor[t=1:T, z=1:Z, w=1:W],
            sum(EP[:vP][y,t,w]-EP[:vCHARGE][y,t,w] for y in intersect(resources_in_zone_by_rid(gen, z), STOR_ALL)))

        EP[:ePowerBalance] += ePowerBalanceStor

        ### Constraints ###

        @constraint(EP, Max_vCHARGE[y in STOR_ALL, t=1:T, w=1:W], vCHARGE[y,t,w] <= existing_cap_mw(gen[y]))
        
        # Maximum energy stored must be less than installed energy capacity
        @constraint(EP, 
            cSTOR_MaxEnergyVol[y in STOR_ALL, t in 1:T, w=1:W], EP[:vS][y,t,w] 
            <= EP[:eTotalCapEnergy][y])


        ### storage_all_reserves
        # intialize storage state of charge
        initial_vS = [y in STOR_ALL ? gen[y].existing_cap_mwh * 0.75 : 0.0 for y in 1:num_gen]

        # parameters
        STOR_REG_RSV = intersect(STOR_ALL, rhinputs["REG"], rhinputs["RSV"]) # Set of storage resources with both REG and RSV reserves

        STOR_REG = intersect(STOR_ALL, rhinputs["REG"]) # Set of storage resources with REG reserves
        STOR_RSV = intersect(STOR_ALL, rhinputs["RSV"]) # Set of storage resources with RSV reserves

        STOR_NO_RES = setdiff(STOR_ALL, STOR_REG, STOR_RSV) # Set of storage resources with no reserves

        STOR_REG_ONLY = setdiff(STOR_REG, STOR_RSV) # Set of storage resources only with REG reserves
        STOR_RSV_ONLY = setdiff(STOR_RSV, STOR_REG) # Set of storage resources only with RSV reserves

        vP = EP[:vP]
        vS = EP[:vS]
        vCHARGE = EP[:vCHARGE]
        vREG = EP[:vREG]
        vRSV = EP[:vRSV]
        vREG_charge = EP[:vREG_charge]
        vRSV_charge = EP[:vRSV_charge]
        vREG_discharge = EP[:vREG_discharge]
        vRSV_discharge = EP[:vRSV_discharge]

        eTotalCap = EP[:eTotalCap]
        eTotalCapEnergy = EP[:eTotalCapEnergy]

        # Maximum storage contribution to reserves is a specified fraction of installed capacity
        @constraint(EP, cSTOR_MaxFreqReg[y in STOR_REG, t in 1:T, w=1:W], vREG[y, t, w]<=reg_max(gen[y]) * eTotalCap[y])
        @constraint(EP, cSTOR_MaxReserves[y in STOR_RSV, t in 1:T, w=1:W], vRSV[y, t, w]<=rsv_max(gen[y]) * eTotalCap[y])

        # Maximum discharging rate and contribution to reserves down must be greater than zero
        # Note: when discharging, reducing discharge rate is contributing to downwards regulation as it drops net supply
        @constraint(EP, cSTOR_TotRegContribution[y in STOR_REG, t in 1:T, w in 1:W],
            vREG[y, t, w]==vREG_charge[y, t, w] + vREG_discharge[y, t, w])
        @constraint(EP, cSTOR_TotResContribution[y in STOR_REG_RSV, t=1:T, w=1:W], 
            EP[:vRSV][y,t,w] == EP[:vRSV_charge][y,t,w]+EP[:vRSV_discharge][y,t,w])

        # Maximum discharging rate and contribution to reserves up must be less than power rating
        @constraint(EP, cSTOR_MaxRegRsvByCap[y in STOR_REG_RSV, t=1:T, w=1:W], 
            EP[:vP][y,t,w]+EP[:vREG_discharge][y,t,w]+EP[:vRSV_discharge][y,t,w] <= EP[:eTotalCap][y])
        
        # Maximum charging rate plus contribution to reserves up must be greater than zero
        # Note: when charging, reducing charge rate is contributing to upwards reserve & regulation as it drops net demand
        # expr = extract_time_series_to_expression(vCHARGE, STOR_ALL)
        # add_similar_to_expression!(expr[STOR_REG, :], -vREG_charge[STOR_REG, :])
        # add_similar_to_expression!(expr[STOR_RSV, :], -vRSV_charge[STOR_RSV, :])
        # @constraint(EP, cSTOR_maxChargeRegRsv[y in STOR_ALL, t in 1:T, w in 1:W], expr[y, t, w]>=0)
        @constraint(EP, cSTOR_MaxChargeRegRsv[y in STOR_ALL, t in 1:T, w in 1:W], 
            EP[:vCHARGE][y,t,w]-EP[:vREG_charge][y,t,w]-EP[:vRSV_charge][y,t,w] >= 0 )
        
        # Maximum discharging rate and contribution to reserves down must be greater than zero
        # Note: when discharging, reducing discharge rate is contributing to downwards regulation as it drops net supply
        # @constraint(EP, [y in STOR_REG, t in 1:T], vP[y, t] - vREG_discharge[y, t]>=0)
        @constraint(EP, cSTOR_NonnegNetDischarge[y in STOR_REG_RSV, t=1:T, w=1:W], 
            EP[:vP][y,t,w]-EP[:vREG_discharge][y,t,w] >= 0)

        # Maximum charging rate plus contribution to regulation down must be less than available storage capacity
        # Reg charge Linking Constraint
        if r == 1
            @constraint(EP, cSTOR_MaxRegChargeLink[y in STOR_REG_RSV, t in START_SUBPERIODS, w=1:W], EP[:vCHARGE][y,t,w]+EP[:vREG_charge][y,t,w] <= 
                EP[:eTotalCapEnergy][y]-initial_vS[y])
        else r > 1
            @constraint(EP, cSTOR_MaxRegChargeLink[y in STOR_REG_RSV, t in START_SUBPERIODS, w=1:W], EP[:vCHARGE][y,t,w]+EP[:vREG_charge][y,t,w] <= 
                EP[:eTotalCapEnergy][y]-s_dp[y,r-1])
        end

        # Maximum charging rate plus contribution to regulation down must be less than available storage capacity
        # Reg charge Interior Constraint
        @constraint(EP, cSTOR_MaxRegChargeInt[y in STOR_REG_RSV, t in INTERIOR_SUBPERIODS, w=1:W], EP[:vCHARGE][y,t,w]+EP[:vREG_charge][y,t,w] <= 
            EP[:eTotalCapEnergy][y]-EP[:vS][y,t-1,w]) 


        # Maximum discharging rate and contribution to reserves up must be less than power rating
        # Reg Rsv Linking
        if r == 1
            @constraint(EP, cSTOR_MaxRegRsvLink[y in STOR_REG_RSV, t in START_SUBPERIODS, w=1:W], EP[:vP][y,t,w]+EP[:vREG_discharge][y,t,w] +
                EP[:vRSV_discharge][y,t,w] <= initial_vS[y] * efficiency_down(gen[y]))
        else r > 1
            @constraint(EP, cSTOR_MaxRegRsvLink[y in STOR_REG_RSV, t in START_SUBPERIODS, w=1:W], EP[:vP][y,t,w]+EP[:vREG_discharge][y,t,w] +
                EP[:vRSV_discharge][y,t,w] <= s_dp[y,r-1] * efficiency_down(gen[y]))
        end

        # Reg Rsv Interior
        @constraint(EP, cSTOR_MaxRegRsvInt[y in STOR_REG_RSV, t in INTERIOR_SUBPERIODS, w=1:W], EP[:vP][y,t,w]+EP[:vREG_discharge][y,t,w] + 
            EP[:vRSV_discharge][y,t,w] <= EP[:vS][y,t-1,w] * efficiency_down(gen[y])) #cSTOR_MaxRegRsvLink

        # SOC balance
        if r == 1 ### XXX This is where the wrap-around or the warm-start needs to be implemented
            @constraint(EP, cSTOR_SOCLink[y in STOR_ALL, t in START_SUBPERIODS, w = 1:W], EP[:vS][y,t,w] ==
            initial_vS[y]-(1/ efficiency_down(gen[y]) *EP[:vP][y,t,w])
                +(efficiency_up(gen[y]) *EP[:vCHARGE][y,t,w])- (self_discharge(gen[y]) * initial_vS[y]))
        else r > 1
            @constraint(EP, cSTOR_SOCLink[y in STOR_ALL, t in START_SUBPERIODS, w = 1:W], EP[:vS][y,t,w] ==
                s_dp[y,r-1]-(1 / efficiency_down(gen[y]) * EP[:vP][y,t,w])
                +(efficiency_up(gen[y]) *EP[:vCHARGE][y,t,w])-(self_discharge(gen[y]) * s_dp[y,r-1]))
        end

        # energy stored for the next hour
        @constraint(EP, cSTOR_SOCInt[y in STOR_ALL, t in INTERIOR_SUBPERIODS, w = 1:W], EP[:vS][y,t,w] ==
            EP[:vS][y,t-1,w]-(1/efficiency_down(gen[y]) * EP[:vP][y,t,w])+(efficiency_up(gen[y]) * EP[:vCHARGE][y,t,w]) 
                -( self_discharge(gen[y]) * EP[:vS][y,t-1,w]))

        println("Thermal Module")
        # thermal.jl
        THERM_COMMIT = rhinputs["THERM_COMMIT"]
        THERM_NO_COMMIT = rhinputs["THERM_NO_COMMIT"]
        THERM_ALL = rhinputs["THERM_ALL"]

        # thermal_commit.jl
        ### Expressions ###

        ## Power Balance Expressions ##
        @expression(EP, ePowerBalanceThermCommit[t=1:T, z=1:Z, w=1:W],
        sum(EP[:vP][y,t,w] for y in intersect(THERM_COMMIT, resources_in_zone_by_rid(gen,z))))

        EP[:ePowerBalance] += ePowerBalanceThermCommit

        ### Capacitated limits on unit commitment decision variables (Constraints #1-3)
        @constraints(EP, begin
            cTC_MaxCommitUnits[y in THERM_COMMIT, t=1:T, w=1:W], EP[:vCOMMIT][y,t,w] <= EP[:eTotalCap][y]/cap_size(gen[y]) #cTC-MaxCommitUnits
            cTC_MaxStartupUnits[y in THERM_COMMIT, t=1:T, w=1:W], EP[:vSTART][y,t,w] <= EP[:eTotalCap][y]/cap_size(gen[y]) #cTC-MaxStartupUnits
            cTC_MaxShutdownUnits[y in THERM_COMMIT, t=1:T, w=1:W], EP[:vSHUT][y,t,w] <= EP[:eTotalCap][y]/cap_size(gen[y]) #cTC-MaxShutdownUnits
        end)

        
        ### cTC
        # Define important parameters for new constraints
        Up_Time = zeros(Int, size(THERM_COMMIT, 1))
        Up_Time[THERM_COMMIT] .= Int.(floor.(up_time.(gen[THERM_COMMIT])))

        Down_Time = zeros(Int, size(THERM_COMMIT,1))
        Down_Time[THERM_COMMIT] .= Int.(floor.(down_time.(gen[THERM_COMMIT])))

        p = policy_model_length # XXX might need to be changed to include warm start # Maybe Tmax?
        T = policy_model_length  # rhindex["T"]

        # if r = 1 # read from CEM expansion results at hoursbefore times
        @constraint(EP, cTC_MinDownTime_Link[y in THERM_COMMIT, t in 1:(Down_Time[y]-1), w = 1:W], 
            EP[:eTotalCap][y]/ cap_size(gen[y]) - EP[:vCOMMIT][y,t,w] >= 
            sum(cem_shut[h,gen[y].resource] for h in lkad_hoursbefore(Tend,r+t-1,1:Down_Time[y]) if h > r)
            + sum(shut_dp[y,h] for h in lkad_hoursbefore(Tend,r+t-1,1:Down_Time[y]) if h < r) 
            + sum(EP[:vSHUT][y,rt,w] for rt in 1:t)
        )
        @constraint(EP, cTC_MinUpTime_Link[y in THERM_COMMIT, t in 1:(Up_Time[y]-1), w = 1:W], 
            EP[:vCOMMIT][y,t,w] >= 
            sum(cem_start[h,gen[y].resource] for h in lkad_hoursbefore(Tend,r+t-1,1:Up_Time[y]) if h > r)
            + sum(start_dp[y,h] for h in lkad_hoursbefore(Tend, r+t-1, 1:Up_Time[y]) if h < r)
            + sum(EP[:vSTART][y,rt,w] for rt in 1:t)
        )

        # add constraints beyond 
        @constraint(EP, cTC_MinDownTime_Interior[y in THERM_COMMIT, t in Down_Time[y]:T, w = 1:W],
            EP[:eTotalCap][y]/ cap_size(gen[y]) - EP[:vCOMMIT][y,t,w] >= sum(EP[:vSHUT][y, lkad_hoursbefore(p, t, 0:(Down_Time[y] - 1)), w])
        )

        @constraint(EP, cTC_MinUpTime_Interior[y in THERM_COMMIT, t in Up_Time[y]:T, w = 1:W],
            EP[:vCOMMIT][y,t,w] >= sum(EP[:vSTART][y, lkad_hoursbefore(p, t, 0:(Up_Time[y] - 1)), w])
        ) 

        # initialize number of units that are started / on
        num_starting_units = [y in THERM_COMMIT ? gen[y].existing_cap_mw / gen[y].cap_size * 0.75 : 0.0 for y in 1:num_gen]

        ### Update Thermal Single Intertemp Constraints
        ## cTC_BalCommitUnits
        @constraint(EP, cTC_CommitUnitsInt[y in THERM_COMMIT, t in INTERIOR_SUBPERIODS, w=1:W], 
            EP[:vCOMMIT][y,t,w] == EP[:vCOMMIT][y,t-1,w] + EP[:vSTART][y,t,w] - EP[:vSHUT][y,t,w])
        ## cTC_LinkCommitUnits
        if r == 1 ### XXX This is where the wrap-around or the warm-start needs to be implemented
            @constraint(EP, cTC_CommitUnitsLink[y in THERM_COMMIT, t in START_SUBPERIODS, w=1:W], 
                EP[:vCOMMIT][y,t,w] == num_starting_units[y] + EP[:vSTART][y,t,w] - EP[:vSHUT][y,t,w])
        else r > 1
            # @constraint(EP, cTC_CommitUnitsLink[y in THERM_COMMIT], EP[:vCOMMIT][y,1] == EP[:vSTART][y,1] - EP[:vSHUT][y,1])
            @constraint(EP, cTC_CommitUnitsLink[y in THERM_COMMIT, t in START_SUBPERIODS, w=1:W], 
                EP[:vCOMMIT][y,t,w] == commit_dp[y,r-1] + EP[:vSTART][y,t,w] - EP[:vSHUT][y,t,w])
        end

        initial_vP = [y in THERM_COMMIT ? gen[y].existing_cap_mw * 0.75 : 0.0 for y in 1:num_gen]

        # XXX could be cleaned up similar to current GenX THERM_COMMIT.jl

        ## cTC_MaxRampDownInt
        @constraint(EP, cTC_MaxRampDownInt[y in THERM_COMMIT, t in INTERIOR_SUBPERIODS, w=1:W],
        EP[:vP][y,t-1,w]-EP[:vP][y,t,w] <= ramp_down_fraction(gen[y]) * cap_size(gen[y]) *(EP[:vCOMMIT][y,t,w]-EP[:vSTART][y,t,w])
            - min_power(gen[y]) * cap_size(gen[y]) *EP[:vSTART][y,t,w]
            + min(rhinputs["pP_Max"][w][y,t], max( min_power(gen[y]) , ramp_down_fraction(gen[y]) )) * cap_size(gen[y]) * EP[:vSHUT][y,t,w])
            
        ## cTC_MaxRampDownLink
        if r == 1 ### XXX This is where the wrap-around or the warm-start needs to be implemented
            @constraint(EP, cTC_MaxRampDownLink[y in THERM_COMMIT, t in START_SUBPERIODS, w=1:W],
            initial_vP[y] - EP[:vP][y,t,w] <= ramp_down_fraction(gen[y]) * cap_size(gen[y]) *(EP[:vCOMMIT][y,t,w]-EP[:vSTART][y,t,w])
                - min_power(gen[y]) * cap_size(gen[y]) *EP[:vSTART][y,t,w]
                + min(rhinputs["pP_Max"][w][y,t],max(min_power(gen[y]), ramp_down_fraction(gen[y]) )) 
                * cap_size(gen[y]) * EP[:vSHUT][y,t,w])
        else r > 1
            @constraint(EP,cTC_MaxRampDownLink[y in THERM_COMMIT, t in START_SUBPERIODS,w=1:W],
            pgen_dp[y, r-1] - EP[:vP][y,t,w] <= ramp_down_fraction(gen[y]) * cap_size(gen[y]) *(EP[:vCOMMIT][y,t,w]-EP[:vSTART][y,t,w])
            - min_power(gen[y]) * cap_size(gen[y]) *EP[:vSTART][y,t,w]
            + min(rhinputs["pP_Max"][w][y,t],max(min_power(gen[y]), ramp_down_fraction(gen[y]) )) 
            * cap_size(gen[y]) *EP[:vSHUT][y,t,w])
        end

        ## cTC_MaxRampUpInt
        @constraint(EP,cTC_MaxRampUpInt[y in THERM_COMMIT, t in INTERIOR_SUBPERIODS,w=1:W],
            EP[:vP][y,t,w]-EP[:vP][y,t-1,w] <= ramp_up_fraction(gen[y]) * cap_size(gen[y]) * (EP[:vCOMMIT][y,t,w]-EP[:vSTART][y,t,w])
                + min(rhinputs["pP_Max"][w][y,t],max(min_power(gen[y]) , ramp_up_fraction(gen[y]) )) 
                * cap_size(gen[y]) * EP[:vSTART][y,t,w]
                - min_power(gen[y]) * cap_size(gen[y]) *EP[:vSHUT][y,t,w])
        ## cTC_MaxRampUpLink
        if r == 1 ### XXX This is where the wrap-around or the warm-start needs to be implemented
            @constraint(EP,cTC_MaxRampUpLink[y in THERM_COMMIT, t in START_SUBPERIODS,w=1:W],
            EP[:vP][y,t,w]- initial_vP[y] <= ramp_up_fraction(gen[y]) * cap_size(gen[y]) * (EP[:vCOMMIT][y,t,w]-EP[:vSTART][y,t,w])
            + min(rhinputs["pP_Max"][w][y,t],max(min_power(gen[y]) , ramp_up_fraction(gen[y]) )) 
            * cap_size(gen[y]) * EP[:vSTART][y,t,w]
            - min_power(gen[y]) * cap_size(gen[y]) *EP[:vSHUT][y,t,w])
        else r > 1
            @constraint(EP,cTC_MaxRampUpLink[y in THERM_COMMIT, t in START_SUBPERIODS,w=1:W],
            EP[:vP][y,t,w]- pgen_dp[y,r-1] <= ramp_up_fraction(gen[y]) * cap_size(gen[y]) * (EP[:vCOMMIT][y,t,w]-EP[:vSTART][y,t,w])
            + min(rhinputs["pP_Max"][w][y,t],max(min_power(gen[y]) , ramp_up_fraction(gen[y]) )) 
            * cap_size(gen[y]) * EP[:vSTART][y,t,w]
            - min_power(gen[y]) * cap_size(gen[y]) *EP[:vSHUT][y,t,w])
        end

        # stochastic_thermal_commit_reserves
        THERM_COMMIT_REG_RSV = intersect(THERM_COMMIT, rhinputs["REG"], rhinputs["RSV"])

        @constraint(EP, cMaxREGContrib[y in THERM_COMMIT_REG_RSV, t=1:T, w=1:W], 
            EP[:vREG][y,t,w] <= rhinputs["pP_Max"][w][y,t]* reg_max(gen[y]) * cap_size(gen[y]) *EP[:vCOMMIT][y,t,w])
        
        @constraint(EP, cMaxRSVContrib[y in THERM_COMMIT_REG_RSV, t=1:T, w=1:W], 
            EP[:vRSV][y,t,w] <= rhinputs["pP_Max"][w][y,t] * rsv_max(gen[y]) * cap_size(gen[y]) *EP[:vCOMMIT][y,t,w])

        @constraint(EP, cMinStablePower[y in THERM_COMMIT_REG_RSV, t=1:T, w=1:W], 
            EP[:vP][y,t,w]-EP[:vREG][y,t,w] >= min_power(gen[y]) * cap_size(gen[y]) *EP[:vCOMMIT][y,t,w])

        @constraint(EP, cMaxPowerWithRegRsv[y in THERM_COMMIT_REG_RSV, t=1:T, w=1:W], 
            EP[:vP][y,t,w]+EP[:vREG][y,t,w]+EP[:vRSV][y,t,w] <= rhinputs["pP_Max"][w][y,t] * cap_size(gen[y]) * EP[:vCOMMIT][y,t,w])
        
        #### Define the Objective ####
        EP = add_quadratic_regularization(EP, gen,W,T)
        ## assign probabilities to stochastic scenarios
        uniform_probs = 1 / W
        ## redefine objective 
        scalar_eObj = uniform_probs .* ones(1,W) *  EP[:eObj]
        # update the objective function expression
        @expression(EP, single_eObj, scalar_eObj[1])


        ## Define the objective function
        @objective(EP, Min, single_eObj)

        ## Power balance constraints
        # demand = generation + storage discharge - storage charge - demand deferral + deferred demand satisfaction - demand curtailment (NSE)
        #          + incoming power flows - outgoing power flows - flow losses - charge of heat storage + generation from NACC
        @constraint(EP, cPowerBalance[t=1:T, z=1:Z, w=1:W], EP[:ePowerBalance][t,z,w] == rhinputs["pD"][w][t,z]) 

        # PowerBalanceObj = sum(cPowerBalance[t=1:T, z=1:Z, w] for w in 1:W) # T x Z, 48 x 1


        # set_attribute(EP, "BarHomogeneous", 1)
        optimize!(EP)
        println("Debugging Solving Model")
        println("termination status : ", termination_status(EP))
        println("result count       : ", result_count(EP))
        println("primal status      : ", primal_status(EP))
        println("dual status        : ", dual_status(EP))
        if termination_status(EP) == MOI.INFEASIBLE
            println("Model did not solve to optimality")
            compute_conflict!(EP)
            iis_model, _ = copy_conflict(EP)
            print(iis_model)

        end
                ## Record pre-solver time
        presolver_time = time() - presolver_start_time
        # println("termination status : ", termination_status(EP))
        inputs["solve_time"] = presolver_time # Store the model solve time in inputs

        # println("Debugging Solving Model")
        # compute_conflict!(EP)
        # iis_model, _ = copy_conflict(EP)
        # print(iis_model)

        #=======================================================================
        WRITE DECISION VARIABLES TO DICTIONARIES
        ========================================================================#
        

        REG_RSV = intersect(inputs["REG"], inputs["RSV"]) # Set of storage resources with both REG and RSV reserves
        Z = inputs["Z"]


        if model_type == "pf"
            # generation / dispatch
            pgen_dp[:,:] = value.(EP[:vP][:,:,1])
            # non-served load
            nse_dp[:,:] = value.(EP[:vNSE][:,:,Z,1]) # 1 for 1 scenarios
            # unmet demand 
            unmet_rsv_dp[:,:] = value.(EP[:vUNMET_RSV][:,1])
            # regulation provided
            reg_dp[REG_RSV, :] = value.(EP[:vREG][:,:,1])
            # reserve provided
            rsv_dp[REG_RSV, :] = value.(EP[:vRSV][:,:,1])
            # shutdown unit
            shut_dp[THERM_COMMIT, :] = value.(EP[:vSHUT][:,:,1])
            # start unit
            start_dp[THERM_COMMIT, :] = value.(EP[:vSTART][:,:,1])
            # commit unit
            commit_dp[THERM_COMMIT, :] = value.(EP[:vCOMMIT][:,:,1])
            # state of charge
            s_dp[STOR_LIST, :] = value.(EP[:vS][:,:,1])
            # charge
            charge_dp[STOR_LIST, :] = value.(EP[:vCHARGE][:,:,1])
            # electricity price
            elec_prices[:,:] = transpose(dual.(EP[:cPowerBalance])[:,:,1]) #* ModelScalingFactor # convert $/GWh to $/MWh
            # regulation price
            reg_prices[:,:] = transpose(dual.(EP[:cReg])[:,:,1])
            # reserve price
            rsv_prices[:,:] = transpose(dual.(EP[:cRsvReq])[:,:,1])

            fuel_costs_dp[:,:] =  value.(EP[:eCFuelOut][:,1:Tend,1]) .* ModelScalingFactor^2

            if setup["UCommit"] >= 1 && !isempty(COMMIT)
                start_costs_loop = value.(EP[:eCStart][COMMIT,1:Tend,1]).data
                start_fuel_costs_loop = value.(EP[:eCFuelStart][COMMIT,1:Tend,1])
                start_costs_dp[COMMIT, :] .= (start_costs_loop .+ start_fuel_costs_loop) * ModelScalingFactor^2
            end


            energy_revs_dp[:,:] = pgen_dp .* elec_prices .* ModelScalingFactor^2
            var_om_costs_dp[:,:] = var_om_cost_per_gen.* pgen_dp .* ModelScalingFactor^2

            reg_revs_dp[:,:] =  reg_dp .* reg_prices .* ModelScalingFactor^2
            rsv_revs_dp[:,:] =  rsv_dp .* rsv_prices .* ModelScalingFactor^2

            charge_costs_dp[:,:] = charge_dp .* elec_prices .* ModelScalingFactor^2
        
            nse_cost[:,:] = rhinputs["pC_D_Curtail"] .* nse_dp .* ModelScalingFactor^2
            unmet_rsv_cost[:,:] = rhinputs["pC_Rsv_Penalty"] .* unmet_rsv_dp .* ModelScalingFactor^2

        elseif model_type == "dlac-p" || model_type == "dlac-i" || model_type == "slac"
            pgen_dp[:,r] = value.(EP[:vP][:,1,1])
            # non-served energy
            nse_dp[1,r] = value(EP[:vNSE][1,1,Z,1])
            # unmet demand 
            unmet_rsv_dp[1,r] = value(EP[:vUNMET_RSV][1,1])
            # regulation provided
            reg_dp[REG_RSV,r] = value.(EP[:vREG][REG_RSV,1,1])
            # reserve provided
            rsv_dp[REG_RSV,r] = value.(EP[:vRSV][REG_RSV,1,1])
            # shutdown unit
            shut_dp[THERM_COMMIT,r] = value.(EP[:vSHUT][THERM_COMMIT,1,1])
            # start unit
            start_dp[THERM_COMMIT,r] = value.(EP[:vSTART][THERM_COMMIT,1,1])
            # commit unit
            commit_dp[THERM_COMMIT,r] = value.(EP[:vCOMMIT][THERM_COMMIT,1,1])
            # state of charge
            s_dp[STOR_ALL,r] = value.(EP[:vS][STOR_ALL,1,1])
            # charge
            charge_dp[STOR_ALL,r] = value.(EP[:vCHARGE][STOR_ALL,1,1])
            # electricity price
            elec_prices[Z,r] = sum(dual.(EP[:cPowerBalance])[1,1,:]) #* ModelScalingFactor # convert $/GWh to $/MWh
            # regulation price
            reg_prices[Z,r] = sum(dual.(EP[:cReg])[1,:,:]) #* ModelScalingFactor
            # reserve price
            rsv_prices[Z,r] = sum(dual.(EP[:cRsvReq])[1,:,:]) #* ModelScalingFactor

            fuel_costs_dp[:,r] =  value.(EP[:eCFuelOut][:,1,1]) .* ModelScalingFactor^2

            if setup["UCommit"] >= 1 && !isempty(COMMIT)
                start_costs_loop = value.(EP[:eCStart][COMMIT,1,1]).data
                start_fuel_costs_loop = value.(EP[:eCFuelStart][COMMIT,1,1])
                start_costs_dp[COMMIT,r] .= (start_costs_loop .+ start_fuel_costs_loop) * ModelScalingFactor^2
            end

            energy_revs_dp[:,r] = pgen_dp[:,r] .* elec_prices[:,r] .* ModelScalingFactor^2
            var_om_costs_dp[:,r] = var_om_cost_per_gen[:] .* pgen_dp[:,r] .* ModelScalingFactor^2

            reg_revs_dp[:,r] = reg_prices[r] .* reg_dp[:,r] .* ModelScalingFactor^2
            rsv_revs_dp[:,r] = rsv_prices[r] .* rsv_dp[:,r] .* ModelScalingFactor^2

            charge_costs_dp[:,r] = charge_dp[:,r] .* elec_prices[:,r] .* ModelScalingFactor^2
        
            nse_cost[:,r] = inputs["pC_D_Curtail"][1] * nse_dp[:,r] .* ModelScalingFactor^2
            unmet_rsv_cost[:,r] = inputs["pC_Rsv_Penalty"] * unmet_rsv_dp[:,r] .* ModelScalingFactor^2

            # save duals on discharge constraints
            # IF STOR_ALL is non-empty then save duals
            if !isempty(STOR_ALL)
                max_discharge_const_duals[r] = dual.(EP[:Max_vP])[STOR_ALL[],:,:]
                max_charge_const_duals[r] = Matrix(dual.(EP[:cSTOR_MaxRegRsvLink])[STOR_ALL[],:,:]) #length(STOR_ALL)
                soc_link_duals[r] = Matrix(dual.(EP[:cSTOR_SOCLink])[STOR_ALL[],:,:])
                soc_int_duals[r] = Matrix(dual.(EP[:cSTOR_SOCInt])[STOR_ALL[],:,:])
            end

        else
            println("Model not recognized")
        end

        prices_scen_array[r] = dual.(EP[:cPowerBalance])[:,:,:] .* ModelScalingFactor
        
    end # End of for r in R loop

    # Calculate financial results
    fixed_om_costs_vec = [fixed_om_cost_per_mwyr(gen[y]) * existing_cap_mw(gen[y]) * ModelScalingFactor^2 for y in 1:num_gen]
    fixed_om_costs = reshape(fixed_om_costs_vec, (num_gen, 1))

    total_energy_revs_dp = sum(energy_revs_dp, dims=2)
    total_reg_revs_dp = sum(reg_revs_dp, dims=2)
    total_rsv_revs_dp = sum(rsv_revs_dp, dims=2)

    rev_per_gen = total_energy_revs_dp + total_reg_revs_dp + total_rsv_revs_dp

    total_var_om_costs_dp = sum(var_om_costs_dp, dims=2)
    total_fuel_costs_dp = sum(fuel_costs_dp, dims=2)
    total_start_costs_dp = sum(start_costs_dp, dims=2)
    total_charge_costs_dp = sum(charge_costs_dp, dims=2)

    cost_per_gen = total_var_om_costs_dp + total_fuel_costs_dp + total_start_costs_dp + total_charge_costs_dp + fixed_om_costs

    # Calculate operating profit per generator
    operating_profit_per_gen = sum(energy_revs_dp, dims=2) + sum(reg_revs_dp, dims=2) +
                            sum(rsv_revs_dp, dims=2) - sum(var_om_costs_dp, dims=2) -
                            sum(fuel_costs_dp, dims=2) - sum(start_costs_dp, dims=2) -
                            sum(charge_costs_dp, dims=2) - fixed_om_costs

    total_welfare = sum(operating_profit_per_gen) - sum(nse_cost) - sum(unmet_rsv_cost)

    # Calculate investment costs
    storage_durations = [y in STOR_ALL ? gen[y].max_duration : 0.0 for y in 1:num_gen]

    invest_costs_perMW_yr = inv_cost_per_mwyr_per_gen .* ModelScalingFactor
    invest_costs_perMWhour_yr = inv_cost_per_mwhyr_per_gen .* ModelScalingFactor

    total_inv_costs_MW_yr = inv_cost_per_mwyr_per_gen .* existing_cap_mw_per_gen * ModelScalingFactor^2
    total_inv_costs_MWhour_yr = inv_cost_per_mwhyr_per_gen .* existing_cap_mwh_per_gen * ModelScalingFactor^2

    operating_profit_per_gen_vec = operating_profit_per_gen[:]
    total_inv_costs_MW_yr_vec = total_inv_costs_MW_yr[:]
    total_inv_costs_MWhour_yr_vec = total_inv_costs_MWhour_yr[:]

    total_inv_costs_MWhour_cost_in_MW_yr_vec = total_inv_costs_MWhour_yr_vec .* storage_durations

    total_both_inv_costs_MW_yr = total_inv_costs_MW_yr_vec + total_inv_costs_MWhour_cost_in_MW_yr_vec
    diff_org = operating_profit_per_gen_vec - total_inv_costs_MW_yr_vec - total_inv_costs_MWhour_yr_vec
    diff = diff_org - offset

    # Create results DataFrame
    results_df = DataFrame(generators = generator_name_per_gen,
                        Capacity_MW = existing_cap_mw_per_gen* ModelScalingFactor,
                        Capacity_MWh = existing_cap_mwh_per_gen * ModelScalingFactor,
                        Inv_cost_MW = total_inv_costs_MW_yr_vec,
                        Inv_cost_MWh = total_inv_costs_MWhour_yr_vec,
                        Fixed_OM_cost_MW = fixed_om_costs[:],
                        Fixed_OM_cost_MWh = zeros(num_gen,1)[:],
                        Var_OM_cost_out = total_var_om_costs_dp[:],
                        Fuel_cost = total_fuel_costs_dp[:],
                        Var_OM_cost_in = zeros(num_gen,1)[:],
                        StartCost = total_start_costs_dp[:],
                        Charge_cost = total_charge_costs_dp[:],
                        EnergyRevenue = total_energy_revs_dp[:],
                        OperatingRegulationRevenue = total_reg_revs_dp[:],
                        OperatingReserveRevenue = total_rsv_revs_dp[:],
                        Operating_Revenue = rev_per_gen[:],
                        Operating_Cost = cost_per_gen[:],
                        operating_profit_per_gen = operating_profit_per_gen_vec,
                        total_inv_costs = total_both_inv_costs_MW_yr,
                        diff_org = diff_org,
                        diff = diff)
end_time = time()-start_time
# Only write results to files if requested
if write_results
    # Create results folder if needed
    results_folder = joinpath(case, "results_" * model_type)
    if !isdir(results_folder)
        println("Creating results folder at: ", results_folder)
        mkpath(results_folder)
    end
    
    # Create dataframes of the results
    shut_df = DataFrame(shut_dp', resource_names)
    start_df = DataFrame(start_dp', resource_names)
    commit_df = DataFrame(commit_dp', resource_names)
    pgen_df = DataFrame(pgen_dp' * ModelScalingFactor, resource_names)
    s_df = DataFrame(s_dp' * ModelScalingFactor, resource_names)
    charge_df = DataFrame(charge_dp' * ModelScalingFactor, resource_names)
    rsv_df = DataFrame(rsv_dp' * ModelScalingFactor, resource_names)
    reg_df = DataFrame(reg_dp' * ModelScalingFactor, resource_names)
    elec_prices_df = DataFrame(elec_prices' * ModelScalingFactor, [string(Z)])
    reg_prices_df = DataFrame(reg_prices' * ModelScalingFactor, [string(Z)])
    rsv_prices_df = DataFrame(rsv_prices' * ModelScalingFactor, [string(Z)])
    nse_df = DataFrame(nse_dp' * ModelScalingFactor, [string(Z)])
    unmet_rsv_df = DataFrame(unmet_rsv_dp' * ModelScalingFactor, [string(Z)])

    # Write result dataframes to CSV files
    CSV.write(joinpath(results_folder, "unit_shut.csv"), shut_df)
    CSV.write(joinpath(results_folder, "unit_start.csv"), start_df)
    CSV.write(joinpath(results_folder, "unit_commit.csv"), commit_df)
    CSV.write(joinpath(results_folder, "unit_pgen.csv"), pgen_df)
    CSV.write(joinpath(results_folder, "unit_state_of_charge.csv"), s_df)
    CSV.write(joinpath(results_folder, "unit_charge.csv"), charge_df)
    CSV.write(joinpath(results_folder, "unit_rsv.csv"), rsv_df)
    CSV.write(joinpath(results_folder, "unit_reg.csv"), reg_df)
    CSV.write(joinpath(results_folder, "price_electricity.csv"), elec_prices_df)
    CSV.write(joinpath(results_folder, "prices_reg.csv"), reg_prices_df)
    CSV.write(joinpath(results_folder, "prices_rsv.csv"), rsv_prices_df)
    CSV.write(joinpath(results_folder, "zone_nse.csv"), nse_df)
    CSV.write(joinpath(results_folder, "zone_unmet_rsv.csv"), unmet_rsv_df)
    
    # Write revenue and cost outputs
    writedlm(joinpath(results_folder, "revenue_operating_profit_per_gen.csv"), operating_profit_per_gen', ',')
    writedlm(joinpath(results_folder, "revenue_total_welfare.csv"), total_welfare', ',')
    writedlm(joinpath(results_folder, "revenue_energy_revs_dp.csv"), energy_revs_dp', ',')
    writedlm(joinpath(results_folder, "revenue_reg_revs_dp.csv"), reg_revs_dp', ',')
    writedlm(joinpath(results_folder, "revenue_rsv_revs_dp.csv"), rsv_revs_dp', ',')
    writedlm(joinpath(results_folder, "revenue_var_om_costs_dp.csv"), var_om_costs_dp', ',')
    writedlm(joinpath(results_folder, "revenue_fuel_costs_dp.csv"), fuel_costs_dp', ',')
    writedlm(joinpath(results_folder, "revenue_start_costs_dp.csv"), start_costs_dp', ',')
    writedlm(joinpath(results_folder, "revenue_charge_costs_dp.csv"), charge_costs_dp', ',')
    writedlm(joinpath(results_folder, "revenue_nse_cost.csv"), nse_cost', ',')
    writedlm(joinpath(results_folder, "revenue_unmet_rsv_cost.csv"), unmet_rsv_cost', ',')
    
    # Write NetRevenue dataframe to CSV
    CSV.write(joinpath(results_folder, "NetRevenue.csv"), results_df, header=true)
    
    # For SLAC, save HDF5 files if needed
    if model_type == "slac"
        # Remove undefined elements from prices_scen_array if any
        prices_scen_array = filter(x -> x !== nothing, prices_scen_array)
        save_hdf5(results_folder, Tend, "prices_scen_array", prices_scen_array)
        
        if !isempty(STOR_ALL)
            # Save duals to HDF5
            save_hdf5(results_folder, Tend, "max_discharge_const_duals", max_discharge_const_duals)
            save_hdf5(results_folder, Tend, "max_charge_const_duals", max_charge_const_duals)
            save_hdf5(results_folder, Tend, "soc_link_duals", soc_link_duals)
            save_hdf5(results_folder, Tend, "soc_int_duals", soc_int_duals)
            
            # Calculate and write battery marginal information
            calculate_battery_marginals(results_folder, STOR_ALL, pgen_dp, s_dp, charge_dp,
                                       max_discharge_const_duals, max_charge_const_duals, Tend)
        end
    end
end

# Return the net profit and financial results
PMR = diff./(total_both_inv_costs_MW_yr + fixed_om_costs_vec)
return Dict(
    "net_profit" => diff,
    "PMR" => PMR*100,
    "total_welfare" => total_welfare[1],
    "operating_profit" => operating_profit_per_gen_vec,
    "inv_costs" => total_both_inv_costs_MW_yr,
    "energy_revenue" => total_energy_revs_dp[:],
    "capacity_mw" => existing_cap_mw_per_gen * ModelScalingFactor,
    "solve_time" => end_time,
    "results_df" => results_df
)
end

# Helper function for calculating battery marginals when using SLAC
function calculate_battery_marginals(results_folder, STOR_ALL, pgen_dp, s_dp, charge_dp, 
                                 max_discharge_const_duals, max_charge_const_duals, Tend)
    positive_counter = 0
    not_marginal_count = 0
    energy_marginal_count = 0
    discharge_marginal_count = 0
    charge_marginal_count = 0
    discharge_charge_marginal_count = 0
    is_marginal_count = 0

    eps = 1e-5

    max_capacity = maximum(pgen_dp[STOR_ALL[],:])

    union_marginal_count = count(t -> sum(max_discharge_const_duals[t][1,:]) < eps && pgen_dp[STOR_ALL[],t] > eps
                                && sum(max_charge_const_duals[t][1,:]) < eps && charge_dp[STOR_ALL[],t] > eps, 1:Tend)

    # Save the hours in which batteries appear to be binding
    t_indices = []

    for t in 1:Tend
        # Check if energy limited
        total_power_from_energy = s_dp[STOR_ALL[],t] / 1 - eps

        if total_power_from_energy < max_capacity
            if pgen_dp[STOR_ALL[],t] > eps && pgen_dp[STOR_ALL[],t] < total_power_from_energy
                energy_marginal_count += 1
                push!(t_indices, t)
            else
                not_marginal_count += 1
            end
        else
            if (sum(max_discharge_const_duals[t][1,:]) < eps && pgen_dp[STOR_ALL[],t] > eps && 
                sum(max_charge_const_duals[t][1,:]) < eps && charge_dp[STOR_ALL[],t] > eps)
                discharge_charge_marginal_count += 1
                push!(t_indices, t)
            elseif (sum(max_discharge_const_duals[t][1,:]) < eps && pgen_dp[STOR_ALL[],t] > eps && 
                sum(max_charge_const_duals[t][1,:]) > eps && charge_dp[STOR_ALL[],t] < eps)
                discharge_marginal_count += 1
                push!(t_indices, t)
            elseif (sum(max_discharge_const_duals[t][1,:]) > eps && pgen_dp[STOR_ALL[],t] < eps && 
                sum(max_charge_const_duals[t][1,:]) < eps && charge_dp[STOR_ALL[],t] > eps)
                charge_marginal_count += 1
                push!(t_indices, t)
            else
                not_marginal_count += 1
            end
        end
        is_marginal_count = energy_marginal_count + discharge_marginal_count + charge_marginal_count + discharge_charge_marginal_count
    end

    positive_charge_and_pgen_count = count(t -> charge_dp[STOR_ALL[],t] > eps && pgen_dp[STOR_ALL[],t] > eps, 1:Tend)
    energy_limited_count = count(x -> x <= maximum(pgen_dp[STOR_ALL[],:]), s_dp[STOR_ALL[],:] / 1)

    println("The number of times battery is marginal is: ", is_marginal_count)

    total_count = is_marginal_count + not_marginal_count
    textfile = joinpath(results_folder, "battery_marginals_counts.txt")

    # Create a text file with the results
    open(textfile, "w") do file
        write(file, "Total Count: $total_count\n")
        write(file, "Energy Limited Count: $energy_limited_count\n")
        write(file, "Energy Marginal Count: $energy_marginal_count\n")
        write(file, "Simultaneous Discharge and Charge Marginal Count: $discharge_charge_marginal_count\n")
        write(file, "Discharge Marginal Count: $discharge_marginal_count\n")
        write(file, "Charge Marginal Count: $charge_marginal_count\n")
        write(file, "Is Marginal Count: $is_marginal_count\n")
        write(file, "Not Marginal Count: $not_marginal_count\n")
    end
end

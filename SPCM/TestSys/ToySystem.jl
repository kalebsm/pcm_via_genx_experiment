using JuMP, Gurobi, LinearAlgebra, Plots, Statistics, Random, CSV, DataFrames
using Printf
# =============================================================================
# 1. SYSTEM DATA DEFINITION
# =============================================================================

struct Generator
    name::String
    fuel_cost::Float64      # $/MWh
    var_om_cost::Float64    # $/MWh  
    inv_cost::Float64       # $/MW/year
    fixed_om_cost::Float64  # $/MW/year
    max_capacity::Float64   # MW (for capacity expansion)
    min_stable_gen::Float64 # Minimum stable generation as fraction of capacity
    ramp_rate::Float64      # MW/h (as fraction of capacity)
    efficiency::Float64     # p.u.
    startup_cost::Float64   # $/startup
end

struct Battery
    name::String
    inv_cost_power::Float64    # $/MW/year (power capacity)
    inv_cost_energy::Float64   # $/MWh/year (energy capacity) 
    fixed_om_cost::Float64     # $/MW/year
    var_om_cost::Float64       # $/MWh
    max_power_capacity::Float64 # MW
    max_energy_capacity::Float64 # MWh
    efficiency_charge::Float64  # p.u.
    efficiency_discharge::Float64 # p.u.
    duration::Float64          # hours (energy/power ratio)
end

# Define the toy system
function create_toy_system()
    # 3 generators: Coal, Gas, Wind
    generators = [
        Generator("Coal", 25.0, 5.0, 50000.0, 30000.0, 1000.0, 0.4, 0.5, 0.38, 500.0),
        Generator("Gas", 45.0, 3.0, 80000.0, 15000.0, 800.0, 0.3, 0.8, 0.50, 100.0),
        Generator("Wind", 0.0, 8.0, 120000.0, 25000.0, 1200.0, 0.0, 1.0, 1.0, 0.0)
    ]
    
    # 1 battery storage
    battery = Battery("Battery", 150000.0, 200.0, 10000.0, 2.0, 500.0, 2000.0, 0.90, 0.90, 4.0)
    
    return generators, battery
end

# Create actual deterministic profiles and scenarios around them
function create_actual_and_scenarios()
    hours = 720  # 30 days
    days = 30
    
    Random.seed!(42)  # For reproducible results
    
    # Base daily demand pattern (MW) - actual deterministic profile
    base_daily_demand = [
        # Hours 1-24
        600, 580, 560, 550, 560, 580, 620, 680, 720, 760, 800, 820,
        850, 860, 870, 880, 900, 920, 950, 900, 850, 780, 720, 660
    ]
    
    # Create ACTUAL deterministic profiles
    actual_demand = Float64[]
    actual_wind = Float64[]
    
    for day in 1:days
        # Weekend scaling
        is_weekend = (day % 7 in [6, 0])
        weekend_factor = is_weekend ? 0.85 : 1.0
        
        # Seasonal variation
        seasonal_factor = 1.0 + 0.1 * sin(2π * day / 365)
        
        daily_demand = base_daily_demand .* weekend_factor .* seasonal_factor
        append!(actual_demand, daily_demand)
        
        # Actual wind pattern
        base_wind_pattern = [
            0.8, 0.85, 0.9, 0.85, 0.8, 0.7, 0.6, 0.4, 0.3, 0.25, 0.2, 0.15,
            0.1, 0.1, 0.15, 0.2, 0.25, 0.4, 0.6, 0.7, 0.75, 0.8, 0.85, 0.8
        ]
        
        # Add some deterministic weather patterns
        weather_factor = 0.8 + 0.4 * sin(2π * day / 7)  # Weekly weather pattern
        daily_wind = [max(min(cf * weather_factor, 1.0), 0.0) for cf in base_wind_pattern]
        append!(actual_wind, daily_wind)
    end
    
    # Create 3 scenarios with actual as the mean
    demand_scenarios = []
    wind_scenarios = []
    
    for scenario in 1:3
        scenario_demand = Float64[]
        scenario_wind = Float64[]
        
        for t in 1:hours
            # Add scenario-specific variation around actual
            demand_noise = 1.0 + 0.1 * (scenario - 2) + 0.05 * randn()  # ±10% base + noise
            wind_noise = 1.0 + 0.15 * (scenario - 2) + 0.1 * randn()    # ±15% base + noise
            
            push!(scenario_demand, max(actual_demand[t] * demand_noise, 0.1 * actual_demand[t]))
            push!(scenario_wind, max(min(actual_wind[t] * wind_noise, 1.0), 0.0))
        end
        
        push!(demand_scenarios, scenario_demand)
        push!(wind_scenarios, scenario_wind)
    end
    
    return actual_demand, actual_wind, demand_scenarios, wind_scenarios
end

# =============================================================================
# 2. CAPACITY EXPANSION MODEL (optimizes for actual deterministic)
# =============================================================================

function solve_capacity_expansion(generators, battery, actual_demand, actual_wind; output_dir="results")
    """
    Solve capacity expansion optimizing for the ACTUAL deterministic profiles
    """
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    
    T = length(actual_demand)
    G = length(generators)
    
    # Use representative time periods (every 6th hour)
    time_sample = 1:6:T
    T_sample = length(time_sample)
    
    println("Capacity expansion using $T_sample representative hours from $T total hours")
    
    # Decision variables
    @variable(model, capacity[1:G] >= 0)  
    @variable(model, battery_power_cap >= 0)  
    @variable(model, battery_energy_cap >= 0)
    
    # Operational variables for representative periods
    @variable(model, generation[1:G, 1:T_sample] >= 0)
    @variable(model, battery_charge[1:T_sample] >= 0)
    @variable(model, battery_discharge[1:T_sample] >= 0)
    @variable(model, battery_soc[1:T_sample] >= 0)
    @variable(model, load_shed[1:T_sample] >= 0)
    @variable(model, commitment[1:G, 1:T_sample], Bin)
    
    # Objective: Minimize annualized investment + operational costs
    time_weight = 6.0  # Each sample hour represents 6 hours
    
    investment_cost = sum(generators[g].inv_cost * capacity[g] for g in 1:G) + 
                     battery.inv_cost_power * battery_power_cap + 
                     battery.inv_cost_energy * battery_energy_cap
                     
    fixed_cost = sum(generators[g].fixed_om_cost * capacity[g] for g in 1:G) + 
                 battery.fixed_om_cost * battery_power_cap
                 
    operational_cost = sum(
        sum((generators[g].fuel_cost + generators[g].var_om_cost) * generation[g,t] +
            generators[g].startup_cost * commitment[g,t] for g in 1:G) +
        battery.var_om_cost * (battery_charge[t] + battery_discharge[t]) +
        10000 * load_shed[t]  # High penalty for unserved demand
        for t in 1:T_sample) * time_weight
    
    @objective(model, Min, investment_cost + fixed_cost + operational_cost)
    
    # Extract actual values for representative periods
    repr_demand = [actual_demand[time_sample[t]] for t in 1:T_sample]
    repr_wind = [actual_wind[time_sample[t]] for t in 1:T_sample]
    
    # Power balance
    @constraint(model, [t=1:T_sample],
        sum(generation[g,t] for g in 1:G) + battery_discharge[t] - 
        battery_charge[t] + load_shed[t] == repr_demand[t])
    
    # Generation limits and unit commitment
    for g in 1:G
        if generators[g].name == "Wind"
            @constraint(model, [t=1:T_sample], 
                generation[g,t] <= capacity[g] * repr_wind[t])
            @constraint(model, [t=1:T_sample], commitment[g,t] == 1)
        else
            @constraint(model, [t=1:T_sample], 
                generation[g,t] <= capacity[g] * commitment[g,t])
            @constraint(model, [t=1:T_sample], 
                generation[g,t] >= capacity[g] * generators[g].min_stable_gen * commitment[g,t])
        end
    end
    
    # Battery constraints
    @constraint(model, [t=1:T_sample], battery_charge[t] <= battery_power_cap)
    @constraint(model, [t=1:T_sample], battery_discharge[t] <= battery_power_cap)
    @constraint(model, [t=1:T_sample], battery_soc[t] <= battery_energy_cap)
    
    # Battery energy balance (simplified periodic)
    @constraint(model, [t=1:T_sample], battery_soc[t] <= 
        battery_energy_cap * 0.5 + battery.efficiency_charge * battery_charge[t] - 
        battery_discharge[t]/battery.efficiency_discharge)
    
    # Battery energy/power ratio
    @constraint(model, battery_energy_cap <= battery_power_cap * battery.duration)
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        result = Dict(
            "status" => "optimal",
            "capacity" => value.(capacity),
            "battery_power_cap" => value(battery_power_cap),
            "battery_energy_cap" => value(battery_energy_cap),
            "total_cost" => objective_value(model),
            "investment_cost" => value(investment_cost),
            "fixed_cost" => value(fixed_cost),
            "operational_cost" => value(operational_cost)
        )
        
        # Save capacity results
        mkpath(output_dir)
        capacity_df = DataFrame(
            Technology = [gen.name for gen in generators],
            Capacity_MW = value.(capacity),
            Investment_Cost = [generators[g].inv_cost * value(capacity[g]) for g in 1:G],
            Fixed_OM_Cost = [generators[g].fixed_om_cost * value(capacity[g]) for g in 1:G]
        )
        
        # Add battery row
        push!(capacity_df, ("Battery_Power", value(battery_power_cap), 
                           battery.inv_cost_power * value(battery_power_cap),
                           battery.fixed_om_cost * value(battery_power_cap)))
        push!(capacity_df, ("Battery_Energy", value(battery_energy_cap),
                           battery.inv_cost_energy * value(battery_energy_cap), 0.0))
        
        CSV.write(joinpath(output_dir, "capacity_expansion_results.csv"), capacity_df)
        
        return result
    else
        return Dict("status" => "infeasible", "termination_status" => termination_status(model))
    end
end

# =============================================================================
# 3. PERFECT FORESIGHT OPERATIONAL MODEL
# =============================================================================

function solve_perfect_foresight_operations(generators, battery, capacities, battery_power_cap, 
                                           battery_energy_cap, demand, wind_cf; output_dir="results")
    """
    Solve operations with perfect foresight for the entire time horizon
    """
    model = Model(Gurobi.Optimizer)
    set_silent(model)
    
    T = length(demand)
    G = length(generators)
    
    println("Solving perfect foresight operations for $T hours...")
    
    # Decision variables
    @variable(model, generation[1:G, 1:T] >= 0)
    @variable(model, battery_charge[1:T] >= 0)
    @variable(model, battery_discharge[1:T] >= 0)
    @variable(model, battery_soc[1:T] >= 0)
    @variable(model, load_shed[1:T] >= 0)
    @variable(model, commitment[1:G, 1:T], Bin)
    @variable(model, startup[1:G, 1:T], Bin)
    
    # Objective: Minimize operational costs
    @objective(model, Min, 
        sum(sum((generators[g].fuel_cost + generators[g].var_om_cost) * generation[g,t] +
                generators[g].startup_cost * startup[g,t] for g in 1:G) +
            battery.var_om_cost * (battery_charge[t] + battery_discharge[t]) +
            10000 * load_shed[t] for t in 1:T))
    
    # Power balance
    @constraint(model, power_balance[t=1:T],
        sum(generation[g,t] for g in 1:G) + battery_discharge[t] - 
        battery_charge[t] + load_shed[t] == demand[t])
    
    # Generation and commitment constraints
    for g in 1:G
        if generators[g].name == "Wind"
            @constraint(model, [t=1:T], generation[g,t] <= capacities[g] * wind_cf[t])
            @constraint(model, [t=1:T], commitment[g,t] == 1)
            @constraint(model, [t=1:T], startup[g,t] == 0)
        else
            @constraint(model, [t=1:T], generation[g,t] <= capacities[g] * commitment[g,t])
            @constraint(model, [t=1:T], 
                generation[g,t] >= capacities[g] * generators[g].min_stable_gen * commitment[g,t])
            
            # Startup logic
            @constraint(model, startup[g,1] >= commitment[g,1])
            @constraint(model, [t=2:T], startup[g,t] >= commitment[g,t] - commitment[g,t-1])
        end
    end
    
    # Battery constraints
    @constraint(model, [t=1:T], battery_charge[t] <= battery_power_cap)
    @constraint(model, [t=1:T], battery_discharge[t] <= battery_power_cap)
    @constraint(model, [t=1:T], battery_soc[t] <= battery_energy_cap)
    
    # Battery energy balance
    @constraint(model, battery_soc[1] == battery_energy_cap * 0.5 +
        battery.efficiency_charge * battery_charge[1] - battery_discharge[1]/battery.efficiency_discharge)
    @constraint(model, [t=2:T], battery_soc[t] == battery_soc[t-1] + 
        battery.efficiency_charge * battery_charge[t] - battery_discharge[t]/battery.efficiency_discharge)
    
    # End with same SOC as start (periodic constraint)
    @constraint(model, battery_soc[T] >= battery_energy_cap * 0.4)
    @constraint(model, battery_soc[T] <= battery_energy_cap * 0.6)
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        result = Dict(
            "status" => "optimal",
            "generation" => value.(generation),
            "battery_charge" => value.(battery_charge),
            "battery_discharge" => value.(battery_discharge),
            "battery_soc" => value.(battery_soc),
            "load_shed" => value.(load_shed),
            "commitment" => value.(commitment),
            "startup" => value.(startup),
            "total_cost" => objective_value(model),
            "prices" => dual.(power_balance)
        )
        
        # Save detailed operational results
        save_operational_results(result, generators, battery, "perfect_foresight", output_dir)
        
        return result
    else
        return Dict("status" => "infeasible", "termination_status" => termination_status(model))
    end
end

# =============================================================================
# 4. DLAC OPERATIONAL MODEL
# =============================================================================

function solve_dlac_i_operations(generators, battery, capacities, battery_power_cap, 
                                 battery_energy_cap, actual_demand, actual_wind, 
                                 demand_scenarios, wind_scenarios; lookahead_hours=24, output_dir="results")
    """
    Solve operations using DLAC-i (Deterministic Lookahead Commitment with Imperfect foresight)
    - Operates on ACTUAL demand/wind (realized values)
    - Uses MEAN of scenarios for forecasting in lookahead horizon
    """
    T = length(actual_demand)
    G = length(generators)
    S = length(demand_scenarios)
    
    # Compute mean forecasts from scenarios
    mean_demand_forecast = zeros(T)
    mean_wind_forecast = zeros(T)
    
    for t in 1:T
        mean_demand_forecast[t] = mean([demand_scenarios[s][t] for s in 1:S])
        mean_wind_forecast[t] = mean([wind_scenarios[s][t] for s in 1:S])
    end
    
    println("Solving DLAC-i operations with $lookahead_hours hour lookahead for $T hours...")
    println("  Using actual demand/wind for operations, mean forecast for lookahead")
    
    # Results storage
    generation_schedule = zeros(G, T)
    battery_charge_schedule = zeros(T)
    battery_discharge_schedule = zeros(T)
    battery_soc_schedule = zeros(T)
    load_shed_schedule = zeros(T)
    commitment_schedule = zeros(G, T)
    startup_schedule = zeros(G, T)
    prices = zeros(T)
    
    # State tracking
    current_soc = battery_energy_cap * 0.5
    previous_commitment = zeros(G)
    
    for t in 1:T
        if t % 100 == 0
            println("  Processing hour $t/$T")
        end
        
        model = Model(Gurobi.Optimizer)
        set_silent(model)
        
        # Determine lookahead horizon
        horizon_end = min(t + lookahead_hours - 1, T)
        horizon = t:horizon_end
        H = length(horizon)
        
        # Decision variables for the lookahead horizon
        @variable(model, gen[1:G, 1:H] >= 0)
        @variable(model, bat_charge[1:H] >= 0)
        @variable(model, bat_discharge[1:H] >= 0)
        @variable(model, bat_soc[1:H] >= 0)
        @variable(model, load_shed[1:H] >= 0)
        @variable(model, commit[1:G, 1:H], Bin)
        @variable(model, start[1:G, 1:H], Bin)
        
        # Objective: minimize cost over lookahead horizon
        @objective(model, Min, 
            sum(sum((generators[g].fuel_cost + generators[g].var_om_cost) * gen[g,τ] +
                    generators[g].startup_cost * start[g,τ] for g in 1:G) +
                battery.var_om_cost * (bat_charge[τ] + bat_discharge[τ]) +
                10000 * load_shed[τ] for τ in 1:H))
        
        # Power balance constraints
        @constraint(model, power_balance_constraint[τ=1:H],
            sum(gen[g,τ] for g in 1:G) + bat_discharge[τ] - 
            bat_charge[τ] + load_shed[τ] == 
            (τ == 1 ? actual_demand[t] : mean_demand_forecast[horizon[τ]]))  # Use actual for t=1, forecast for t>1
        
        # Generation and commitment constraints
        for g in 1:G
            if generators[g].name == "Wind"
                # Use actual wind for current period, mean forecast for future periods
                @constraint(model, [τ=1:H], gen[g,τ] <= capacities[g] * 
                    (τ == 1 ? actual_wind[t] : mean_wind_forecast[horizon[τ]]))
                @constraint(model, [τ=1:H], commit[g,τ] == 1)
                @constraint(model, [τ=1:H], start[g,τ] == 0)
            else
                @constraint(model, [τ=1:H], gen[g,τ] <= capacities[g] * commit[g,τ])
                @constraint(model, [τ=1:H], 
                    gen[g,τ] >= capacities[g] * generators[g].min_stable_gen * commit[g,τ])
                
                # Startup logic
                @constraint(model, start[g,1] >= commit[g,1] - previous_commitment[g])
                @constraint(model, [τ=2:H], start[g,τ] >= commit[g,τ] - commit[g,τ-1])
            end
        end
        
        # Battery constraints
        @constraint(model, [τ=1:H], bat_charge[τ] <= battery_power_cap)
        @constraint(model, [τ=1:H], bat_discharge[τ] <= battery_power_cap)
        @constraint(model, [τ=1:H], bat_soc[τ] <= battery_energy_cap)
        
        # Battery energy balance
        @constraint(model, bat_soc[1] == current_soc + 
            battery.efficiency_charge * bat_charge[1] - bat_discharge[1]/battery.efficiency_discharge)
        @constraint(model, [τ=2:H], bat_soc[τ] == bat_soc[τ-1] + 
            battery.efficiency_charge * bat_charge[τ] - bat_discharge[τ]/battery.efficiency_discharge)
        
        optimize!(model)
        
        if termination_status(model) == MOI.OPTIMAL
            # Store only first-period decisions (what actually happens)
            generation_schedule[:, t] = value.(gen[:, 1])
            battery_charge_schedule[t] = value(bat_charge[1])
            battery_discharge_schedule[t] = value(bat_discharge[1])
            battery_soc_schedule[t] = value(bat_soc[1])
            load_shed_schedule[t] = value(load_shed[1])
            commitment_schedule[:, t] = value.(commit[:, 1])
            startup_schedule[:, t] = value.(start[:, 1])
            
            # Extract price
            prices[t] = dual(power_balance_constraint[1])
            
            # Update state
            current_soc = value(bat_soc[1])
            previous_commitment = value.(commit[:, 1])
        else
            println("Warning: DLAC-i optimization failed at hour $t")
            load_shed_schedule[t] = actual_demand[t]  # Use actual demand for emergency
            prices[t] = 10000
        end
    end
    
    result = Dict(
        "status" => "optimal",
        "generation" => generation_schedule,
        "battery_charge" => battery_charge_schedule,
        "battery_discharge" => battery_discharge_schedule,
        "battery_soc" => battery_soc_schedule,
        "load_shed" => load_shed_schedule,
        "commitment" => commitment_schedule,
        "startup" => startup_schedule,
        "prices" => prices,
        "total_cost" => sum(sum((generators[g].fuel_cost + generators[g].var_om_cost) * 
                               generation_schedule[g,t] + generators[g].startup_cost * startup_schedule[g,t] 
                               for g in 1:G) + battery.var_om_cost * 
                               (battery_charge_schedule[t] + battery_discharge_schedule[t]) +
                               10000 * load_shed_schedule[t] for t in 1:T)
    )
    
    # Save detailed operational results
    save_operational_results(result, generators, battery, "dlac_i", output_dir)
    
    return result
end

# =============================================================================
# 5. RESULTS SAVING AND ANALYSIS
# =============================================================================

function save_operational_results(results, generators, battery, model_name, output_dir)
    """Save detailed operational results to CSV files"""
    
    mkpath(output_dir)
    T = length(results["prices"])
    G = length(generators)
    
    # Create main results DataFrame
    df = DataFrame(
        Hour = 1:T,
        Price = results["prices"],
        Load_Shed = results["load_shed"],
        Battery_Charge = results["battery_charge"],
        Battery_Discharge = results["battery_discharge"],
        Battery_SOC = results["battery_soc"]
    )
    
    # Add generation columns
    for g in 1:G
        df[!, "$(generators[g].name)_Generation"] = results["generation"][g, :]
        df[!, "$(generators[g].name)_Commitment"] = results["commitment"][g, :]
    end
    
    # Save main results
    CSV.write(joinpath(output_dir, "$(model_name)_operations.csv"), df)
    
    # Create summary statistics
    summary_df = DataFrame(
        Metric = String[],
        Value = Float64[]
    )
    
    push!(summary_df, ("Total_Operational_Cost", results["total_cost"]))
    push!(summary_df, ("Total_Load_Shed_MWh", sum(results["load_shed"])))
    push!(summary_df, ("Average_Price", mean(results["prices"])))
    push!(summary_df, ("Max_Price", maximum(results["prices"])))
    push!(summary_df, ("Battery_Total_Charge_MWh", sum(results["battery_charge"])))
    push!(summary_df, ("Battery_Total_Discharge_MWh", sum(results["battery_discharge"])))
    
    for g in 1:G
        total_gen = sum(results["generation"][g, :])
        push!(summary_df, ("$(generators[g].name)_Total_Generation_MWh", total_gen))
        push!(summary_df, ("$(generators[g].name)_Total_Startups", sum(results["startup"][g, :])))
    end
    
    CSV.write(joinpath(output_dir, "$(model_name)_summary.csv"), summary_df)
end

function calculate_profits_and_save(generators, battery, operational_results, capacities, 
                                   battery_power_cap, battery_energy_cap, model_name, output_dir)
    """Calculate and save detailed profit analysis"""
    
    G = length(generators)
    T = size(operational_results["generation"], 2)
    
    # Create profit DataFrame
    profit_df = DataFrame(
        Technology = String[],
        Capacity_MW = Float64[],
        Total_Generation_MWh = Float64[],
        Capacity_Factor = Float64[],
        Energy_Revenue = Float64[],
        Fuel_Costs = Float64[],
        VOM_Costs = Float64[],
        Startup_Costs = Float64[],
        Fixed_OM_Costs = Float64[],
        Investment_Costs = Float64[],
        Operating_Profit = Float64[],
        Net_Profit = Float64[],
        Profit_Margin_per_MW = Float64[]
    )
    
    # Generator profits
    for g in 1:G
        gen_name = generators[g].name
        capacity = capacities[g]
        total_gen = sum(operational_results["generation"][g, :])
        capacity_factor = capacity > 0 ? total_gen / (capacity * T) : 0.0
        
        # Revenues and costs
        energy_revenue = sum(operational_results["prices"][t] * operational_results["generation"][g,t] for t in 1:T)
        fuel_costs = sum(generators[g].fuel_cost * operational_results["generation"][g,t] for t in 1:T)
        vom_costs = sum(generators[g].var_om_cost * operational_results["generation"][g,t] for t in 1:T)
        startup_costs = sum(generators[g].startup_cost * operational_results["startup"][g,t] for t in 1:T)
        fixed_om_costs = generators[g].fixed_om_cost * capacity
        investment_costs = generators[g].inv_cost * capacity
        
        operating_profit = energy_revenue - fuel_costs - vom_costs - startup_costs - fixed_om_costs
        net_profit = operating_profit - investment_costs
        profit_margin = capacity > 0 ? net_profit / capacity : 0.0
        
        push!(profit_df, (gen_name, capacity, total_gen, capacity_factor, energy_revenue,
                         fuel_costs, vom_costs, startup_costs, fixed_om_costs, investment_costs,
                         operating_profit, net_profit, profit_margin))
    end
    
    # Battery profit
    battery_energy_revenue = sum(operational_results["prices"][t] * operational_results["battery_discharge"][t] for t in 1:T)
    battery_energy_costs = sum(operational_results["prices"][t] * operational_results["battery_charge"][t] for t in 1:T)
    battery_net_energy_revenue = battery_energy_revenue - battery_energy_costs
    
    battery_vom_costs = sum(battery.var_om_cost * (operational_results["battery_charge"][t] + 
                           operational_results["battery_discharge"][t]) for t in 1:T)
    battery_fixed_costs = battery.fixed_om_cost * battery_power_cap
    battery_investment_costs = battery.inv_cost_power * battery_power_cap + battery.inv_cost_energy * battery_energy_cap
    
    battery_operating_profit = battery_net_energy_revenue - battery_vom_costs - battery_fixed_costs
    battery_net_profit = battery_operating_profit - battery_investment_costs
    battery_profit_margin = battery_power_cap > 0 ? battery_net_profit / battery_power_cap : 0.0
    
    total_discharge = sum(operational_results["battery_discharge"])
    battery_capacity_factor = battery_power_cap > 0 ? total_discharge / (battery_power_cap * T) : 0.0
    
    push!(profit_df, ("Battery", battery_power_cap, total_discharge, battery_capacity_factor,
                     battery_energy_revenue, battery_energy_costs, battery_vom_costs, 0.0,
                     battery_fixed_costs, battery_investment_costs, battery_operating_profit,
                     battery_net_profit, battery_profit_margin))
    
    CSV.write(joinpath(output_dir, "$(model_name)_profits.csv"), profit_df)
    
    return profit_df
end

# =============================================================================
# 6. COMPLETE SOLVE FUNCTION
# =============================================================================

function solve_complete_system(output_dir="results")
    """
    Complete function that solves everything and generates output files
    Returns all results needed for fixed point iteration
    """
    
    println("="^80)
    println("COMPLETE 3-GENERATOR TOY SYSTEM SOLUTION")
    println("="^80)
    
    # Create system
    generators, battery = create_toy_system()
    
    # Generate actual deterministic profiles and scenarios
    println("\nGenerating actual deterministic profiles and 3 scenarios...")
    actual_demand, actual_wind, demand_scenarios, wind_scenarios = create_actual_and_scenarios()
    
    println("  - Actual demand: $(length(actual_demand)) hours")
    println("  - Generated 3 scenarios around actual as mean")
    println("  - Mean demand across scenarios: $(round(mean([mean(s) for s in demand_scenarios]), digits=1)) MW")
    println("  - Actual demand mean: $(round(mean(actual_demand), digits=1)) MW")
    
    # Save profiles
    mkpath(output_dir)
    profiles_df = DataFrame(
        Hour = 1:length(actual_demand),
        Actual_Demand = actual_demand,
        Actual_Wind_CF = actual_wind,
        Scenario1_Demand = demand_scenarios[1],
        Scenario1_Wind = wind_scenarios[1],
        Scenario2_Demand = demand_scenarios[2],
        Scenario2_Wind = wind_scenarios[2],
        Scenario3_Demand = demand_scenarios[3],
        Scenario3_Wind = wind_scenarios[3]
    )
    CSV.write(joinpath(output_dir, "demand_wind_profiles.csv"), profiles_df)
    
    # Step 1: Capacity Expansion (optimizing for actual deterministic)
    println("\n" * "="^60)
    println("STEP 1: CAPACITY EXPANSION (optimizing for actual)")
    println("="^60)
    
    cap_result = solve_capacity_expansion(generators, battery, actual_demand, actual_wind, output_dir=output_dir)
    
    if cap_result["status"] != "optimal"
        error("Capacity expansion failed!")
    end
    
    capacities = cap_result["capacity"]
    battery_power_cap = cap_result["battery_power_cap"] 
    battery_energy_cap = cap_result["battery_energy_cap"]
    
    println("Optimal Capacities (optimized for actual deterministic profiles):")
    for (i, gen) in enumerate(generators)
        println("  $(gen.name): $(round(capacities[i], digits=1)) MW")
    end
    println("  Battery Power: $(round(battery_power_cap, digits=1)) MW")
    println("  Battery Energy: $(round(battery_energy_cap, digits=1)) MWh")
    println("  Total Investment Cost: $(round(cap_result["investment_cost"], digits=0))")
    
    # Step 2: Perfect Foresight Operations (on actual deterministic)
    println("\n" * "="^60)
    println("STEP 2: PERFECT FORESIGHT OPERATIONS (actual deterministic)")
    println("="^60)
    
    pf_result = solve_perfect_foresight_operations(generators, battery, capacities, 
                                                  battery_power_cap, battery_energy_cap,
                                                  actual_demand, actual_wind, output_dir=output_dir)
    
    if pf_result["status"] != "optimal"
        error("Perfect foresight operations failed!")
    end
    
    println("Perfect Foresight Results:")
    println("  Total Operational Cost: $(round(pf_result["total_cost"], digits=0))")
    println("  Total Load Shed: $(round(sum(pf_result["load_shed"]), digits=1)) MWh")
    println("  Average Price: $(round(mean(pf_result["prices"]), digits=2))/MWh")
    
    # Calculate and save PF profits
    pf_profits = calculate_profits_and_save(generators, battery, pf_result, capacities,
                                           battery_power_cap, battery_energy_cap, 
                                           "perfect_foresight", output_dir)
    
    # Step 3: DLAC Operations (on actual deterministic)
    println("\n" * "="^60)
    println("STEP 3: DLAC OPERATIONS (actual deterministic)")
    println("="^60)
    
    dlac_result = solve_dlac_operations(generators, battery, capacities,
                                       battery_power_cap, battery_energy_cap,
                                       actual_demand, actual_wind, 
                                       lookahead_hours=24, output_dir=output_dir)
    
    if dlac_result["status"] != "optimal"
        error("DLAC operations failed!")
    end
    
    println("DLAC Results:")
    println("  Total Operational Cost: $(round(dlac_result["total_cost"], digits=0))")
    println("  Total Load Shed: $(round(sum(dlac_result["load_shed"]), digits=1)) MWh")
    println("  Average Price: $(round(mean(dlac_result["prices"]), digits=2))/MWh")
    
    # Calculate and save DLAC profits
    dlac_profits = calculate_profits_and_save(generators, battery, dlac_result, capacities,
                                             battery_power_cap, battery_energy_cap,
                                             "dlac", output_dir)
    
    # Step 4: Scenario Analysis (optional - showing differences across scenarios)
    println("\n" * "="^60)
    println("STEP 4: SCENARIO ANALYSIS (DLAC on 3 scenarios)")
    println("="^60)
    
    scenario_results = []
    for scenario in 1:3
        println("\nSolving DLAC for Scenario $scenario...")
        scenario_result = solve_dlac_operations(generators, battery, capacities,
                                               battery_power_cap, battery_energy_cap,
                                               demand_scenarios[scenario], wind_scenarios[scenario],
                                               lookahead_hours=24, output_dir=output_dir)
        
        # Save with scenario-specific naming
        save_operational_results(scenario_result, generators, battery, "dlac_scenario_$scenario", output_dir)
        scenario_profits = calculate_profits_and_save(generators, battery, scenario_result, capacities,
                                                     battery_power_cap, battery_energy_cap,
                                                     "dlac_scenario_$scenario", output_dir)
        
        push!(scenario_results, Dict("operations" => scenario_result, "profits" => scenario_profits))
        
        println("  Scenario $scenario - Cost: $(round(scenario_result["total_cost"], digits=0)), Load Shed: $(round(sum(scenario_result["load_shed"]), digits=1)) MWh")
    end
    
    # Step 5: Comparison Summary
    println("\n" * "="^60)
    println("STEP 5: COMPARISON SUMMARY")
    println("="^60)
    
    cost_diff = dlac_result["total_cost"] - pf_result["total_cost"]
    cost_pct = (cost_diff / pf_result["total_cost"]) * 100
    
    println("Perfect Foresight vs DLAC (on actual deterministic):")
    println("  PF Cost: $(round(pf_result["total_cost"], digits=0))")
    println("  DLAC Cost: $(round(dlac_result["total_cost"], digits=0))")
    println("  Difference: $(round(cost_diff, digits=0)) ($(round(cost_pct, digits=2))% increase)")
    
    println("\nProfit Margin Comparison (Net Profit / Capacity):")
    println("Technology | PF Margin | DLAC Margin | Difference")
    println("-"^50)
    for i in 1:length(generators)
        tech = generators[i].name
        pf_margin = pf_profits[i, :Profit_Margin_per_MW]
        dlac_margin = dlac_profits[i, :Profit_Margin_per_MW]
        diff = dlac_margin - pf_margin
        @printf("%-10s | %9.0f | %11.0f | %10.0f\n", tech, pf_margin, dlac_margin, diff)
    end
    
    # Battery comparison
    pf_bat_margin = pf_profits[end, :Profit_Margin_per_MW]
    dlac_bat_margin = dlac_profits[end, :Profit_Margin_per_MW]
    bat_diff = dlac_bat_margin - pf_bat_margin
    @printf("%-10s | %9.0f | %11.0f | %10.0f\n", "Battery", pf_bat_margin, dlac_bat_margin, bat_diff)
    
    # Create summary comparison file
    comparison_df = DataFrame(
        Technology = vcat([gen.name for gen in generators], ["Battery"]),
        PF_Profit_Margin = vcat([pf_profits[i, :Profit_Margin_per_MW] for i in 1:length(generators)+1]),
        DLAC_Profit_Margin = vcat([dlac_profits[i, :Profit_Margin_per_MW] for i in 1:length(generators)+1]),
        Margin_Difference = vcat([dlac_profits[i, :Profit_Margin_per_MW] - pf_profits[i, :Profit_Margin_per_MW] for i in 1:length(generators)+1])
    )
    CSV.write(joinpath(output_dir, "pf_vs_dlac_comparison.csv"), comparison_df)
    
    println("\nAll results saved to: $output_dir/")
    
    return Dict(
        "generators" => generators,
        "battery" => battery,
        "actual_demand" => actual_demand,
        "actual_wind" => actual_wind,
        "demand_scenarios" => demand_scenarios,
        "wind_scenarios" => wind_scenarios,
        "capacity_expansion" => cap_result,
        "capacities" => capacities,
        "battery_power_cap" => battery_power_cap,
        "battery_energy_cap" => battery_energy_cap,
        "perfect_foresight" => pf_result,
        "dlac" => dlac_result,
        "pf_profits" => pf_profits,
        "dlac_profits" => dlac_profits,
        "scenario_results" => scenario_results
    )
end

# =============================================================================
# 7. FIXED POINT ITERATION
# =============================================================================

function calculate_profit_margins(generators, battery, operational_results, capacities, battery_power_cap, battery_energy_cap)
    """Calculate profit margins (profit per MW - investment cost per MW) for fixed point iteration"""
    
    G = length(generators)
    T = size(operational_results["generation"], 2)
    
    profit_margins = zeros(G + 1)  # G generators + 1 battery
    
    # Generator profit margins
    for g in 1:G
        if capacities[g] > 0
            # Calculate profit per MW
            energy_revenue = sum(operational_results["prices"][t] * operational_results["generation"][g,t] for t in 1:T)
            fuel_costs = sum(generators[g].fuel_cost * operational_results["generation"][g,t] for t in 1:T)
            vom_costs = sum(generators[g].var_om_cost * operational_results["generation"][g,t] for t in 1:T)
            startup_costs = sum(generators[g].startup_cost * operational_results["startup"][g,t] for t in 1:T)
            fixed_om_costs = generators[g].fixed_om_cost * capacities[g]
            
            operating_profit = energy_revenue - fuel_costs - vom_costs - startup_costs - fixed_om_costs
            profit_per_mw = operating_profit / capacities[g]
            
            # Profit margin = profit per MW - investment cost per MW
            profit_margins[g] = profit_per_mw - generators[g].inv_cost
        else
            # For zero capacity, estimate profit margin with small test capacity
            profit_margins[g] = -generators[g].inv_cost  # Conservative estimate
        end
    end
    
    # Battery profit margin
    if battery_power_cap > 0
        battery_energy_revenue = sum(operational_results["prices"][t] * operational_results["battery_discharge"][t] for t in 1:T)
        battery_energy_costs = sum(operational_results["prices"][t] * operational_results["battery_charge"][t] for t in 1:T)
        battery_net_energy_revenue = battery_energy_revenue - battery_energy_costs
        
        battery_vom_costs = sum(battery.var_om_cost * (operational_results["battery_charge"][t] + 
                               operational_results["battery_discharge"][t]) for t in 1:T)
        battery_fixed_costs = battery.fixed_om_cost * battery_power_cap
        
        battery_operating_profit = battery_net_energy_revenue - battery_vom_costs - battery_fixed_costs
        battery_profit_per_mw = battery_operating_profit / battery_power_cap
        
        # For battery, use power investment cost
        profit_margins[G + 1] = battery_profit_per_mw - battery.inv_cost_power
    else
        profit_margins[G + 1] = -battery.inv_cost_power
    end
    
    return profit_margins
end

function fixed_point_iteration_pf_vs_dlac(system_data; 
                                          max_iterations=20, 
                                          tolerance=1e-3, 
                                          step_size=0.05,
                                          smoothing_beta=10.0,
                                          output_dir="results")
    """
    Fixed point iteration comparing Perfect Foresight vs DLAC policies
    """
    
    println("\n" * "="^80)
    println("FIXED POINT ITERATION: PERFECT FORESIGHT vs DLAC")
    println("="^80)
    
    generators = system_data["generators"]
    battery = system_data["battery"]
    actual_demand = system_data["actual_demand"]
    actual_wind = system_data["actual_wind"]
    
    G = length(generators)
    
    # Initialize with capacity expansion results
    current_capacities = copy(system_data["capacities"])
    current_battery_power = system_data["battery_power_cap"]
    current_battery_energy = current_battery_power * battery.duration
    
    println("Starting Fixed Point Iteration:")
    println("Initial capacities from capacity expansion:")
    for (i, gen) in enumerate(generators)
        println("  $(gen.name): $(round(current_capacities[i], digits=1)) MW")
    end
    println("  Battery: $(round(current_battery_power, digits=1)) MW")
    
    # Storage for tracking convergence
    iteration_results = []
    
    for iteration in 1:max_iterations
        println("\n--- Iteration $iteration ---")
        
        # Solve Perfect Foresight operations
        println("Solving Perfect Foresight operations...")
        pf_results = solve_perfect_foresight_operations(generators, battery, current_capacities,
                                                       current_battery_power, current_battery_energy,
                                                       actual_demand, actual_wind, output_dir=output_dir)
        
        if pf_results["status"] != "optimal"
            println("ERROR: Perfect foresight failed at iteration $iteration")
            break
        end
        
        # Solve DLAC operations
        println("Solving DLAC operations...")
        dlac_results = solve_dlac_operations(generators, battery, current_capacities,
                                            current_battery_power, current_battery_energy,
                                            actual_demand, actual_wind, 
                                            lookahead_hours=24, output_dir=output_dir)
        
        if dlac_results["status"] != "optimal"
            println("ERROR: DLAC failed at iteration $iteration")
            break
        end
        
        # Calculate profit margins for both policies
        pf_margins = calculate_profit_margins(generators, battery, pf_results, 
                                             current_capacities, current_battery_power, current_battery_energy)
        dlac_margins = calculate_profit_margins(generators, battery, dlac_results,
                                               current_capacities, current_battery_power, current_battery_energy)
        
        println("Current capacities: ", [round(c, digits=1) for c in current_capacities], ", Battery: $(round(current_battery_power, digits=1))")
        println("PF profit margins: ", [round(m, digits=0) for m in pf_margins])
        println("DLAC profit margins: ", [round(m, digits=0) for m in dlac_margins])
        
        # Check convergence for both policies
        max_pf_margin = maximum(abs.(pf_margins))
        max_dlac_margin = maximum(abs.(dlac_margins))
        max_overall_margin = max(max_pf_margin, max_dlac_margin)
        
        println("Max PF margin: $(round(max_pf_margin, digits=0))")
        println("Max DLAC margin: $(round(max_dlac_margin, digits=0))")
        println("Max overall margin: $(round(max_overall_margin, digits=0))")
        
        # Store iteration results
        iter_result = Dict(
            "iteration" => iteration,
            "capacities" => copy(current_capacities),
            "battery_power" => current_battery_power,
            "pf_margins" => copy(pf_margins),
            "dlac_margins" => copy(dlac_margins),
            "pf_cost" => pf_results["total_cost"],
            "dlac_cost" => dlac_results["total_cost"],
            "max_pf_margin" => max_pf_margin,
            "max_dlac_margin" => max_dlac_margin
        )
        push!(iteration_results, iter_result)
        
        # Check convergence (for demonstration, use DLAC margins as the equilibrium condition)
        if max_dlac_margin < tolerance
            println("\nFixed point converged for DLAC policy!")
            println("Final DLAC equilibrium capacities:")
            for (i, gen) in enumerate(generators)
                println("  $(gen.name): $(round(current_capacities[i], digits=1)) MW")
            end
            println("  Battery: $(round(current_battery_power, digits=1)) MW")
            
            # Save convergence results
            save_fixed_point_results(iteration_results, generators, output_dir)
            
            return Dict(
                "converged" => true,
                "converged_policy" => "DLAC",
                "final_capacities" => current_capacities,
                "final_battery_power" => current_battery_power,
                "final_pf_margins" => pf_margins,
                "final_dlac_margins" => dlac_margins,
                "iteration_results" => iteration_results,
                "iterations" => iteration
            )
        end
        
        # Update capacities using DLAC profit margins (since we want DLAC equilibrium)
        # Use softplus smoothing for stability
        println("Updating capacities based on DLAC profit margins...")
        
        for g in 1:G
            if current_capacities[g] > 0
                capacity_update = current_capacities[g] + step_size * current_capacities[g] * dlac_margins[g] / 1000.0  # Scale down margins
                # Apply softplus smoothing
                current_capacities[g] = log(1 + exp(smoothing_beta * capacity_update)) / smoothing_beta
                current_capacities[g] = min(current_capacities[g], generators[g].max_capacity)
            else
                # For zero capacity technologies
                if dlac_margins[g] > 0
                    current_capacities[g] = step_size * 10.0  # Start with small capacity
                end
            end
        end
        
        # Update battery capacity
        if current_battery_power > 0
            battery_update = current_battery_power + step_size * current_battery_power * dlac_margins[G + 1] / 1000.0
            current_battery_power = log(1 + exp(smoothing_beta * battery_update)) / smoothing_beta
            current_battery_power = min(current_battery_power, battery.max_power_capacity)
            current_battery_energy = current_battery_power * battery.duration
        else
            if dlac_margins[G + 1] > 0
                current_battery_power = step_size * 10.0
                current_battery_energy = current_battery_power * battery.duration
            end
        end
        
        println("Updated capacities: ", [round(c, digits=1) for c in current_capacities], ", Battery: $(round(current_battery_power, digits=1))")
    end
    
    println("\nFixed point iteration completed without convergence")
    save_fixed_point_results(iteration_results, generators, output_dir)
    
    return Dict(
        "converged" => false,
        "final_capacities" => current_capacities,
        "final_battery_power" => current_battery_power,
        "iteration_results" => iteration_results,
        "iterations" => max_iterations
    )
end

function save_fixed_point_results(iteration_results, generators, output_dir)
    """Save fixed point iteration results"""
    
    mkpath(output_dir)
    
    # Create iteration summary DataFrame
    iter_df = DataFrame(
        Iteration = [r["iteration"] for r in iteration_results],
        Max_PF_Margin = [r["max_pf_margin"] for r in iteration_results],
        Max_DLAC_Margin = [r["max_dlac_margin"] for r in iteration_results],
        PF_Cost = [r["pf_cost"] for r in iteration_results],
        DLAC_Cost = [r["dlac_cost"] for r in iteration_results],
        Cost_Difference = [r["dlac_cost"] - r["pf_cost"] for r in iteration_results]
    )
    
    # Add capacity columns
    G = length(generators)
    for g in 1:G
        iter_df[!, "$(generators[g].name)_Capacity"] = [r["capacities"][g] for r in iteration_results]
    end
    iter_df[!, "Battery_Power_Capacity"] = [r["battery_power"] for r in iteration_results]
    
    # Add margin columns
    for g in 1:G
        iter_df[!, "$(generators[g].name)_PF_Margin"] = [r["pf_margins"][g] for r in iteration_results]
        iter_df[!, "$(generators[g].name)_DLAC_Margin"] = [r["dlac_margins"][g] for r in iteration_results]
    end
    iter_df[!, "Battery_PF_Margin"] = [r["pf_margins"][G+1] for r in iteration_results]
    iter_df[!, "Battery_DLAC_Margin"] = [r["dlac_margins"][G+1] for r in iteration_results]
    
    CSV.write(joinpath(output_dir, "fixed_point_iteration_results.csv"), iter_df)
    
    println("Fixed point iteration results saved to: $(joinpath(output_dir, "fixed_point_iteration_results.csv"))")
end


function save_fixed_point_results_dlac_i(iteration_results, generators, output_dir)
    """Save fixed point iteration results for DLAC-i comparison"""
    
    mkpath(output_dir)
    
    # Create iteration summary DataFrame
    iter_df = DataFrame(
        Iteration = [r["iteration"] for r in iteration_results],
        Max_PF_Margin = [r["max_pf_margin"] for r in iteration_results],
        Max_DLAC_i_Margin = [r["max_dlac_i_margin"] for r in iteration_results],
        PF_Cost = [r["pf_cost"] for r in iteration_results],
        DLAC_i_Cost = [r["dlac_i_cost"] for r in iteration_results],
        Cost_Difference = [r["dlac_i_cost"] - r["pf_cost"] for r in iteration_results],
        Cost_Increase_Pct = [(r["dlac_i_cost"] - r["pf_cost"]) / r["pf_cost"] * 100 for r in iteration_results]
    )
    
    # Add capacity columns
    G = length(generators)
    for g in 1:G
        iter_df[!, "$(generators[g].name)_Capacity"] = [r["capacities"][g] for r in iteration_results]
    end
    iter_df[!, "Battery_Power_Capacity"] = [r["battery_power"] for r in iteration_results]
    
    # Add margin columns
    for g in 1:G
        iter_df[!, "$(generators[g].name)_PF_Margin"] = [r["pf_margins"][g] for r in iteration_results]
        iter_df[!, "$(generators[g].name)_DLAC_i_Margin"] = [r["dlac_i_margins"][g] for r in iteration_results]
    end
    iter_df[!, "Battery_PF_Margin"] = [r["pf_margins"][G+1] for r in iteration_results]
    iter_df[!, "Battery_DLAC_i_Margin"] = [r["dlac_i_margins"][G+1] for r in iteration_results]
    
    CSV.write(joinpath(output_dir, "fixed_point_iteration_pf_vs_dlac_i.csv"), iter_df)
    
    println("Fixed point iteration results saved to: $(joinpath(output_dir, "fixed_point_iteration_pf_vs_dlac_i.csv"))")
end

# =============================================================================
# 8. MAIN EXECUTION
# =============================================================================

function main()
    println("="^80)
    println("3-GENERATOR TOY SYSTEM: PF vs DLAC-i ANALYSIS WITH FIXED POINT ITERATION")
    println("="^80)
    
    # Step 1: Solve complete system and generate all output files
    println("Step 1: Solving complete system and generating output files...")
    system_results = solve_complete_system("results")
    
    # Step 2: Run fixed point iteration comparing PF and DLAC-i
    println("\nStep 2: Running fixed point iteration (PF vs DLAC-i)...")
    fixed_point_results = fixed_point_iteration_pf_vs_dlac_i(system_results, 
                                                             max_iterations=15,
                                                             tolerance=500.0,  # $500/MW tolerance
                                                             step_size=0.02,
                                                             output_dir="results")
    
    println("\n" * "="^80)
    println("ANALYSIS COMPLETE")
    println("="^80)
    println("All results saved to: results/")
    println("\nKey files:")
    println("  - capacity_expansion_results.csv (capacities optimized for actuals)")
    println("  - perfect_foresight_operations.csv (PF operating on actuals)")
    println("  - dlac_i_operations.csv (DLAC-i: actual operations, mean forecasts)")
    println("  - pf_vs_dlac_i_comparison.csv (profit margin comparison)")
    println("  - forecast_quality_analysis.csv (forecast errors)")
    println("  - fixed_point_iteration_pf_vs_dlac_i.csv (convergence tracking)")
    
    println("\nModel Structure:")
    println("  - Capacity Expansion: Optimized for actual deterministic profiles")
    println("  - Perfect Foresight: Operates on actual deterministic profiles")
    println("  - DLAC-i: Operates on actuals, forecasts using mean of 3 scenarios")
    println("  - Fixed Point: Finds equilibrium where DLAC-i policy breaks even")
    
    if fixed_point_results["converged"]
        println("\nFixed point iteration CONVERGED!")
        println("This shows the capacity equilibrium under DLAC-i operational policy")
        println("(accounting for forecast uncertainty in operational decisions)")
    else
        println("\nFixed point iteration did not converge within iteration limit")
        println("Check fixed_point_iteration_pf_vs_dlac_i.csv to analyze convergence behavior")
    end
    
    # Summary of key differences
    println("\n" * "="^60)
    println("KEY INSIGHTS:")
    println("="^60)
    
    # Show the impact of forecast uncertainty
    pf_cost = system_results["perfect_foresight"]["total_cost"]
    dlac_i_cost = system_results["dlac_i"]["total_cost"]
    cost_increase = ((dlac_i_cost - pf_cost) / pf_cost) * 100
    
    println("Cost of Forecast Uncertainty:")
    println("  Perfect Foresight Cost: $(round(pf_cost, digits=0))")
    println("  DLAC-i Cost: $(round(dlac_i_cost, digits=0))")
    println("  Cost Increase: $(round(cost_increase, digits=2))%")
    
    # Show forecast quality
    forecast_demand_error = mean(abs.([system_results["actual_demand"][t] - 
        mean([system_results["demand_scenarios"][s][t] for s in 1:3]) for t in 1:length(system_results["actual_demand"])]))
    forecast_wind_error = mean(abs.([system_results["actual_wind"][t] - 
        mean([system_results["wind_scenarios"][s][t] for s in 1:3]) for t in 1:length(system_results["actual_wind"])]))
    
    println("\nForecast Quality (Mean Absolute Error):")
    println("  Demand Forecast Error: $(round(forecast_demand_error, digits=1)) MW")
    println("  Wind Forecast Error: $(round(forecast_wind_error, digits=3))")
    
    return system_results, fixed_point_results
end

system_results = solve_complete_system("results")

# Run the complete analysis
# system_results, fixed_point_results = main()

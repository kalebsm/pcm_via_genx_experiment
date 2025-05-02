module CapacityEquilibrium
using SPCMviaGenX
using LinearAlgebra
using Statistics
using Distributions

export compute_equilibrium, analyze_equilibrium, run_equilibrium_analysis


"""
    compute_equilibrium(context, model_type; 
                        initial_capacities=Dict(), 
                        tolerance=1e-3, 
                        max_iterations=50, 
                        step_size = 0.005,
                        smoothing_param=10.0,
                        perturbation_size=0.01,
                        save_iterations=false)

Compute capacity equilibrium using fixed point iteration method.

Arguments:
- `context`: Context dictionary from initialize_policy_model
- `model_type`: String specifying the policy model type ("slac", "dlac-p", or "dlac-i")
- `initial_capacities`: Dictionary mapping generator indices to initial capacities
- `tolerance`: Convergence tolerance for capacity changes
- `max_iterations`: Maximum number of iterations
- `step_size`: Step size for capacity adjustments
- `smoothing_param`: Parameter β controlling the sharpness of the softplus approximation
- `perturbation_size`: Small value δ for estimating profits of zero-capacity technologies
- `save_iterations`: Whether to save results from all iterations

Returns a dictionary with results and convergence information.
"""
function compute_equilibrium(context, model_type; 
                            initial_capacities=Dict(), 
                            tolerance=1e-3, 
                            max_iterations=150, 
                            step_size=0.05,
                            smoothing_param=10.0,
                            perturbation_size=0.01,
                            save_iterations=false)
    

    # Get resource information from context
    inputs = context["inputs"]
    gen = inputs["RESOURCES"]
    num_gen = inputs["G"]
    ModelScalingFactor = context["ModelScalingFactor"]
    
    # Initialize capacities if not provided
    if isempty(initial_capacities)
        for y in 1:num_gen
            initial_capacities[y] = (gen[y].existing_cap_mw)
        end
    end
    current_capacities = [initial_capacities[y] for y in 1:num_gen]
    
    # For tracking convergence
    capacity_history = []
    profit_history = []
    max_pmr_history = []
    log_file = joinpath(context["case"], "equilibrium_qp_$(model_type).csv")
    resource_names = [gen[y].resource for y in 1:num_gen]
    # # Initialize log file with headers
    # open(log_file, "w") do f
    #     # Write header row
    #     headers = ["Iteration", "max_pmr", "time"]
        
    #     # Add columns for each generator's capacity
    #     for name in resource_names
    #         push!(headers, "$(name)_capacity_MW")
    #     end
        
    #     # Add columns for each generator's profit
    #     for name in resource_names
    #         push!(headers, "$(name)_pmr")
    #     end
    #     # Write the header row
    #     println(f, join(headers, ","))
    # end
    # Main iteration loop
    for iteration in 97:max_iterations
        start = time()
        println("Iteration $iteration of $max_iterations")
        # current_capacities = current_capacities .* 1.05
        # Run policy model with current capacities
        results = run_policy_model_new(context, model_type, current_capacities; write_results=false)
        # Extract profits and investment costs
        net_profit = results["net_profit"]  # Vector of profits minus investment costs
        capacities = results["capacity_mw"]./ModelScalingFactor  # Vector of capacities in MW
        pmr = results["PMR"]  # Vector of PMR values
        # Record history
        push!(capacity_history, copy(capacities))
        push!(profit_history, copy(pmr))
        
        # Compute next capacity values using the softplus-smoothed fixed point mapping
        capacity_diffs = zeros(num_gen)
        adjustment = current_capacities .+ step_size*(pmr/100)
        next_capacities = softplus.(adjustment, smoothing_param)
        max_pmr = maximum(abs.(pmr))
        if max_pmr < 10
            step_size = 0.01
            if max_pmr < 1
                step_size = 0.001
            end
        end
        push!(max_pmr_history, max_pmr)
        println("Maximum abs profit margin: $max_pmr MW")
        iter_time = time()-start
        if max_pmr/100 < tolerance
            println("Converged after $iteration iterations")
            
            # Run one final time to get results
            final_results = run_policy_model_new(context, model_type, next_capacities; write_results=true)
            
            # Return results
            return Dict(
                "converged" => true,
                "iterations" => iteration,
                "final_capacities" => next_capacities,
                "capacity_history" => capacity_history,
                "profit_history" => profit_history,
                "capacity_diff_history" => capacity_diff_history,
                "final_results" => final_results
            )
        end
         # Log current iteration data to CSV
         open(log_file, "a") do f
            row_data = [float(iteration), max_pmr, iter_time]
            
            # Add capacities
            for y in 1:num_gen
                push!(row_data, capacities[y])
            end
            
            # Add profit violations
            for y in 1:num_gen
                push!(row_data, pmr[y])
            end
            
            # Write the row
            println(f, join(row_data, ","))
        end
        # Update current capacities for next iteration
        current_capacities = next_capacities
    end
    
    # If we reach here, we didn't converge
    println("Failed to converge after $max_iterations iterations")
    
    # Run one final time to get results
    final_results = run_policy_model_new(context, model_type, current_capacities; write_results=false)
    
    # Return results
    return Dict(
        "converged" => false,
        "iterations" => max_iterations,
        "final_capacities" => current_capacities,
        "capacity_history" => capacity_history,
        "profit_history" => profit_history,
        "capacity_diff_history" => capacity_diff_history,
        "final_results" => final_results
    )
end

"""
    softplus(x, β=10.0)

Compute the softplus function as a smooth approximation to max(0, x).
The parameter β controls the sharpness of the approximation.
"""
function softplus(x, β=10.0)
    return (1.0 / β) * log(1.0 + exp(β * x))
end

"""
    analyze_equilibrium(result)

Analyze the equilibrium results and print summary statistics.
"""
function analyze_equilibrium(result)
    if result["converged"]
        println("Capacity equilibrium converged after $(result["iterations"]) iterations.")
    else
        println("Capacity equilibrium did not converge within $(result["iterations"]) iterations.")
    end
    
    # Get final capacities and results
    final_capacities = result["final_capacities"]
    final_results = result["final_results"]
    
    # Print capacity and profit summary
    println("\nFinal Capacity and Profit Summary:")
    println("Generator | Capacity (MW) | Profit (\$ million) | Inv. Cost (\$ million) | Net Profit (\$ million)")
    println("----------|--------------|-------------------|---------------------|--------------------")
    
    for (idx, gen_name) in enumerate(final_results["results_df"].generators)
        capacity = final_results["results_df"].Capacity_MW[idx]
        profit = final_results["results_df"].operating_profit_per_gen[idx] / 1e6  # Convert to millions
        inv_cost = final_results["results_df"].total_inv_costs[idx] / 1e6  # Convert to millions
        net_profit = final_results["results_df"].diff[idx] / 1e6  # Convert to millions
        
        printf("%9s | %12.2f | %19.2f | %21.2f | %20.2f\n", 
                gen_name, capacity, profit, inv_cost, net_profit)
    end
    
    # Print convergence metrics
    println("\nConvergence Metrics:")
    capacity_diff_history = result["capacity_diff_history"]
    max_diffs = [maximum(diff) for diff in capacity_diff_history]
    
    println("Initial max capacity change: $(max_diffs[1])")
    println("Final max capacity change: $(max_diffs[end])")
    
    # Return any additional analysis metrics
    return Dict(
        "convergence_rate" => max_diffs,
        "final_welfare" => final_results["total_welfare"]
    )
end

"""
    run_equilibrium_analysis(case, model_types; kwargs...)

Run equilibrium analysis for multiple policy models and compare results.

Arguments:
- `case`: Path to case directory
- `model_types`: Array of model types to analyze (e.g., ["slac", "dlac-p", "dlac-i"])
- `kwargs`: Additional arguments passed to compute_equilibrium

Returns a dictionary with results for each model type.
"""
function run_equilibrium_analysis(case, model_types; kwargs...)
    
    # Initialize the context once
    context = initialize_policy_model(case)
    
    # Run equilibrium analysis for each model type
    results = Dict()
    for model_type in model_types
        println("\n=== Running equilibrium analysis for $model_type ===")
        
        # Compute equilibrium
        eq_result = compute_equilibrium(context, model_type; kwargs...)
        
        # Analyze results
        analysis = analyze_equilibrium(eq_result)
        
        # Store results
        results[model_type] = Dict(
            "equilibrium" => eq_result,
            "analysis" => analysis
        )
    end
    
    # Compare results across model types
    compare_results(results, model_types)
    
    return results
end

"""
    compare_results(results, model_types)

Compare equilibrium results across different policy models.
"""
function compare_results(results, model_types)
    println("\n=== Comparison of Equilibrium Results Across Policy Models ===")
    
    # Compare welfare
    println("\nTotal Welfare (\$ million):")
    for model_type in model_types
        welfare = results[model_type]["analysis"]["final_welfare"] / 1e6  # Convert to millions
        println("$model_type: $welfare")
    end
    
    # Compare convergence
    println("\nNumber of Iterations to Convergence:")
    for model_type in model_types
        iterations = results[model_type]["equilibrium"]["iterations"]
        converged = results[model_type]["equilibrium"]["converged"]
        status = converged ? "converged" : "did not converge"
        println("$model_type: $iterations iterations ($status)")
    end
    
    # Compare capacity mix
    println("\nFinal Capacity Mix (MW):")
    
    # Get generator names from the first model
    first_model = model_types[1]
    generators = results[first_model]["equilibrium"]["final_results"]["results_df"].generators
    
    # Print header
    print("Generator | ")
    for model_type in model_types
        printf("%10s | ", model_type)
    end
    println()
    
    print("----------|")
    for _ in model_types
        print("-----------|")
    end
    println()
    
    # Print capacity for each generator and model
    for (idx, gen_name) in enumerate(generators)
        printf("%9s | ", gen_name)
        
        for model_type in model_types
            capacity = results[model_type]["equilibrium"]["final_results"]["results_df"].Capacity_MW[idx]
            printf("%10.2f | ", capacity)
        end
        println()
    end
end

export compute_equilibrium, analyze_equilibrium, run_equilibrium_analysis

end  # module
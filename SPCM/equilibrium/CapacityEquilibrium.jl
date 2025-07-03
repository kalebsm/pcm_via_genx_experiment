module CapacityEquilibrium
using SPCMviaGenX
using LinearAlgebra
using Statistics
using Distributions

export compute_equilibrium, analyze_equilibrium, run_equilibrium_analysis

"""
    softplus(x, β=10.0)

Compute the softplus function as a smooth approximation to max(0, x).
The parameter β controls the sharpness of the approximation.
"""
function softplus(x, β=10.0)
    return (1.0 / β) * log(1.0 + exp(β * x))
end


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
    tolerance=1e-2, 
    max_iterations=1000, 
    step_size=0.5,
    smoothing_param=10.0,
    perturbation_size=0.01,
    save_iterations=false,
    aa_memory=10,         # Anderson Acceleration memory size
    aa_regularization=1e-10,  # Regularization parameter for matrix inversion
    aa_mixing=0.6)       # Mixing parameter for Anderson acceleration

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
    log_file = joinpath(context["case"], "equilibrium_anderson_LARGE_$(model_type).csv")
    resource_names = [gen[y].resource for y in 1:num_gen]

    # Anderson Acceleration variables
    # We store x_k and g_k = F(x_k) where F is our fixed point operator
    aa_x_history = Vector{Vector{Float64}}()  # History of iterates
    aa_g_history = Vector{Vector{Float64}}()  # History of fixed point evaluations

    open(log_file, "w") do f
        # Write header row
        headers = ["Iteration", "max_pmr", "time", "sys_costs"]

        # Add columns for each generator's capacity
        for name in resource_names
            push!(headers, "$(name)_capacity_MW")
        end

        # Add columns for each generator's profit
        for name in resource_names
            push!(headers, "$(name)_pmr")
        end
        # Write the header row
        println(f, join(headers, ","))
    end

    for iteration in 0:max_iterations
        start = time()
        println("Iteration $iteration of $max_iterations")

        # Run policy model with current capacities
        results = run_policy_model_new(context, model_type, current_capacities; write_results=false)

        # Extract profits and investment costs
        net_profit = results["net_profit"]
        capacities = results["capacity_mw"]./ModelScalingFactor
        pmr = results["PMR"]
        sys_costs = results["sys_costs"]

        # Record history
        push!(capacity_history, copy(capacities))
        push!(profit_history, copy(pmr))

        # Adjust step size based on maximum PMR
        max_pmr = maximum(abs.(pmr))
        if max_pmr < 10
            # step_size = 0.005
            if max_pmr < 5
                step_size = 0.01
                if max_pmr < 2
                    step_size = 0.0005
                end
            end
        end

        # Compute the standard fixed-point mapping g(x_k) = F(x_k)
        # This is the result of our fixed-point operator F
        adjustment = current_capacities .+ step_size * (pmr / 100)
        g_k = softplus.(adjustment, smoothing_param)

        # Store current point and its fixed point evaluation for Anderson Acceleration
        push!(aa_x_history, copy(current_capacities))  # x_k
        push!(aa_g_history, copy(g_k))                # g_k = F(x_k)

        # Default to standard fixed-point iteration
        next_capacities = g_k

        # Apply Anderson Acceleration if we have enough history
        if length(aa_x_history) > 1
            # Keep only the most recent aa_memory iterations
            if length(aa_x_history) > aa_memory
                # Remove oldest entries
                popfirst!(aa_x_history)
                popfirst!(aa_g_history)
            end

            # Number of previous iterations to use
            m = length(aa_x_history) - 1

            # Form the R and S matrices from the paper
            # R consists of differences between consecutive iterates
            # S consists of differences between consecutive fixed point evaluations
            R = zeros(length(current_capacities), m)
            S = zeros(length(current_capacities), m)

            for i in 1:m
                R[:, i] = aa_x_history[i+1] - aa_x_history[i]
                S[:, i] = aa_g_history[i+1] - aa_g_history[i]
            end

            # Calculate the residual at the latest point
            # r_k = g_k - x_k (fixed point residual)
            r_k = aa_g_history[end] - aa_x_history[end]

            # Solve for the coefficients that minimize ‖S*α - r_k‖
            # Using the normal equations: (S'S + reg*I)*α = S'*r_k
            try
                # Form the matrix and right-hand side
                A = S'S 
                b = S'r_k

                # Solve for α
                α = A \ b

                # Compute the Anderson acceleration step
                # The correction is a linear combination of the differences in iterates
                correction = R * α

                # Apply mixing parameter to get the next iterate
                # x_{k+1} = (1-β)g_k + β(g_k - correction)
                #          = g_k - β*correction
                next_capacities = g_k - aa_mixing * correction

                # Apply softplus to ensure non-negative capacities
                next_capacities = softplus.(next_capacities, smoothing_param)
            catch e
                # Fallback to standard iteration if numerical issues occur
                println("Anderson acceleration failed, using standard iteration: $e")
                next_capacities = g_k
            end
        end

        push!(max_pmr_history, max_pmr)
        println("Maximum abs profit margin: $max_pmr MW")
        iter_time = time()-start

        if max_pmr < 0.5
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
                "max_pmr_history" => max_pmr_history,
                "final_results" => final_results
            )
        end

        # Log current iteration data to CSV
        open(log_file, "a") do f
            row_data = [float(iteration), max_pmr, iter_time, sys_costs]

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

        GC.gc()
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
        "max_pmr_history" => max_pmr_history,
        "final_results" => final_results
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
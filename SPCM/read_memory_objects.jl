# read_memory_objects.jl

using Printf

function read_and_process_memory_objects(file_path::String)
    # Read the file
    lines = readlines(file_path)
    
    # Parse the lines into a dictionary
    memory_objects = Dict{String, Int}()
    for line in lines
        # Skip lines starting with '#'
        if startswith(line, "#")
            continue
        end
        
        # Match the pattern "object_name: number bytes"
        match = Base.match(r"^(.*): (\d+) bytes$", line)
        if match !== nothing
            object_name = match.captures[1]
            size_in_bytes = parse(Int, match.captures[2])
            memory_objects[object_name] = size_in_bytes
        end
    end
    
    # Sort the objects by size in descending order
    sorted_objects = sort(collect(memory_objects), by = x -> x[2], rev = true)
    
    return sorted_objects
end

# Example usage
file_path = "objects_memory.txt"
sorted_objects = read_and_process_memory_objects(file_path)

# Write the sorted objects to a text file
output_file_path = "sorted_objects_memory.txt"
open(output_file_path, "w") do io
    for (object_name, size_in_bytes) in sorted_objects
        println(io, "$object_name: $(@sprintf("%.3e", size_in_bytes)) bytes")
    end
end

println("Sorted objects written to $output_file_path")

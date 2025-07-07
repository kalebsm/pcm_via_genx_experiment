open("objects_memory.txt", "w") do file
    for name in names(Main, all=true)
        if isdefined(Main, name)
            obj = getfield(Main, name)
            try
                println(file, "$name: $(Base.summarysize(obj)) bytes")
            catch
                println(file, "$name: Unable to determine size")
            end
        end
    end
end
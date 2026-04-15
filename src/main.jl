using JSON

include("timer.jl")
include("ipinstance.jl")

function main()
    if length(ARGS) != 1
        println(stderr, "Usage: julia main.jl <input_file>")
        exit(1)
    end

    input_file = ARGS[1]
    filename   = basename(input_file)

    timer = Timer()
    start!(timer)

    instance = IPInstance(input_file)
    solution, obj_value = solve!(instance)

    stop!(timer)

    output_dict = Dict{String, Any}(
        "Instance" => filename,
        "Time"     => round(get_time(timer), digits=2),
        "Result"   => isnothing(obj_value) ? "--" : round(obj_value, digits=6),
        "Solution" => isnothing(solution)  ? "--" : "OPT",
        "SolVec" => solution
    )

    println(JSON.json(output_dict))
end

main()
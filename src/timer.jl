mutable struct Timer
    start_time::UInt64
    end_time::UInt64
    is_running::Bool

    Timer() = new(0, 0, false)
end

const NANO = 1_000_000_000

function reset!(t::Timer)
    t.start_time = 0
    t.is_running = false
end

function start!(t::Timer)
    t.start_time = time_ns()
    t.is_running = true
end

function stop!(t::Timer)
    if t.is_running
        t.end_time = time_ns()
        t.is_running = false
    end
end

function get_time(t::Timer)::Float64
    elapsed = t.is_running ? (time_ns() - t.start_time) : (t.end_time - t.start_time)
    return round(elapsed / NANO, digits=4)
end

include(joinpath(@__DIR__, "..", "src", "IANeuralCore.jl"))
using .IANeuralCore
using Printf
using Serialization

function _argval(flag::String, default::String)
    idx = findfirst(==(flag), ARGS)
    if isnothing(idx) || idx == length(ARGS)
        return default
    end
    return ARGS[idx + 1]
end

layers_path = _argval("--layers", joinpath(@__DIR__, "..", "data", "quant", "layers_fp.jls"))
sens_path = _argval("--sens", joinpath(@__DIR__, "..", "data", "quant", "sensitivity.jls"))
alloc_path = _argval("--alloc", joinpath(@__DIR__, "..", "data", "quant", "allocation.jls"))
out_md = _argval("--output-md", joinpath(@__DIR__, "..", "data", "quant", "benchmark_report.md"))
out_jls = _argval("--output-jls", joinpath(@__DIR__, "..", "data", "quant", "benchmark_report.jls"))
lat_iters = parse(Int, _argval("--lat-iters", "8"))
avg_bits = parse(Float64, _argval("--avg-bits", "3.2"))

layers = load_layers(layers_path)

mixed_alloc = if isfile(alloc_path)
    load_allocation(alloc_path)
else
    sens = if isfile(sens_path)
        load_sensitivity(sens_path)
    else
        sensitivity_scan(layers; k = 0.1)
    end
    budget = Int(round(sum(s.n_params for s in sens) * avg_bits))
    bit_allocator(sens; budget_bits = budget, bit_choices = [2, 3, 4, 8])
end

mixed_alloc.feasible || error("allocation mista inviavel; ajuste budget/avg-bits")

report = compare_mixed_vs_uniform4(layers, mixed_alloc; latency_iters = lat_iters)
save_comparison_report_md(out_md, report)
open(abspath(out_jls), "w") do io
    serialize(io, report)
end

println("Benchmark comparativo concluido.")
println("report_md: $(abspath(out_md))")
println("report_jls: $(abspath(out_jls))")
@printf("MSE ratio (mixed/uniform4): %.6f\n", report.mse_ratio_mixed_over_uniform)
@printf("Latency speedup (uniform4/mixed): %.6f\n", report.latency_speedup_uniform_over_mixed)
@printf("mixed mean_mse=%.10e | uniform4 mean_mse=%.10e\n", report.mixed.mean_mse, report.uniform4.mean_mse)
@printf("mixed latency_ms=%.6f | uniform4 latency_ms=%.6f\n", report.mixed.latency_ms, report.uniform4.latency_ms)

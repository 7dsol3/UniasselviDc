include(joinpath(@__DIR__, "..", "src", "IANeuralCore.jl"))
using .IANeuralCore
using Serialization

function _argval(flag::String, default::String)
    idx = findfirst(==(flag), ARGS)
    if isnothing(idx) || idx == length(ARGS)
        return default
    end
    return ARGS[idx + 1]
end

function _parse_patterns(s::String)
    isempty(strip(s)) && return String[]
    return [String(strip(x)) for x in split(s, ",") if !isempty(strip(x))]
end

src = _argval("--source", "")
if isempty(src)
    println("Uso:")
    println("  julia bin/pipeline_real.jl --source <arquivo_ou_pasta_safetensors> [--out .\\data\\real]")
    exit(1)
end

out_dir = abspath(_argval("--out", joinpath(@__DIR__, "..", "data", "real")))
mkpath(out_dir)

layers_path = joinpath(out_dir, "layers_fp.jls")
import_report = joinpath(out_dir, "import_report.md")
sens_path = joinpath(out_dir, "sensitivity.jls")
alloc_path = joinpath(out_dir, "allocation.jls")
qmodel_path = joinpath(out_dir, "model_mixed.jls")
eval_path = joinpath(out_dir, "eval_report.jls")
bench_md = joinpath(out_dir, "benchmark_report.md")
bench_jls = joinpath(out_dir, "benchmark_report.jls")

include_pat = _parse_patterns(_argval("--include", ".*"))
avg_bits = parse(Float64, _argval("--avg-bits", "3.2"))

println("=== PIPELINE REAL WEIGHTS ===")
println("source: $(abspath(src))")
println("out:    $out_dir")

layers, infos, rep = import_hf_safetensors(
    src;
    include_patterns = include_pat,
    max_tensors = parse(Int, _argval("--max-tensors", "128")),
    total_param_cap = parse(Int, _argval("--total-param-cap", "120000000")),
)
save_layers(layers_path, layers)
save_import_report_md(import_report, infos, rep)
println("1) importacao real concluida: $(rep.tensors_selected) tensor(es)")

sens = sensitivity_scan(layers; k = 0.1)
save_sensitivity(sens_path, sens)
println("2) sensitivity_scan concluido")

budget = Int(round(sum(s.n_params for s in sens) * avg_bits))
alloc = bit_allocator(sens; budget_bits = budget, bit_choices = [2, 3, 4, 8])
alloc.feasible || error("allocation inviavel; aumente --avg-bits")
save_allocation(alloc_path, alloc)
println("3) bit_allocator concluido (used=$(alloc.total_bits), budget=$(alloc.budget_bits))")

qm = quantize_mixed(layers, alloc; model_name = "real_weights_mixed")
save_quantized_model(qmodel_path, qm)
println("4) quantize_mixed concluido")

ev = eval_unique(layers, qm)
save_eval_report(eval_path, ev)
println("5) eval_unique concluido (mean_mse=$(ev.mean_mse), compressao=$(ev.compression_ratio)x)")

cmp = compare_mixed_vs_uniform4(layers, alloc; latency_iters = parse(Int, _argval("--lat-iters", "8")))
save_comparison_report_md(bench_md, cmp)
open(bench_jls, "w") do io
    serialize(io, cmp)
end
println("6) benchmark comparativo concluido")

println("\nArquivos gerados:")
println("- $layers_path")
println("- $import_report")
println("- $sens_path")
println("- $alloc_path")
println("- $qmodel_path")
println("- $eval_path")
println("- $bench_md")
println("- $bench_jls")
println("\nPronto para chat backend unique_quant via config/default.toml (quant_model).")

include(joinpath(@__DIR__, "..", "src", "IANeuralCore.jl"))
using .IANeuralCore
using Printf

function _argval(flag::String, default::String)
    idx = findfirst(==(flag), ARGS)
    if isnothing(idx) || idx == length(ARGS)
        return default
    end
    return ARGS[idx + 1]
end

out_dir = abspath(_argval("--out", joinpath(@__DIR__, "..", "data", "quant")))
n_layers = parse(Int, _argval("--layers", "16"))
rows = parse(Int, _argval("--rows", "256"))
cols = parse(Int, _argval("--cols", "256"))
k = parse(Float64, _argval("--k", "0.1"))
budget_avg_bits = parse(Float64, _argval("--avg-bits", "3.2"))

mkpath(out_dir)
layers_path = joinpath(out_dir, "layers_fp.jls")
sens_path = joinpath(out_dir, "sensitivity.jls")
alloc_path = joinpath(out_dir, "allocation.jls")
qmodel_path = joinpath(out_dir, "model_mixed.jls")
eval_path = joinpath(out_dir, "eval_report.jls")

println("=== PIPELINE UNIQUE QUANT ===")
println("out_dir: $out_dir")

layers = generate_synthetic_layers(n_layers = n_layers, rows = rows, cols = cols)
save_layers(layers_path, layers)
println("1) layers base salvos: $layers_path")

sens = sensitivity_scan(layers; k = k)
save_sensitivity(sens_path, sens)
println("2) sensitivities salvas: $sens_path")

total_params = sum(s.n_params for s in sens)
budget = Int(round(total_params * budget_avg_bits))
alloc = bit_allocator(sens; budget_bits = budget, bit_choices = [2, 3, 4, 8])
alloc.feasible || error("allocation inviavel: aumente avg-bits")
save_allocation(alloc_path, alloc)
println("3) allocation salvo: $alloc_path")
println("   budget=$(alloc.budget_bits) used=$(alloc.total_bits)")

qm = quantize_mixed(layers, alloc; model_name = "unique_quant_pipeline")
save_quantized_model(qmodel_path, qm)
println("4) modelo quantizado salvo: $qmodel_path")

report = eval_unique(layers, qm; t0 = 0.0, t1 = 10.0)
save_eval_report(eval_path, report)
println("5) avaliacao salva: $eval_path")

println("\nResumo:")
@printf("- mean_mse: %.10e\n", report.mean_mse)
@printf("- max_mse: %.10e\n", report.max_mse)
@printf("- compressao FP16->mixed: %.4fx\n", report.compression_ratio)
@printf("- I_gato(0): %.8f | I_gato(10): %.8f\n", report.igato_start, report.igato_end)

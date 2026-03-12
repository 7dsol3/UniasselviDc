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

layers_path = _argval("--layers", joinpath(@__DIR__, "..", "data", "quant", "layers_fp.jls"))
qmodel_path = _argval("--quant", joinpath(@__DIR__, "..", "data", "quant", "model_mixed.jls"))
report_path = _argval("--output", joinpath(@__DIR__, "..", "data", "quant", "eval_report.jls"))
t0 = parse(Float64, _argval("--t0", "0.0"))
t1 = parse(Float64, _argval("--t1", "10.0"))

layers = load_layers(layers_path)
qm = load_quantized_model(qmodel_path)
report = eval_unique(layers, qm; t0 = t0, t1 = t1)
save_eval_report(report_path, report)

println("Avaliacao concluida.")
println("Saida: $(abspath(report_path))")
@printf("mean_mse: %.10e\n", report.mean_mse)
@printf("max_mse:  %.10e\n", report.max_mse)
@printf("weighted_error: %.10e\n", report.weighted_error)
@printf("compressao FP16->mixed: %.4fx\n", report.compression_ratio)
@printf("I_gato(t0)=%.8f | I_gato(t1)=%.8f\n", report.igato_start, report.igato_end)
